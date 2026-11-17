"""Real HF constrained generation with a frozen one-position prefill activation intervention."""
from __future__ import annotations

import argparse
import collections
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import activation_probe as proxy

HERE = Path(__file__).resolve().parent
METHODS = ('baseline', 'paired_svd', 'condition_protected_svd')
BASE_SEED = 20261025
REPLICATES = 32
STRENGTH = .5


class CandidateTrie:
    """Exact legal suffix token lists; EOS can occur only after a complete JSON candidate."""
    def __init__(self, suffixes, eos, pad=None):
        self.eos = int(eos)
        self.pad = self.eos if pad is None else int(pad)
        self.root = {}
        self.terminal = set()
        if len(suffixes) != 2 or len({tuple(s) for s in suffixes}) != 2:
            raise ValueError('Exactly two distinct suffixes required')
        for suffix in suffixes:
            if not suffix or self.eos in suffix:
                raise ValueError('Empty suffix or EOS embedded in legal candidate')
            node = self.root
            for token in suffix:
                node = node.setdefault(int(token), {})
            self.terminal.add(tuple(int(t) for t in suffix))

    def allowed(self, generated):
        generated = tuple(int(t) for t in generated)
        if self.eos in generated:
            before, after = generated[:generated.index(self.eos)], generated[generated.index(self.eos):]
            if before not in self.terminal or any(t not in (self.eos, self.pad) for t in after):
                raise ValueError('EOS before a complete candidate or non-EOS after termination')
            return [self.eos]
        node = self.root
        for token in generated:
            if token not in node:
                raise ValueError('Generated token is outside the complete-candidate trie')
            node = node[token]
        result = sorted(node)
        if generated in self.terminal:
            result.append(self.eos)
        if not result:
            raise ValueError('No legal next token')
        return result

    def validate(self, suffix):
        suffix = [int(t) for t in suffix]
        if self.eos not in suffix:
            return False, 'missing_eos'
        stop = suffix.index(self.eos)
        if tuple(suffix[:stop]) not in self.terminal:
            return False, 'not_a_complete_candidate'
        if any(t not in (self.eos, self.pad) for t in suffix[stop + 1:]):
            return False, 'non_padding_after_eos'
        return True, None


def generation_trial_id(model, fold, pid, method, replicate):
    return proxy.digest([model, fold, pid, method, replicate])[:32]


def chunk_seed(model, fold, pid, start):
    # Method deliberately excluded; matching batches consume matching initial RNG streams.
    return int(proxy.digest([BASE_SEED, model, fold, pid, start])[:8], 16) % (2**31 - 1)


def should_edit_prefill(seen_prefill, length, prefix_length):
    """CPU-testable position policy: exactly one full prefill, subsequent decode is length 1."""
    if not seen_prefill:
        if length != prefix_length:
            raise ValueError('First call is not the full fixed-prefix prefill')
        return True
    if length != 1:
        raise ValueError('Unexpected second prefill/chunk; cannot silently edit another position')
    return False


def assert_scope(base, proxy_dir, sources_selected):
    all_inputs, inputs, pairs, qs, sources, formats, uses = proxy.load_dataset(base)
    split_path = proxy_dir / 'activation_split_manifest.json'
    identity_path = proxy_dir / 'identity.json'
    split = json.loads(split_path.read_text(encoding='utf-8-sig'))
    identity = json.loads(identity_path.read_text(encoding='utf-8-sig'))
    assert identity.get('training_edge_policy') == proxy.TRAINING_EDGE_POLICY, 'Only corrected v2 order-matched proxy parameters are eligible'
    assert proxy.hashfile(base / 'inputs.jsonl') == split['inputs_sha256'] == identity['input_sha256']
    chosen = [f for f in split['folds'] if f['heldout_source'] in sources_selected]
    assert len(chosen) == len(sources_selected), 'Unknown or duplicated source fold requested'
    for fold in chosen:
        train, test = set(fold['train_prompt_ids']), set(fold['test_prompt_ids'])
        source = fold['heldout_source']
        assert not train & test
        assert all(pid in inputs and source not in sources[pid] for pid in train)
        assert all(pid in inputs and source in sources[pid] for pid in test)
        # Reconstruct source closure from ALL dataset references, then restrict to this proxy's scope.
        reconstructed = proxy.build_folds(pairs, qs, sources, train | test)
        expected = next(f for f in reconstructed['folds'] if f['heldout_source'] == source)
        assert expected['train_prompt_ids'] == fold['train_prompt_ids']
        assert expected['test_prompt_ids'] == fold['test_prompt_ids']
    return inputs, qs, sources, formats, chosen, split, identity


def load_intervention(proxy_dir, fold, method, hidden_size):
    folder = proxy_dir / fold['fold']
    parameter_path = folder / (method + '_parameters.npz')
    fit_path = folder / (method + '_fit.json')
    audit = json.loads(fit_path.read_text(encoding='utf-8-sig'))
    z = np.load(parameter_path, allow_pickle=False)
    train = z['train_ids'].tolist()
    assert train == fold['train_prompt_ids']
    assert z['train_ids_hash'].item() == proxy.digest(train) == audit['train_ids_hash']
    assert audit['source'] == fold['heldout_source']
    assert fold['heldout_source'] not in audit['train_sources']
    assert audit['strength'] == STRENGTH
    assert audit.get('training_edge_policy') == proxy.TRAINING_EDGE_POLICY
    mu, basis = z['mu'], z['basis']
    assert mu.shape == (hidden_size,) and basis.shape[0] == hidden_size and basis.shape[1] <= 4
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(basis))
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-4)
    hashes = dict(parameters_sha256=proxy.hashfile(parameter_path), fit_sha256=proxy.hashfile(fit_path),
                  train_ids_sha256=proxy.digest(train))
    return mu, basis, hashes


def sample_chunk(scorer, prepared, count, seed, intervention):
    """HF generate performs true sampling. The fixed prefix and suffix tokens are both recorded."""
    import torch
    from transformers import GenerationConfig

    tokenizer, model = scorer.tokenizer, scorer.model
    eos = tokenizer.eos_token_id
    if eos is None or not isinstance(eos, int):
        raise ValueError('A single tokenizer EOS token is required by this exact-candidate protocol')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    pos = prepared['feature_position']
    prefix = prepared['sequences'][0][:pos + 1]
    assert all(seq[:pos + 1] == prefix for seq in prepared['sequences'])
    suffixes = [seq[pos + 1:] for seq in prepared['sequences']]
    trie = CandidateTrie(suffixes, eos, pad)
    ids = torch.tensor([prefix] * count, device=scorer.device, dtype=torch.long)
    attention_mask = torch.ones_like(ids)
    state = dict(seen_prefill=False, prefill_calls=0, decode_calls=0, edited_positions=0,
                 max_edit_l2=0., feature_position=pos, prefix_length=len(prefix))

    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != count:
            raise ValueError('Unexpected HF block output layout/batch expansion')
        prefill = should_edit_prefill(state['seen_prefill'], hidden.shape[1], len(prefix))
        if not prefill:
            state['decode_calls'] += 1
            return output
        state['seen_prefill'] = True
        state['prefill_calls'] += 1
        if intervention is None:
            return output
        mu, basis = intervention
        selected = hidden[:, pos].float()
        edited = selected - STRENGTH * ((selected - mu) @ basis) @ basis.T
        state['max_edit_l2'] = float(torch.linalg.vector_norm(edited - selected, dim=-1).max().item())
        state['edited_positions'] += count
        hidden = hidden.clone()
        hidden[:, pos] = edited.to(hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def allowed(batch_id, complete_ids):
        tokens = complete_ids.tolist()
        if tokens[:len(prefix)] != prefix:
            raise ValueError('Prefix token IDs changed during generation')
        return trie.allowed(tokens[len(prefix):])

    config = GenerationConfig(do_sample=True, temperature=.7, top_p=.95, top_k=0,
                              repetition_penalty=1., max_new_tokens=128, num_beams=1,
                              eos_token_id=eos, pad_token_id=pad, bos_token_id=tokenizer.bos_token_id,
                              use_cache=True, return_dict_in_generate=True,
                              output_scores=False, output_hidden_states=False)
    # Some installed Qwen versions can avoid all-prefix LM-head logits at prefill.
    extra = {'logits_to_keep': 1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
    handle = model.model.layers[scorer.layer_one_based - 1].register_forward_hook(hook)
    device_index = torch.device(scorer.device).index
    if device_index is None and scorer.device.startswith('cuda'):
        device_index = torch.cuda.current_device()
    rng_devices = [device_index] if scorer.device.startswith('cuda') else []
    begun = time.time()
    try:
        with torch.random.fork_rng(devices=rng_devices), torch.inference_mode():
            torch.manual_seed(seed)
            if rng_devices:
                torch.cuda.manual_seed(seed)
            result = model.generate(input_ids=ids, attention_mask=attention_mask, generation_config=config,
                                    prefix_allowed_tokens_fn=allowed, **extra)
    finally:
        handle.remove()
    assert state['prefill_calls'] == 1, state
    assert state['edited_positions'] == (count if intervention is not None else 0), state
    records = []
    for seq in result.sequences.tolist():
        assert seq[:len(prefix)] == prefix
        generated = seq[len(prefix):]
        valid, reason = trie.validate(generated)
        stop = generated.index(eos) if eos in generated else len(generated)
        trailing = generated[stop + 1:] if stop < len(generated) else []
        if any(t not in (pad, eos) for t in trailing):
            valid, reason = False, 'non_padding_after_eos'
        actual = prefix + generated[:stop]
        candidate = next((i for i, expected in enumerate(prepared['sequences']) if actual == expected), None)
        if candidate is None:
            valid, reason = False, 'token_sequence_not_equal_to_either_legal_candidate'
        response_tokens = actual[prepared['prompt_tokens']:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=False)
        try:
            obj = json.loads(response)
            if set(obj) != {prepared['field']} or obj[prepared['field']] not in prepared['values']:
                valid, reason = False, 'decoded_json_not_in_public_menu'
        except Exception:
            valid, reason = False, 'not_json'
        records.append(dict(response_text=response, status='ok' if valid else 'invalid',
                            finish_reason='stop' if valid else ('length' if eos not in generated else reason),
                            generated_suffix_text=tokenizer.decode(generated[:stop], skip_special_tokens=False),
                            generated_suffix_token_ids=generated[:stop], generated_tokens_including_eos=stop + int(stop < len(generated)),
                            fixed_json_prefix_text=tokenizer.decode(prefix[prepared['prompt_tokens']:], skip_special_tokens=False),
                            fixed_json_prefix_token_ids=prefix[prepared['prompt_tokens']:],
                            input_tokens=len(prefix), original_native_prompt_tokens=prepared['prompt_tokens'],
                            candidate_index=candidate, candidate_value=None if candidate is None else prepared['values'][candidate],
                            valid_json_and_menu=valid, invalid_reason=reason))
    return records, dict(state, seconds=time.time() - begun, seed=seed)


def score_actual_generations(raw_paths, inputs, qs, folds, output):
    """Empirical frequencies from real saved rows; no proxy probabilities are sampled."""
    results = []
    for fold in folds:
        for method in METHODS:
            path = raw_paths[(fold['fold'], method)]
            rows = proxy.read(path)
            seen = set()
            values = collections.defaultdict(list)
            for row in rows:
                key = (row['prompt_id'], row['replicate'])
                assert key not in seen and row['status'] == 'ok' and row['finish_reason'] == 'stop'
                seen.add(key)
                values[row['prompt_id']].append(row['candidate_index'])
            assert set(values) == set(fold['test_prompt_ids'])
            assert all(len(v) == REPLICATES for v in values.values())
            probabilities = {pid: np.bincount(v, minlength=2) / len(v) for pid, v in values.items()}
            metrics, skipped = proxy.proxy_quartets(qs, probabilities, inputs, method, fold['fold'], fold['heldout_source'])
            for row in metrics:
                row['interface'] = 'HF_actual_trie_constrained_generation_fixed_common_JSON_prefix'
                row['sampled_trials_per_prompt'] = REPLICATES
                row['sampled_trials'] = REPLICATES
                row['uncertainty'] = 'point estimates only; no confidence interval or significance claim'
                row.pop('argmax_accuracy', None)
            results.extend(metrics)
            proxy.write_json(path.parent / (method + '_scoring_coverage.json'), dict(n_prompts=len(values),
                             n_responses=len(rows), n_quartets=len(metrics), skipped_quartets=skipped))
    with (output / 'generation_quartets.jsonl').open('w', encoding='utf-8') as f:
        f.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in results))
    summary = proxy.summarize(results)
    for row in summary:
        row['interpretation'] = '32 true HF constrained generations per prompt; matched HF baseline; development screening'
        row['accuracy_probability_name'] = 'empirical sampled accuracy; not sequence-likelihood proxy'
    proxy.write_json(output / 'generation_summary.json', summary)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=HERE.parent / 'baseline_v2')
    ap.add_argument('--model', choices=['qwen3_8b', 'qwen25_14b_instruct'], default='qwen3_8b')
    ap.add_argument('--proxy-dir', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, choices=[4, 8], default=4)
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument('--fold', default='HPVBG', help='One source ID; default HPVBG only')
    scope.add_argument('--all-folds', action='store_true')
    ap.add_argument('--audit-only', action='store_true')
    a = ap.parse_args()
    base = a.base.resolve()
    proxy_dir = (a.proxy_dir or HERE / 'activation_proxy_screen' / a.model).resolve()
    output = (a.output or HERE / 'activation_generation_screen' / a.model / ('all_folds' if a.all_folds else a.fold)).resolve()
    split = json.loads((proxy_dir / 'activation_split_manifest.json').read_text(encoding='utf-8-sig'))
    selected = {f['heldout_source'] for f in split['folds']} if a.all_folds else {a.fold}
    inputs, qs, sources, formats, folds, split, proxy_identity = assert_scope(base, proxy_dir, selected)
    assert proxy_identity['model'] == a.model
    assert proxy_identity['layer_one_based'] in (18, 24)
    dim = proxy_identity['hidden_size']
    parameters = {}
    for fold in folds:
        for method in METHODS[1:]:
            parameters[(fold['fold'], method)] = load_intervention(proxy_dir, fold, method, dim)
    output.mkdir(parents=True, exist_ok=True)
    scope_report = dict(model=a.model, sources=sorted(selected), folds=folds,
                        n_unique_prompts=len({pid for fold in folds for pid in fold['test_prompt_ids']}),
                        replicates=REPLICATES, methods=METHODS,
                        n_responses=sum(len(f['test_prompt_ids']) for f in folds) * REPLICATES * len(METHODS),
                        proxy_split_sha256=proxy.hashfile(proxy_dir / 'activation_split_manifest.json'),
                        proxy_identity_sha256=proxy.hashfile(proxy_dir / 'identity.json'),
                        parameters={fid + '/' + method: value[2] for (fid, method), value in parameters.items()})
    proxy.write_json(output / 'scope_audit.json', scope_report)
    print(json.dumps({k: v for k, v in scope_report.items() if k not in ('folds', 'parameters')}, ensure_ascii=False), flush=True)
    if a.audit_only:
        return
    renderer = proxy.load_renderer(base)
    weights = a.weights or Path(renderer.MODEL_PATHS[a.model])
    scorer = proxy.HFScorer(a.model, weights, base, a.device, a.batch_size)
    assert scorer.metadata['weight_config_sha256'] == proxy_identity['weight_config_sha256']
    assert scorer.layer_one_based == proxy_identity['layer_one_based']
    import torch
    ids = sorted({pid for fold in folds for pid in fold['test_prompt_ids']})
    prepared = {pid: scorer.prepare(inputs[pid]) for pid in ids}
    identity = dict(protocol='INVAR_HF_actual_activation_generation_screen_v1', model=a.model,
                    code_sha256=proxy.hashfile(__file__), proxy_module_sha256=proxy.hashfile(proxy.__file__),
                    proxy_identity_sha256=scope_report['proxy_identity_sha256'], proxy_split_sha256=scope_report['proxy_split_sha256'],
                    parameters=scope_report['parameters'], model_metadata=scorer.metadata,
                    input_sha256=proxy.hashfile(base / 'inputs.jsonl'), source_ids=sorted(selected),
                    methods=METHODS, batch_size=a.batch_size, replicates=REPLICATES, base_seed=BASE_SEED,
                    chunk_seed_policy='sha256(base_seed,model,fold,prompt_id,chunk_replicate_start); excludes method',
                    seed_caveat='one real torch RNG stream per chunk; rows are not separately seeded; paired initial streams across methods',
                    temperature=.7, top_p=.95, top_k=0, max_new_tokens=128, use_cache=True,
                    repetition_penalty=1., strength=STRENGTH,
                    decoder='HF generate exact two-candidate token trie; common JSON token prefix fixed as input',
                    vllm_equivalent=False, correctness_labels_used_in_inference=False,
                    intervention='only LCP final token, one middle-block prefill call; decode untouched',
                    generated_response='disclosed fixed common JSON prefix + actual sampled suffix')
    identity_path = output / 'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Refuse to mix generation identities'
    proxy.write_json(identity_path, identity)
    identity_hash = proxy.digest(identity)
    raw_paths = {}
    start_time = time.time()
    for fold in folds:
        fid, source = fold['fold'], fold['heldout_source']
        folder = output / fid
        folder.mkdir(exist_ok=True)
        for method in METHODS:
            path = folder / (method + '.jsonl')
            raw_paths[(fid, method)] = path
            completed = set()
            if path.exists():
                for row in proxy.read(path):
                    assert row['identity_sha256'] == identity_hash
                    assert row['prompt_id'] in fold['test_prompt_ids']
                    assert 0 <= row['replicate'] < REPLICATES
                    assert row['model'] == a.model and row['method'] == method and row['fold'] == fid
                    assert row['trial_id'] == generation_trial_id(a.model, fid, row['prompt_id'], method, row['replicate'])
                    key = (row['prompt_id'], row['replicate'])
                    assert key not in completed and row['status'] == 'ok'
                    completed.add(key)
            edit = None
            parameter_hashes = None
            if method != 'baseline':
                mu, basis, parameter_hashes = parameters[(fid, method)]
                edit = (torch.tensor(mu, device=a.device), torch.tensor(basis, device=a.device))
            with path.open('a', encoding='utf-8') as f, (folder / (method + '_chunks.jsonl')).open('a', encoding='utf-8') as chunks:
                for pid in fold['test_prompt_ids']:
                    for rep_start in range(0, REPLICATES, a.batch_size):
                        reps = list(range(rep_start, min(REPLICATES, rep_start + a.batch_size)))
                        present = [(pid, r) in completed for r in reps]
                        if all(present):
                            continue
                        seed = chunk_seed(a.model, fid, pid, rep_start)
                        generated, chunk_audit = sample_chunk(scorer, prepared[pid], len(reps), seed, edit)
                        out = []
                        for pos, (rep, generated_row) in enumerate(zip(reps, generated)):
                            if present[pos]:
                                continue
                            row = dict(generated_row, trial_id=generation_trial_id(a.model, fid, pid, method, rep),
                                       model=a.model, method=method, fold=fid, heldout_source=source,
                                       prompt_id=pid, prompt_sources=sorted(sources[pid]), replicate=rep, batch='activation_generation',
                                       sampling_seed=seed, chunk_replicate_start=rep_start, chunk_position=pos,
                                       identity_sha256=identity_hash, parameters=parameter_hashes,
                                       candidate_values=prepared[pid]['values'], interface=identity['decoder'])
                            out.append(row)
                        f.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in out))
                        f.flush()
                        chunks.write(json.dumps(dict(chunk_audit, model=a.model, fold=fid, method=method,
                                           prompt_id=pid, chunk_replicate_start=rep_start, n_rows_written=len(out)), ensure_ascii=False) + '\n')
                        chunks.flush()
                        assert all(row['status'] == 'ok' for row in out), 'Illegal/truncated rows retained; validation failed'
                        completed.update((pid, row['replicate']) for row in out)
                        proxy.write_json(output / 'progress.json', dict(model=a.model, fold=fid, method=method,
                                          done_rows=len(completed), expected_rows=len(fold['test_prompt_ids']) * REPLICATES,
                                          elapsed_seconds=time.time() - start_time))
            assert len(completed) == len(fold['test_prompt_ids']) * REPLICATES
    scores = score_actual_generations(raw_paths, inputs, qs, folds, output)
    proxy.write_json(output / 'complete.json', dict(status='complete', identity_sha256=identity_hash,
                     response_rows=scope_report['n_responses'], quartet_rows=len(scores),
                     raw_sha256={str(path.relative_to(output)): proxy.hashfile(path) for path in raw_paths.values()},
                     final_interpretation='32 actual HF trie constrained draws; fixed 12-pair development subset; no vLLM equivalence claim'))
    print(json.dumps(dict(status='complete', output=str(output))), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
