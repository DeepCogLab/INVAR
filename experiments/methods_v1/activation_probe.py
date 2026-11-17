"""Fixed-hyperparameter LOO proxy screening; no sampling or correctness labels in inference."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ('baseline', 'shared_format_calibration', 'paired_svd', 'condition_protected_svd')
SEED = 20261025
RANK = 4
PROTECTED_RANK = 8
STRENGTH = .5
CALIBRATION_STRENGTH = .5
TRAINING_EDGE_POLICY = 'v2: exclude fact edges with unequal option_semantic_order; metadata used in training grouping only'


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(s) for s in f if s.strip()]


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def hashfile(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def two_candidates(p):
    """Only the legal output schema defines candidates; expected_* is never inspected."""
    if p.get('task_type') not in ('behavior', 'changed_fact', 'stable_fact', 'goal'):
        return None
    properties = p['output_schema']['properties']
    if len(properties) != 1:
        return None
    field, spec = next(iter(properties.items()))
    values = spec.get('enum')
    if values is None and spec.get('type') == 'integer':
        lo, hi = spec.get('minimum'), spec.get('maximum')
        if lo is not None and hi is not None and hi - lo == 1:
            values = [lo, hi]
    if values is None or len(values) != 2 or values[0] == values[1]:
        return None
    return field, list(values)


def calibration_key(p):
    field, values = two_candidates(p)
    return json.dumps([p['task_type'], field, values], ensure_ascii=False, separators=(',', ':'))


def observed_style(messages):
    """Fixed observable-text features only; no variant/case/pair/condition metadata."""
    text = '\n'.join(m['content'] for m in messages)
    blocks = re.split(r'【([^】]+)】', text)
    names = ('个体设定', '实验规则', '当前情况')
    styles = []
    for i in range(1, len(blocks), 2):
        name, body = blocks[i], blocks[i + 1]
        if name in names:
            kind = 'table' if re.search(r'^\s*\|', body, re.M) else ('list' if re.search(r'^\s*[-*]\s', body, re.M) else 'paragraph')
            styles.append((name, kind))
    # Identifier LETTER vocabulary is observable, exact numbers are not a group key.
    letters = sorted(set(re.findall(r'\b([A-Z])\d+\b', text)))
    return json.dumps([styles, letters], ensure_ascii=False, separators=(',', ':'))


def load_dataset(base):
    all_inputs = {p['prompt_id']: p for p in read(base / 'inputs.jsonl')}
    pairs = {p['pair_id']: p for p in read(base / 'INVAR_dataset/data/pairs.jsonl')}
    quartets = []
    for file, kind in [('INVAR_dataset/data/quartets.jsonl', 'behavior'),
                       ('fact_quartets.jsonl', 'fact'), ('goal_quartets.jsonl', 'goal')]:
        for q in read(base / file):
            q = dict(q)
            q['kind'] = kind
            q['source_id'] = pairs[q['pair_id']]['source_id']
            quartets.append(q)
    sources = collections.defaultdict(set)
    formats = collections.defaultdict(set)
    uses = collections.defaultdict(set)
    for q in quartets:
        for t in 'ABCD':
            pid = q[t]
            assert pid in all_inputs
            sources[pid].add(q['source_id'])
            uses[pid].add(q['pair_id'])
            formats[pid].add('original' if t in 'AC' else q['variant_id'])
    inputs = {pid: p for pid, p in all_inputs.items()
              if two_candidates(p) and pid in sources}
    return all_inputs, inputs, pairs, quartets, sources, formats, uses


def build_folds(pairs, quartets, sources, supported):
    """All shared prompt references close pair groups before source-level LOO."""
    parent = {pid: pid for pid in pairs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    owners = {}
    for q in quartets:
        pair = q['pair_id']
        for t in 'ABCD':
            pid = q[t]
            if pid in owners:
                union(pair, owners[pid])
            owners[pid] = pair
    groups = collections.defaultdict(list)
    for pair in pairs:
        groups[find(pair)].append(pair)
    pair_group = {p: root for root, ps in groups.items() for p in ps}
    group_sources = {g: {pairs[p]['source_id'] for p in ps} for g, ps in groups.items()}
    pair_pids = collections.defaultdict(set)
    for q in quartets:
        pair_pids[q['pair_id']].update(q[t] for t in 'ABCD')
    folds = []
    for source in sorted({p['source_id'] for p in pairs.values()}):
        test_pairs = {p for p in pairs if pairs[p]['source_id'] == source}
        train_pairs = {p for p in pairs if source not in group_sources[pair_group[p]]}
        test = sorted({pid for p in test_pairs for pid in pair_pids[p]} & set(supported))
        train = sorted({pid for p in train_pairs for pid in pair_pids[p]} & set(supported))
        assert not set(train) & set(test)
        assert all(source not in sources[pid] for pid in train)
        assert all(source in sources[pid] for pid in test)
        folds.append(dict(fold='leave_' + source + '_out', heldout_source=source,
                          train_pair_ids=sorted(train_pairs), test_pair_ids=sorted(test_pairs),
                          train_prompt_ids=train, test_prompt_ids=test))
    return dict(protocol='activation_proxy_LOSO_v1', closure='shared prompt -> pair groups; whole heldout source excluded',
                pair_groups={g: sorted(ps) for g, ps in groups.items()}, folds=folds)


def right_subspace(x, rank, seed=SEED):
    """Seeded CPU randomized uncentered SVD, avoiding a full d x d decomposition."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError('Matrix required')
    k = min(rank, *x.shape)
    if k <= 0 or not np.any(x):
        return np.zeros((x.shape[1], 0), dtype=np.float32)
    width = min(k + 8, *x.shape)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((x.shape[0], width)).astype(np.float32)
    q, _ = np.linalg.qr(x.T @ omega, mode='reduced')
    for _ in range(2):
        q, _ = np.linalg.qr(x.T @ (x @ q), mode='reduced')
    _, s, vt = np.linalg.svd(x @ q, full_matrices=False)
    keep = min(k, int(np.sum(s > max(float(s[0]) * 1e-6, 1e-8))))
    basis = q @ vt[:keep].T
    return basis.astype(np.float32)


def training_edge_allowed(a, b, inputs):
    """Prevent factual A/B rebinding from being learnt as a nuisance direction.

    This reads only TRAINING prompt metadata after membership checks in callers.
    It never reads the truth label and is not part of prediction/style detection.
    """
    if inputs is None:
        return True  # Only analytic synthetic matrix tests omit prompt metadata.
    pa, pb = inputs[a], inputs[b]
    factual = ('changed_fact', 'stable_fact')
    if pa['task_type'] in factual or pb['task_type'] in factual:
        if pa['task_type'] not in factual or pb['task_type'] not in factual:
            return False
        if 'option_semantic_order' not in pa or 'option_semantic_order' not in pb:
            raise ValueError('Fact edge missing public option-order metadata; do not infer it from expected truth')
        return pa['option_semantic_order'] == pb['option_semantic_order']
    return True


def fit_basis(train_ids, qs, features, index, protected=False, inputs=None):
    train = set(train_ids)
    # Deduplicate input-pair edges across repeated use in quartets.
    equivalent, conditions = set(), set()
    excluded_equivalent, excluded_conditions = set(), set()
    for q in qs:
        for a, b in [('A', 'B'), ('C', 'D')]:
            if q[a] in train and q[b] in train:
                target = equivalent if training_edge_allowed(q[a], q[b], inputs) else excluded_equivalent
                target.add((q[a], q[b]))
        for a, b in [('A', 'C'), ('B', 'D')]:
            if q[a] in train and q[b] in train:
                target = conditions if training_edge_allowed(q[a], q[b], inputs) else excluded_conditions
                target.add((q[a], q[b]))
    dim = features.shape[1]
    diffs = np.array([features[index[b]] - features[index[a]] for a, b in sorted(equivalent)],
                     dtype=np.float32).reshape(-1, dim)
    protected_basis = np.zeros((dim, 0), np.float32)
    if protected and conditions:
        c = np.array([features[index[b]] - features[index[a]] for a, b in sorted(conditions)], np.float32)
        protected_basis = right_subspace(c, PROTECTED_RANK)
        # Project DIFFERENCES before SVD; this is not post-hoc projection of selected singular vectors.
        original_norm = np.linalg.norm(diffs)
        diffs = diffs - (diffs @ protected_basis) @ protected_basis.T
        if np.linalg.norm(diffs) < max(1e-8, original_norm * 1e-5):
            diffs[:] = 0.
    basis = right_subspace(diffs, RANK)
    mu = np.mean(features[[index[pid] for pid in train_ids]], axis=0, dtype=np.float64).astype(np.float32)
    return mu, basis, protected_basis, dict(n_equivalence_edges=len(equivalent),
                                         n_condition_edges=len(conditions), fitted_rank=basis.shape[1],
                                         protected_rank=protected_basis.shape[1],
                                         excluded_fact_option_order_equivalence_edges=len(excluded_equivalent),
                                         excluded_fact_option_order_condition_edges=len(excluded_conditions),
                                         training_edge_policy=TRAINING_EDGE_POLICY)


def fit_calibration(train_ids, qs, inputs, baseline_scores, return_audit=False):
    """A template's paired log-odds offset; one estimate shared by both conditions."""
    train = set(train_ids)
    groups = collections.defaultdict(dict)
    excluded_option_order, eligible_edges = set(), set()
    for q in qs:
        for a, b in [('A', 'B'), ('C', 'D')]:
            pa, pb = q[a], q[b]
            if pa not in train or pb not in train:
                continue
            if not training_edge_allowed(pa, pb, inputs):
                excluded_option_order.add((pa, pb))
                continue
            eligible_edges.add((pa, pb))
            ka, kb = calibration_key(inputs[pa]), calibration_key(inputs[pb])
            if ka != kb:
                continue
            sa, sb = observed_style(inputs[pa]['messages']), observed_style(inputs[pb]['messages'])
            if sa == sb:
                continue
            # Logsoftmax removes sequence-wide constants without equalizing any true condition.
            d = log_softmax(baseline_scores[pb]) - log_softmax(baseline_scores[pa])
            d -= d.mean()
            groups[ka][(pa, pb)] = (sa, sb, d)
    result = {}
    for menu, pairs in groups.items():
        records = list(pairs.values())
        styles = sorted({s for sa, sb, _ in records for s in (sa, sb)})
        si = {s: i for i, s in enumerate(styles)}
        design = np.zeros((len(records), len(styles)), dtype=np.float64)
        targets = []
        for i, (sa, sb, d) in enumerate(records):
            design[i, si[sa]], design[i, si[sb]] = -1., 1.
            targets.append(d)
        # Minimum-norm graph solution fixes the additive gauge within each connected component.
        offsets = np.linalg.lstsq(design, np.asarray(targets), rcond=1e-8)[0]
        for style, offset in zip(styles, offsets):
            result[(menu, style)] = offset - offset.mean()
    audit = dict(training_edge_policy=TRAINING_EDGE_POLICY,
                 eligible_equivalence_edges=len(eligible_edges),
                 excluded_fact_option_order_equivalence_edges=len(excluded_option_order),
                 n_fitted_menu_style_offsets=len(result),
                 n_observed_style_difference_edges=sum(len(edges) for edges in groups.values()))
    return (result, audit) if return_audit else result


def normalized_support(p, values):
    outcome = p['outcome']
    if outcome['kind'] == 'probability':
        return np.array([float(v == outcome['target_action']) for v in values])
    return np.array(values, dtype=float) / outcome['denominator']


def w1(sa, pa, sb, pb):
    grid = np.unique(np.concatenate([sa, sb]))
    return float(sum(abs(pa[sa <= lo].sum() - pb[sb <= lo].sum()) * (hi - lo)
                     for lo, hi in zip(grid[:-1], grid[1:])))


def proxy_quartets(qs, predictions, inputs, method, fold, source):
    rows, skipped = [], collections.Counter()
    for q in qs:
        if q['source_id'] != source:
            continue
        if any(q[t] not in predictions for t in 'ABCD'):
            skipped[q['kind']] += 1
            continue
        ps = [predictions[q[t]] for t in 'ABCD']
        row = {k: v for k, v in q.items() if k not in 'ABCD'}
        row.update(method=method, fold=fold, heldout_source=source,
                   interface='HF_full_candidate_conditional_sequence_softmax_proxy', sampled_trials=0)
        if q['kind'] == 'behavior':
            xs = [normalized_support(inputs[q[t]], two_candidates(inputs[q[t]])[1]) for t in 'ABCD']
            means = [float(x @ p) for x, p in zip(xs, ps)]
            row.update(E=(w1(xs[0], ps[0], xs[1], ps[1]) + w1(xs[2], ps[2], xs[3], ps[3])) / 2,
                       R=(w1(xs[0], ps[0], xs[2], ps[2]) + w1(xs[1], ps[1], xs[3], ps[3])) / 2,
                       G=abs((means[2] - means[0]) - (means[3] - means[1])),
                       delta_original=means[2] - means[0], delta_variant=means[3] - means[1])
        else:
            correct, argmax = [], []
            for tag, probs in zip('ABCD', ps):
                p = inputs[q[tag]]
                vals = two_candidates(p)[1]
                truth = set(p['expected_actions']) if q['kind'] == 'goal' else {p['expected_answer']}
                correct.append(sum(float(pr) for val, pr in zip(vals, probs) if val in truth))
                argmax.append(float(vals[int(np.argmax(probs))] in truth))
            row.update(accuracy_probability=float(np.mean(correct)), argmax_accuracy=float(np.mean(argmax)),
                       worst_side_probability=min(correct), U=float(np.mean(correct)) if q['kind'] == 'fact' else None,
                       Q=float(np.mean(correct)) if q['kind'] == 'goal' else None)
        rows.append(row)
    return rows, dict(skipped)


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        metric = row.get('probe_type', 'goal' if row['kind'] == 'goal' else 'behavior')
        for key, value in [('all', 'all'), ('task', row['task']), ('variant_id', row['variant_id'])]:
            groups[(row['method'], metric, key, value)].append(row)
    summary = []
    for (method, typ, group, value), items in sorted(groups.items()):
        counts = collections.Counter(x['pair_id'] for x in items)
        result = dict(method=method, kind=typ, group=group, value=value,
                      n_pairs=len(counts), n_quartets=len(items),
                      interpretation='fixed two-candidate HF proxy, not V2 sampled generation')
        for metric in ('E', 'R', 'G', 'accuracy_probability', 'argmax_accuracy', 'worst_side_probability', 'U', 'Q'):
            available = [x for x in items if x.get(metric) is not None]
            if available:
                result[metric] = sum(x[metric] / counts[x['pair_id']] / len(counts) for x in available)
        summary.append(result)
    return summary


def load_renderer(base):
    spec = importlib.util.spec_from_file_location('baseline_native_renderer', base / 'INVAR_dataset/run_vllm_batch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def longest_common_prefix(sequences):
    size = min(map(len, sequences))
    for i in range(size):
        if any(seq[i] != sequences[0][i] for seq in sequences[1:]):
            return i
    return size


class HFScorer:
    def __init__(self, model_id, path, base, device='cuda:0', batch_size=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                       torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32)
        self.model.to(device).eval()
        self.renderer = load_renderer(base).prompt_text
        self.model_id, self.batch_size = model_id, batch_size
        n = self.model.config.num_hidden_layers
        self.layer_one_based = n // 2
        assert (model_id, n) in [('qwen3_8b', 36), ('qwen25_14b_instruct', 48)], (model_id, n)
        assert hasattr(self.model, 'model') and hasattr(self.model.model, 'layers')
        self.metadata = dict(model=model_id, resolved_weights=str(Path(path).resolve()),
                             weight_config_sha256=hashfile(Path(path) / 'config.json'),
                             layer_one_based=self.layer_one_based, hidden_size=self.model.config.hidden_size,
                             dtype=str(next(self.model.parameters()).dtype), transformers=__import__('transformers').__version__,
                             torch=torch.__version__, architecture=type(self.model).__name__)

    def prepare(self, p):
        field, values = two_candidates(p)
        prompt = self.renderer(self.tokenizer, p['messages'], self.model_id)
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        candidates = [json.dumps({field: v}, ensure_ascii=False, separators=(',', ':')) for v in values]
        sequences = [self.tokenizer.encode(prompt + s, add_special_tokens=False) for s in candidates]
        if any(seq[:len(prefix)] != prefix for seq in sequences):
            raise ValueError('Prompt/continuation tokenizer boundary changed; do not silently change conditional likelihood definition')
        lcp = longest_common_prefix(sequences)
        assert lcp >= len(prefix) and lcp < min(map(len, sequences))
        feature_position = lcp - 1
        assert feature_position >= 0
        context = self.model.config.max_position_embeddings
        assert max(map(len, sequences)) <= context
        return dict(pid=p['prompt_id'], field=field, values=values, candidates=candidates,
                    sequences=sequences, prompt_tokens=len(prefix), feature_position=feature_position,
                    prompt_sha256=digest(prompt), candidate_token_lengths=[len(s) - len(prefix) for s in sequences])

    def score(self, prepared, intervention=None, collect_features=False):
        """Teacher-forced sum log P(JSON continuation tokens | complete prompt), no length normalization."""
        torch = self.torch
        jobs = [(x, candidate, seq) for x in prepared for candidate, seq in enumerate(x['sequences'])]
        scores, features = {}, {}
        saved = {}
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start:start + self.batch_size]
            length = max(len(seq) for _, _, seq in batch)
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id
            ids = torch.full((len(batch), length), pad, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for i, (_, _, seq) in enumerate(batch):
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                mask[i, :len(seq)] = 1
            positions = torch.tensor([x['feature_position'] for x, _, _ in batch], device=self.device)
            indices = torch.arange(len(batch), device=self.device)
            saved.clear()

            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                selected = hidden[indices, positions]
                if collect_features:
                    saved['features'] = selected.detach().float().cpu().numpy()
                if intervention is None:
                    return output
                mu, basis = intervention
                altered = selected.float() - STRENGTH * ((selected.float() - mu) @ basis) @ basis.T
                hidden = hidden.clone()
                hidden[indices, positions] = altered.to(hidden.dtype)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            handle = self.model.model.layers[self.layer_one_based - 1].register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    # Avoid [batch, prompt_length, vocabulary] allocations: project continuation states only.
                    out = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                    selected_hidden, labels, ownership = [], [], []
                    for i, (x, candidate, seq) in enumerate(batch):
                        first = x['prompt_tokens']
                        assert first > 0
                        selected_hidden.append(out.last_hidden_state[i, first - 1:len(seq) - 1])
                        labels.extend(seq[first:])
                        ownership.extend([(x['pid'], candidate)] * (len(seq) - first))
                    hidden = torch.cat(selected_hidden, dim=0)
                    target = torch.tensor(labels, device=self.device)
                    lp = []
                    for j in range(0, len(labels), 256):
                        logits = self.model.lm_head(hidden[j:j + 256]).float()
                        values = torch.log_softmax(logits, dim=-1).gather(1, target[j:j + 256, None]).squeeze(1)
                        lp.extend(values.cpu().tolist())
                for key, value in zip(ownership, lp):
                    scores[key] = scores.get(key, 0.) + value
                if collect_features:
                    for i, (x, candidate, _) in enumerate(batch):
                        pid = x['pid']
                        feat = saved['features'][i]
                        if pid in features:
                            # Different future candidate cannot change a causal prefix representation.
                            if not np.allclose(features[pid], feat, atol=.03, rtol=.01):
                                raise ValueError('Candidate future affected feature; position or causal masking bug')
                        else:
                            features[pid] = feat.copy()
            finally:
                handle.remove()
        return {x['pid']: np.array([scores[(x['pid'], i)] for i in range(2)]) for x in prepared}, features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=HERE.parent / 'baseline_v2')
    ap.add_argument('--model', required=True, choices=['qwen3_8b', 'qwen25_14b_instruct'])
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=8, help='candidate sequences, not prompts')
    ap.add_argument('--split-manifest', type=Path, help='Optional standard manifest with folds/train_prompt_ids/test_prompt_ids')
    ap.add_argument('--screening-only', action='store_true', help='Restrict to selected_prompt_ids or selected_pairs from --split-manifest; still source LOO')
    ap.add_argument('--audit-only', action='store_true', help='Build coverage and source folds without importing torch')
    a = ap.parse_args()
    base = a.base.resolve()
    output = (a.output or HERE / 'activation_proxy' / a.model).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_inputs, inputs, pairs, qs, sources, formats, uses = load_dataset(base)
    external = None
    if a.split_manifest:
        external = json.loads(a.split_manifest.read_text(encoding='utf-8-sig'))
    if a.screening_only:
        if external is None:
            ap.error('--screening-only requires --split-manifest containing selected_prompt_ids or selected_pairs')
        selected = external.get('selected_prompt_ids')
        if selected is None:
            selected_pairs = external.get('selected_pairs', external.get('selected_pair_ids'))
            if selected_pairs is None:
                raise ValueError('No selected_prompt_ids/selected_pairs in screening manifest')
            selected = {q[t] for q in qs if q['pair_id'] in set(selected_pairs) for t in 'ABCD'}
        inputs = {pid: p for pid, p in inputs.items() if pid in set(selected)}
        assert inputs, 'Empty screening subset'
    manifest = build_folds(pairs, qs, sources, inputs)
    if a.split_manifest:
        manifest['external_manifest_sha256'] = hashfile(a.split_manifest)
        manifest['external_usage'] = 'selected scope only; source LOO folds reconstructed with complete prompt closure'
    manifest['inputs_sha256'] = hashfile(base / 'inputs.jsonl')
    manifest['manifest_sha256'] = digest(manifest)
    old = output / 'activation_split_manifest.json'
    if old.exists():
        assert json.loads(old.read_text(encoding='utf-8')) == manifest, 'Cannot mix split manifests'
    write_json(old, manifest)
    audit = dict(supported_prompts=len(inputs), total_prompts=len(all_inputs),
                 supported_types=dict(collections.Counter(p['task_type'] for p in inputs.values())),
                 folds=[dict(fold=f['fold'], heldout_source=f['heldout_source'],
                             n_train=len(f['train_prompt_ids']), n_test=len(f['test_prompt_ids'])) for f in manifest['folds']],
                 excluded_definition='Any legal output schema with other than exactly two candidates; roles excluded',
                 protocol='HF full candidate conditional sequence likelihood softmax, T=1; no top-p, no sampling, no xgrammar',
                 fixed_hyperparameters=dict(nuisance_rank=RANK, condition_rank=PROTECTED_RANK, strength=STRENGTH,
                                            calibration_strength=CALIBRATION_STRENGTH, layer='n_layers//2', seed=SEED),
                 training_edge_policy=TRAINING_EDGE_POLICY)
    write_json(output / 'coverage.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if a.audit_only:
        return
    renderer = load_renderer(base)
    path = a.weights or Path(renderer.MODEL_PATHS[a.model])
    scorer = HFScorer(a.model, path, base, a.device, a.batch_size)
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prepared = {pid: scorer.prepare(p) for pid, p in sorted(inputs.items())}
    model_meta = dict(scorer.metadata, input_sha256=manifest['inputs_sha256'], split_sha256=manifest['manifest_sha256'],
                      code_sha256=hashfile(__file__), scoring_temperature=1., top_p=None,
                      interface='HF_complete_JSON_conditional_sequence_likelihood_sum_then_two_candidate_softmax',
                      feature_position='last token of tokenized full-candidate longest common prefix',
                      intervention_position='that one token at middle block output; no intervention elsewhere',
                      correctness_labels_used_for_inference=False, candidate_eos_included=False,
                      training_edge_policy=TRAINING_EDGE_POLICY,
                      numerical_policy='BF16 backbone; FP32 per-token logsoftmax; FP64 score accumulation; CPU fitting')
    identity = output / 'identity.json'
    if identity.exists():
        assert json.loads(identity.read_text()) == model_meta, 'Output directory already contains a different identity'
    write_json(identity, model_meta)
    pids = sorted(prepared)
    index = {pid: i for i, pid in enumerate(pids)}
    cache = output / 'baseline_features.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        assert z['pids'].tolist() == pids
        assert z['identity_hash'].item() == digest(model_meta)
        features, score_matrix = z['features'], z['scores']
        baseline = {pid: score_matrix[index[pid]] for pid in pids}
    else:
        baseline, all_features = {}, {}
        started = time.time()
        # Chunked durable progress; no generated samples or artificial replicate records.
        sharddir = output / 'baseline_feature_shards'
        sharddir.mkdir(exist_ok=True)
        for start in range(0, len(pids), 64):
            chunk = pids[start:start + 64]
            shard = sharddir / (str(start).zfill(6) + '.npz')
            if shard.exists():
                z = np.load(shard, allow_pickle=False)
                assert z['pids'].tolist() == chunk and z['identity_hash'].item() == digest(model_meta)
                scores = {pid: z['scores'][i] for i, pid in enumerate(chunk)}
                feats = {pid: z['features'][i] for i, pid in enumerate(chunk)}
            else:
                scores, feats = scorer.score([prepared[pid] for pid in chunk], collect_features=True)
                np.savez_compressed(shard, pids=np.array(chunk), scores=np.stack([scores[pid] for pid in chunk]),
                                    features=np.stack([feats[pid] for pid in chunk]), identity_hash=np.array(digest(model_meta)))
            baseline.update(scores)
            all_features.update(feats)
            write_json(output / 'progress.json', dict(stage='baseline', done=len(baseline), total=len(pids),
                                                      elapsed_seconds=time.time() - started))
        features = np.stack([all_features[pid] for pid in pids])
        score_matrix = np.stack([baseline[pid] for pid in pids])
        np.savez_compressed(cache, pids=np.array(pids), features=features, scores=score_matrix,
                            identity_hash=np.array(digest(model_meta)))
    all_rows = []
    for fold in manifest['folds']:
        fid, source = fold['fold'], fold['heldout_source']
        train, test = fold['train_prompt_ids'], fold['test_prompt_ids']
        assert not set(train) & set(test)
        fdir = output / fid
        fdir.mkdir(exist_ok=True)
        biases, calibration_audit = fit_calibration(train, qs, inputs, baseline, return_audit=True)
        write_json(fdir / 'calibration_fit_audit.json', dict(calibration_audit, train_ids_hash=digest(train),
                         source=source, train_sources=sorted({s for pid in train for s in sources[pid]})))
        write_json(fdir / 'calibration.json', [dict(menu=k[0], observed_style=k[1], offset=v.tolist()) for k, v in sorted(biases.items())])
        baseline_test = {pid: baseline[pid] for pid in test}
        calibrated, hit = {}, {}
        for pid in test:
            fmt = observed_style(inputs[pid]['messages'])
            bias = biases.get((calibration_key(inputs[pid]), fmt))
            hit[pid] = bias is not None
            calibrated[pid] = baseline[pid] - CALIBRATION_STRENGTH * (bias if bias is not None else 0.)
        score_sets = {'baseline': baseline_test, 'shared_format_calibration': calibrated}
        for method, protected in [('paired_svd', False), ('condition_protected_svd', True)]:
            mu, basis, protect, fit_audit = fit_basis(train, qs, features, index, protected, inputs=inputs)
            np.savez_compressed(fdir / (method + '_parameters.npz'), mu=mu, basis=basis, protected_basis=protect,
                                train_ids=np.array(train), train_ids_hash=np.array(digest(train)))
            write_json(fdir / (method + '_fit.json'), dict(fit_audit, train_ids_hash=digest(train),
                         source=source, strength=STRENGTH, train_sources=sorted({s for pid in train for s in sources[pid]}),
                         basis_protected_overlap=float(np.linalg.norm(protect.T @ basis)) if protect.size else 0.))
            cached_score = fdir / (method + '_scores.npz')
            if cached_score.exists():
                z = np.load(cached_score, allow_pickle=False)
                assert z['pids'].tolist() == test
                assert z['identity_hash'].item() == digest(model_meta)
                scores = {pid: z['scores'][i] for i, pid in enumerate(test)}
            else:
                edit = (torch.tensor(mu, device=a.device), torch.tensor(basis, device=a.device))
                scores = {}
                for start in range(0, len(test), 64):
                    scores.update(scorer.score([prepared[pid] for pid in test[start:start + 64]], intervention=edit)[0])
                    write_json(output / 'progress.json', dict(stage=method, fold=fid, done=len(scores), total=len(test)))
                np.savez_compressed(cached_score, pids=np.array(test), scores=np.stack([scores[pid] for pid in test]),
                                    identity_hash=np.array(digest(model_meta)))
            score_sets[method] = scores
        for method, scores in score_sets.items():
            predictions = {pid: np.exp(log_softmax(s)) for pid, s in scores.items()}
            outfile = fdir / (method + '_prompt_probabilities.jsonl')
            with outfile.open('w', encoding='utf-8') as f:
                for pid in test:
                    p = inputs[pid]
                    truth = p.get('expected_actions', [p.get('expected_answer')])
                    probs = predictions[pid]
                    vals = prepared[pid]['values']
                    accuracy = None if p['task_type'] == 'behavior' else sum(float(prob) for val, prob in zip(vals, probs) if val in truth)
                    row = dict(prompt_id=pid, method=method, model=a.model, fold=fid, heldout_source=source,
                               prompt_sources=sorted(sources[pid]), task_type=p['task_type'],
                               variants=sorted(formats[pid]), candidate_values=vals,
                               candidate_json=prepared[pid]['candidates'],
                               sequence_log_scores=(baseline[pid] if method == 'shared_format_calibration' else scores[pid]).tolist(),
                               method_log_scores=scores[pid].tolist(),
                               candidate_probabilities=probs.tolist(), expected_accuracy_probability=accuracy,
                               argmax_correct=None if accuracy is None else vals[int(np.argmax(probs))] in truth,
                               output_entropy=float(-sum(v * math.log(max(v, 1e-300)) for v in probs)),
                               calibration_applied=hit[pid] if method == 'shared_format_calibration' else False,
                               interface=model_meta['interface'], sampled_trials=0,
                               feature_position=prepared[pid]['feature_position'],
                               candidate_token_lengths=prepared[pid]['candidate_token_lengths'])
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows, skipped = proxy_quartets(qs, predictions, inputs, method, fid, source)
            all_rows.extend(rows)
            write_json(fdir / (method + '_coverage.json'), dict(scored_prompts=len(test), scored_quartets=len(rows),
                                                               skipped_quartets=skipped, calibration_hits=sum(hit.values())))
    with (output / 'proxy_quartets.jsonl').open('w', encoding='utf-8') as f:
        f.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in all_rows))
    write_json(output / 'proxy_summary.json', summarize(all_rows))
    write_json(output / 'complete.json', dict(status='complete', model=a.model, methods=METHODS,
                 source_folds=len(manifest['folds']), scored_unique_prompts=len(inputs), proxy_quartet_rows=len(all_rows),
                 identity_hash=digest(model_meta), input_sha256=manifest['inputs_sha256'],
                 interpretation='deterministic two-candidate likelihood proxy; no generated samples or V2 scores'))
    print(json.dumps(dict(status='complete', output=str(output))), flush=True)


if __name__ == '__main__':
    main()
