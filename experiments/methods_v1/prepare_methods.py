"""Freeze a score-independent, source/task/role-stratified development screen."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'baseline_v2'


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    inputs = {r['prompt_id']: r for r in read(BASE / 'inputs.jsonl')}
    pairs = {r['pair_id']: r for r in read(BASE / 'INVAR_dataset/data/pairs.jsonl')}
    paths = {'behavior': BASE / 'INVAR_dataset/data/quartets.jsonl',
             'fact': BASE / 'fact_quartets.jsonl', 'goal': BASE / 'goal_quartets.jsonl'}
    qs = {kind: read(path) for kind, path in paths.items()}
    refs = collections.defaultdict(set)
    for group in qs.values():
        for q in group:
            for side in 'ABCD':
                refs[q[side]].add(q['pair_id'])
    for pid, row in inputs.items():
        if row.get('pair_id'):
            refs[pid].add(row['pair_id'])
        for use in row.get('uses', []):
            refs[pid].add(use['pair_id'])
    sources = {pid: sorted({pairs[p]['source_id'] for p in ids}) for pid, ids in refs.items()}
    assert set(inputs) <= sources.keys(), 'Unresolved provenance'
    strata = collections.defaultdict(list)
    for pair_id, pair in pairs.items():
        strata[(pair['source_id'], pair['task'], pair['role'])].append(pair_id)
    selected_pairs = sorted(min(ids, key=lambda p: digest(['invar_methods_screen_20261025', p]))
                            for ids in strata.values())
    selected_qs = {kind: [q for q in group if q['pair_id'] in selected_pairs] for kind, group in qs.items()}
    selected_ids = {q[s] for group in selected_qs.values() for q in group for s in 'ABCD'}
    selected_ids |= {pid for pid, r in inputs.items() if r['task_type'] == 'role'}
    assert all(set(q[s] for s in 'ABCD') <= selected_ids for group in selected_qs.values() for q in group)
    all_sources = sorted({p['source_id'] for p in pairs.values()})
    folds = []
    for source in all_sources:
        test = sorted(pid for pid in selected_ids if source in sources[pid])
        train = sorted(pid for pid in selected_ids if source not in sources[pid])
        assert not set(test) & set(train)
        folds.append(dict(heldout_source=source, train_prompt_ids=train, test_prompt_ids=test,
                          train_sources=sorted({s for pid in train for s in sources[pid]})))
    reps = dict(behavior=32, changed_fact=8, stable_fact=8, goal=16, role=16)
    counts = collections.Counter(inputs[pid]['task_type'] for pid in selected_ids)
    manifest = dict(version='INVAR_methods_development_20261025_v1',
        status='frozen_development_screen_not_blind_confirmation',
        selection='one pair per source/task/role by sha256; no model scores used',
        baseline_inputs_sha256=hashlib.sha256((BASE / 'inputs.jsonl').read_bytes()).hexdigest(),
        selected_pairs=selected_pairs, selected_prompt_ids=sorted(selected_ids),
        selected_quartet_ids={k: [q['quartet_id'] for q in v] for k, v in selected_qs.items()},
        selected_pair_details=[pairs[p] for p in selected_pairs],
        provenance_sources=sources, source_folds=folds, repetitions=reps,
        inputs_by_type=dict(counts), calls_per_model_method=sum(counts[k] * reps[k] for k in counts),
        seed=20261025,
        prompt_methods=['baseline', 'generic', 'social_audit', 'ledger_original', 'ledger_only'],
        activation_methods=['baseline', 'shared_format_calibration', 'paired_svd', 'condition_protected_svd'],
        initial_models=['qwen3_8b', 'qwen25_14b_instruct', 'glm4_9b'],
        learning=dict(initial_model='qwen3_8b', heldout_source='HPVBG', excluded_variants=['V4_rules_table', 'V6_relation'],
                      modes=['sft', 'sft_consistency'], rank=8, learning_rate=1e-4, steps=150),
        private_metadata_policy='IDs/provenance/truth used for sampling, fit grouping and scoring only; never injected in inference',
        interface_policy='HF sequence likelihood proxy must not be subtracted from vLLM sampled baseline')
    manifest['content_sha256'] = digest(manifest)
    path = ROOT / 'split_manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == manifest, 'Frozen manifest differs; create a new version'
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('selected_pairs', 'inputs_by_type', 'calls_per_model_method', 'content_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
