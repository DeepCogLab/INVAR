"""Freeze the complete 92-pair follow-up and all retrospective transfer folds."""
import collections
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
BASE = R.parent / 'baseline_v2'
MODELS = ['qwen3_8b', 'qwen25_14b_instruct']
SEEDS = [20261026, 20261028, 20261029]
REPS = dict(behavior=256, changed_fact=64, stable_fact=64, goal=256, role=64)


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    data['content_sha256'] = hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == data, 'Frozen file changed; use a new version'
    else:
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parent = json.loads((R.parent/'methods_v1/split_manifest.json').read_text(encoding='utf-8'))
    inputs = {r['prompt_id']: r for r in read(BASE/'inputs.jsonl')}
    pairs = {r['pair_id']: r for r in read(BASE/'INVAR_dataset/data/pairs.jsonl')}
    assert len(inputs) == 8666 and len(pairs) == 92
    assert sha(BASE/'inputs.jsonl') == parent['baseline_inputs_sha256']
    paths = dict(behavior='INVAR_dataset/data/quartets.jsonl', fact='fact_quartets.jsonl', goal='goal_quartets.jsonl')
    quartets = {k: read(BASE/p) for k, p in paths.items()}
    sources = sorted({p['source_id'] for p in pairs.values()})
    assert all(len(parent['provenance_sources'][pid]) == 1 for pid in inputs)
    manifests = {}
    for scope in ['all92'] + sources:
        selected_pairs = sorted(p for p, row in pairs.items() if scope == 'all92' or row['source_id'] == scope)
        selected = sorted(pid for pid in inputs if scope == 'all92' or scope in parent['provenance_sources'][pid])
        counts = collections.Counter(inputs[pid]['task_type'] for pid in selected)
        qs = {k: [q for q in rows if q['pair_id'] in selected_pairs] for k, rows in quartets.items()}
        assert all(q[s] in selected for group in qs.values() for q in group for s in 'ABCD')
        m = dict(version='INVAR_transfer_20261026_v1', status='frozen_retrospective_not_blind_confirmation',
                 scope=scope, seed=20261030, selection='all frozen pairs; no score-based selection',
                 baseline_inputs_sha256=sha(BASE/'inputs.jsonl'), selected_pairs=selected_pairs,
                 selected_pair_details=[pairs[p] for p in selected_pairs], selected_prompt_ids=selected,
                 selected_quartet_ids={k:[q['quartet_id'] for q in rows] for k, rows in qs.items()},
                 provenance_sources={pid:parent['provenance_sources'][pid] for pid in selected},
                 inputs_by_type=dict(counts), repetitions=REPS,
                 calls_per_model_method=sum(counts[k]*REPS[k] for k in counts),
                 initial_models=MODELS, model_methods={model:['baseline','anchor_sft','anchor_js'] for model in MODELS},
                 learning=dict(heldout_source='HPVBG' if scope=='all92' else scope,
                               excluded_variants=['V4_rules_table','V6_relation']),
                 private_metadata_policy='Source/truth only for split, fit grouping and scoring; inference uses original messages/menu only')
        save(R/'manifests'/f'{scope}.json', m)
        manifests[scope] = dict(path=f'manifests/{scope}.json', sha256=sha(R/'manifests'/f'{scope}.json'),
                                pairs=len(selected_pairs), prompts=len(selected), calls=m['calls_per_model_method'],
                                inputs_by_type=dict(counts))
    assert sum(m['calls'] for key,m in manifests.items() if key!='all92') == manifests['all92']['calls']
    jobs = []
    for gpu, model in enumerate(MODELS):
        jobs.append(dict(id=f'{model}_baseline_all92', kind='baseline', gpu=gpu, model=model,
                         scope='all92', method='baseline', calls=manifests['all92']['calls']))
        if model == 'qwen3_8b':
            for arm in ['anchor_sft','anchor_js']:
                jobs.append(dict(id=f'{model}_legacy_{arm}_all92', kind='legacy_evaluation', gpu=gpu,
                                 model=model, scope='all92', method=arm, calls=manifests['all92']['calls']))
        for seed in SEEDS:
            for source in sources:
                for arm in ['anchor_sft','anchor_js']:
                    jobs.append(dict(id=f'{model}_{source}_{seed}_{arm}', kind='train_and_evaluate', gpu=gpu,
                                     model=model, scope=source, method=arm, training_seed=seed,
                                     calls=manifests[source]['calls']))
    plan = dict(version='INVAR_transfer_stage_20261026_v1', status='frozen_not_yet_started',
                question='Does free-behavior equivalence training mitigate supervised reliability tradeoffs across sources, seeds and models?',
                models=MODELS, training_seeds=SEEDS, inference_seed=20261030,
                arms=['anchor_sft','anchor_js'], steps=150, rank=8, learning_rate=1e-4,
                gradient_accumulation=8, lambda_js=dict(anchor_sft=0.,anchor_js=1.),
                heldout_variants=['V4_rules_table','V6_relation'], manifests=manifests, jobs=jobs,
                training_runs=sum(j['kind']=='train_and_evaluate' for j in jobs),
                evaluation_runs=len(jobs), planned_generated_responses=sum(j['calls'] for j in jobs),
                baseline_reuse='One full92 raw baseline per model; fold scores project exact rows with parent raw SHA; no extra independent baseline claims',
                limitations=['All existing sources observed; retrospective transfer only',
                             'Source and task/menu shifts confounded; report shared-task versus source-unique tasks',
                             'Source is four protocols, not 92 independent studies',
                             'Inference repeats are not independent training seeds'],
                endpoints=['E','G','E2','G2','U_changed','U_stable','Q','role','entropy','between_condition_variation','R_descriptive','cost'],
                aggregation='Equal applicable variants within pair, equal pairs; also equal source macro; all seeds and counterexamples retained',
                no_success_stopping=True)
    save(R/'stage_manifest.json',plan)
    print(json.dumps(dict(training_runs=plan['training_runs'], evaluation_runs=len(jobs),
                          responses=plan['planned_generated_responses'], scopes=manifests)))


if __name__ == '__main__':
    main()
