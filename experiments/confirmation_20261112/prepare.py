"""Freeze all scientific inputs before any confirmation inference."""
import collections
import shutil
from common import R, MODELS, digest, jl, save, sha, write_jl
from parameter_cases import generate


def main():
    assert not (R/'freeze.json').exists(), 'Do not overwrite a frozen experiment'
    baseline = R.parent/'baseline_v2'; old = R.parent/'shortpaper_20261111'
    frozen = R/'frozen'; frozen.mkdir(exist_ok=True)
    inputs = {p['prompt_id']: p for p in jl(baseline/'inputs.jsonl')}
    pairs = {p['pair_id']: p for p in jl(baseline/'INVAR_dataset/data/pairs.jsonl')}
    bq = [q for q in jl(baseline/'INVAR_dataset/data/quartets.jsonl') if q['variant_id'] in ['V1_table', 'V4_rules_table']]
    behavior = {q[k] for q in bq for k in 'ABCD'}
    goals = {pid for pid, p in inputs.items() if p['task_type'] == 'goal'}
    assert len(behavior) == 429 and len(goals) == 1410 and len(bq) == 184
    assert len({q['pair_id'] for q in bq}) == 92
    ps = []; labels = []
    for pid in sorted(behavior | goals):
        p = inputs[pid]
        ps.append({k: p[k] for k in ['prompt_id', 'task_type', 'messages', 'output_schema']})
        fields = ['prompt_id', 'task_type', 'pair_id', 'task', 'variant_id', 'condition', 'policy', 'expected_actions', 'action_space', 'outcome']
        label = {k: p[k] for k in fields if k in p}
        if p['task_type'] == 'behavior':
            # Behavior inputs are content-deduplicated and may serve several pairs.
            uses = p['uses']; sources = {pairs[u['pair_id']]['source_id'] for u in uses}
            task_names = {pairs[u['pair_id']]['task'] for u in uses}
            assert len(sources) == len(task_names) == 1
            label.update(source_id=next(iter(sources)), task=next(iter(task_names)), uses=uses)
        else: label['source_id'] = pairs[p['pair_id']]['source_id']
        labels.append(label)
    canonical = frozen/'canonical'; canonical.mkdir()
    write_jl(canonical/'model_inputs.jsonl', ps); write_jl(canonical/'evaluation.jsonl', labels)
    write_jl(canonical/'behavior_quartets.jsonl', bq)
    gq = [dict(q, source_id=pairs[q['pair_id']]['source_id']) for q in jl(baseline/'goal_quartets.jsonl')]
    assert len(gq) == 840
    write_jl(canonical/'goal_quartets.jsonl', gq)
    pp, ll, qq, cases, categories = generate()
    parameter = frozen/'parameter'; parameter.mkdir()
    write_jl(parameter/'model_inputs.jsonl', pp); write_jl(parameter/'evaluation.jsonl', ll)
    write_jl(parameter/'goal_quartets.jsonl', qq); write_jl(parameter/'cases.jsonl', cases)
    # Distinct fixed engineering pilot, never included in scientific denominators.
    # Five new task families, three old goal tasks, two behavior tasks.
    def pick(rows): return min(rows, key=lambda p: digest(['pilot', p['prompt_id']]))
    pilot = [pick([p for p in pp if next(l['task'] for l in ll if l['prompt_id'] == p['prompt_id']) == task]) for task in sorted({l['task'] for l in ll})]
    label_by_id = {p['prompt_id']: p for p in labels}
    for task in ['dictator', 'trust', 'ultimatum']:
        pilot.append(pick([p for p in ps if p['task_type'] == 'goal' and inputs[p['prompt_id']]['task'] == task]))
    for task in ['public_goods', 'third_party_punishment']:
        pilot.append(pick([p for p in ps if p['task_type'] == 'behavior' and label_by_id[p['prompt_id']]['task'] == task]))
    assert len(pilot) == len({p['prompt_id'] for p in pilot}) == 10
    write_jl(frozen/'pilot.jsonl', pilot)
    vendor = R/'vendor'; vendor.mkdir(exist_ok=True)
    for name in ['audit_new_runs.py', 'score_joint_reasoning_v2.py']:
        shutil.copyfile(old/name, vendor/name)
    all_ps = ps + pp
    assert all(set(p) == {'prompt_id', 'task_type', 'messages', 'output_schema'} for p in all_ps)
    hashes = {str(p.relative_to(R)).replace('\\', '/'): sha(p) for p in frozen.rglob('*') if p.is_file()}
    code = {str(p.relative_to(R)).replace('\\', '/'): sha(p) for p in list(R.glob('*.py')) + list(vendor.glob('*.py'))}
    cfg = dict(protocol='INVAR_confirmation_20261112_v1', models=MODELS, arms=['native_no_thinking', 'native_thinking'],
        max_tokens=4096, max_model_len=8192, temperature=.7, top_p=.95, top_k=-1, repetition_penalty=1.,
        behavior_repeats=128, goal_repeats=8, stages=['canonical', 'parameter'],
        canonical=dict(behavior_prompts=429, goal_prompts=1410, pairs=92, behavior_quartets=184, goal_quartets=840, rows_per_model=132384),
        parameter=dict(prompts=480, pairs=40, goal_quartets=120, categories=categories, rows_per_model=7680),
        pilot_rows_per_model=20, scientific_rows=280128, pilot_rows=40, seed_base=20261112,
        files=hashes, runtime_sha256=code, source_input_sha256=sha(baseline/'inputs.jsonl'),
        previous_verified_runs_root='/workspace/project_social_simulation/invar_shortpaper_runs/20261111',
        scope='New independent batches on already observed canonical inputs; parameter diagnostic is new numbers in explicitly modified solvable mechanisms, not a new empirical source.',
        scoring='All-attempt Q; legal menu and type checks; conditional behavior E/G/R with separate missing-quality bounds. G<=2E, R has no preferred direction. Pair equal primary, source/task equal sensitivity.',
        stop_rule='Run the fixed two-model matrix once; retain all failures. No result-based rerolls, source/seed deletion, or expansion. Restore actual saved services sequentially within each GPU.')
    save(R/'freeze.json', cfg)
    save(R/'preparation_audit.json', dict(status='passed_cpu_preparation', parameter_independent_payoffs_and_optima_verified=True,
        equivalent_render_rule_fragments_once=True, canonical_input_payload_allowlist_verified=True,
        parameter_unique_messages=len({digest(p['messages']) for p in pp}), parameter_categories=categories, freeze_sha256=sha(R/'freeze.json')))
    print('Frozen scientific rows:', cfg['scientific_rows'], 'pilot:', cfg['pilot_rows'], 'parameter categories:', categories)


if __name__ == '__main__': main()
