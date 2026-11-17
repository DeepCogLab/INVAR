"""Read frozen reports; write new descriptive diagnostics without changing old scores."""
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/baseline_v2'
OUT = ROOT / 'results/20261111_短文补实验'


def jl(p):
    return [json.loads(s) for s in p.open(encoding='utf-8-sig') if s.strip()]


def csvread(p):
    return list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))


def write(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def avg(rows, key):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['pair_id']].append(float(r[key]))
    return sum(sum(v)/len(v) for v in by.values())/len(by) if by else None


def main():
    OUT.mkdir(exist_ok=True)
    ps = {p['prompt_id']: p for p in jl(BASE/'inputs.jsonl')}
    gq = jl(BASE/'goal_quartets.jsonl')
    fq = jl(BASE/'fact_quartets.jsonl')
    qlookup = {q['quartet_id']:q for q in gq+fq}
    findex = collections.defaultdict(list)
    for q in fq:
        findex[q['pair_id'],q['variant_id']].append(q)
    pairs = {p['pair_id']: p for p in jl(BASE/'INVAR_dataset/data/pairs.jsonl')}
    cases = {p['case_id']: p for p in jl(BASE/'INVAR_dataset/data/cases.jsonl')}
    bq = jl(BASE/'INVAR_dataset/data/quartets.jsonl')
    # Independent objective computation, using canonical participant payoffs only.
    oracle_checks = []
    case_for_goal = collections.defaultdict(set)
    for q in gq:
        for tag, side in zip('ABCD', [0, 0, 1, 1]):
            p = ps[q[tag]]
            case = cases[pairs[q['pair_id']][f'case{side}_id']]
            acts = case['action_space'].get('values')
            pays = case['payoffs'].get('by_action', case['payoffs'])
            values = {}
            for a in acts:
                pay = dict(pays.get(a, pays.get(str(a))))
                if 'self' in pay:
                    pay.setdefault(case['focal_id'], pay['self'])
                vector = [pay[k] for k in case['entities']]
                policy = p['policy']
                values[a] = pay[case['focal_id']] if policy == 'own' else sum(vector) if policy == 'total' else min(vector)-max(vector)
            maximum = max(values.values())
            optimal = {str(a) for a, v in values.items() if v == maximum}
            assert optimal == {str(a) for a in p['expected_actions']}, (p['prompt_id'], optimal)
            case_for_goal[q[tag]].add(case['case_id'])
            oracle_checks.append((q[tag], case['case_id']))

    folders = [('baseline_v2', p.name, p) for p in (BASE/'reports').iterdir() if (p/'completion_audit.json').exists()]
    social = ROOT/'experiments/social_cognition_v1/remote_records/runs'
    folders += [('social_cognition_v1', p.name, p/'full/scoring/reports'/p.name)
                for p in social.iterdir() if (p/'full/scoring/reports'/p.name/'completion_audit.json').exists()]
    switch_rows, fact_goal_rows, noise_rows, inventory = [], [], [], []

    def controls(label, model, acc, meta):
        for q in gq:
            if not all(q[t] in acc for t in 'ABCD'):
                continue
            fact_matches = findex[q['pair_id'],q['variant_id']]
            for tag in 'ABCD':
                fp = {f[tag] for f in fact_matches if f[tag] in acc}
                if not fp:
                    continue
                fmean = sum(acc[x] for x in fp)/len(fp)
                fact_goal_rows.append(dict(campaign=label, model=model, **meta,
                    pair_id=q['pair_id'], source=pairs[q['pair_id']]['source_id'], task=q['task'],
                    variant=q['variant_id'], side=tag, policy=q['policy'], goal_prompt=q[tag],
                    fact_accuracy=fmean, goal_accuracy=acc[q[tag]],
                    fact_correct_goal_wrong_independent=fmean*(1-acc[q[tag]]),
                    both_fact_types_at_least_95=all(acc[x] >= .95 for x in fp),
                    n_fact_prompts=len(fp)))
        # Compare all disjoint-optimum policy pairs, deduplicated by prompt IDs.
        buckets = collections.defaultdict(dict)
        for q in gq:
            if not all(q[t] in acc for t in 'ABCD'):
                continue
            for tag in 'ABCD':
                buckets[q['pair_id'], q['variant_id'], tag][q['policy']] = q[tag]
        seen = set()
        for (pair, variant, tag), goals in sorted(buckets.items()):
            for p1, p2 in itertools.combinations(sorted(goals), 2):
                a, b = goals[p1], goals[p2]
                if set(map(str, ps[a]['expected_actions'])) & set(map(str, ps[b]['expected_actions'])):
                    continue
                key = tuple(sorted([a, b]))
                if key in seen:
                    continue
                seen.add(key)
                switch_rows.append(dict(campaign=label, model=model, **meta, pair_id=pair,
                    source=pairs[pair]['source_id'], task=pairs[pair]['task'], variant=variant, side=tag,
                    policy1=p1, policy2=p2, prompt1=a, prompt2=b,
                    accuracy1=acc[a], accuracy2=acc[b], mean_Q=(acc[a]+acc[b])/2,
                    worst_policy=min(acc[a], acc[b]), both_correct_independent=acc[a]*acc[b]))

    for campaign, model, folder in folders:
        audit = json.loads((folder/'completion_audit.json').read_text())
        assert audit['status'] == 'passed'
        acc = {}
        for row in csvread(folder/'control_quartets.csv'):
            q = qlookup[row['quartet_id']]
            for t in 'ABCD':
                x = float(row['accuracy_'+t])
                assert q[t] not in acc or abs(acc[q[t]]-x) < 1e-12
                acc[q[t]] = x
        controls(campaign, model, acc, {})
        rows = csvread(folder/'behavior_quartets.csv')
        structural = [r for r in rows if r['variant_id'] != 'V6_relation']
        main = [r for r in csvread(folder/'behavior_summary.csv') if r['group'] == 'all'][0]
        noise_rows.append(dict(campaign=campaign, model=model, E=avg(rows, 'E'),
            E_pooled_null_reference=avg(rows, 'E_permutation_reference'),
            E_minus_null_reference_descriptive=avg(rows, 'E')-avg(rows, 'E_permutation_reference'),
            structural_E=avg(structural, 'E'), structural_G=avg(structural, 'G'),
            all_G=avg(rows, 'G'), repeat_mean_W1=sum(float(r['W1']) for r in csvread(folder/'same_prompt_repeat.csv'))/127,
            note='Permutation mean is a noise reference, not a p value or unbiased W1 correction. Repeat subset has different weights.'))
        inventory.append(dict(campaign=campaign, model=model, status='passed', rows=audit['response_rows'],
            raw_sha256=audit['raw_sha256'], path=str(folder), E=float(main['E']), G=float(main['G'])))

    mirror = Path('artifacts/methods_v4_20261028/evaluations/transfer')
    for file in sorted(mirror.rglob('prompts.csv')):
        model, source, seed = file.relative_to(mirror).parts[:3]
        grouped = collections.defaultdict(dict)
        for row in csvread(file):
            if row['task_type'] != 'behavior':
                grouped[row['method']][row['prompt_id']] = float(row['mean_normalized_or_correct'])
        for method, acc in grouped.items():
            if method == 'baseline' and seed != 'seed20261026':
                continue  # Shared original model is not three independent runs.
            controls('methods_v4', model, acc, dict(source_fold=source, seed=seed, method=method))

    write('verified_model_inventory.csv', inventory)
    write('presentation_noise_and_structural_sensitivity.csv', noise_rows)
    write('goal_discrimination_per_context.csv', switch_rows)
    write('matched_fact_goal_per_context.csv', fact_goal_rows)
    summaries = []
    for campaign, model, method, seed in sorted({(r['campaign'], r['model'], r.get('method',''), r.get('seed','')) for r in switch_rows}):
        rows = [r for r in switch_rows if (r['campaign'],r['model'],r.get('method',''),r.get('seed','')) == (campaign,model,method,seed)]
        for source in ['all'] + sorted({r['source'] for r in rows}):
            sub = [r for r in rows if source == 'all' or r['source'] == source]
            summaries.append(dict(campaign=campaign, model=model, method=method, seed=seed, source=source,
                n_unique_policy_prompt_pairs=len(sub), n_pair_references=len({r['pair_id'] for r in sub}),
                mean_Q=sum(r['mean_Q'] for r in sub)/len(sub),
                worst_policy=sum(r['worst_policy'] for r in sub)/len(sub),
                both_correct_independent=sum(r['both_correct_independent'] for r in sub)/len(sub),
                weighting='unique disjoint-policy prompt pairs; supplementary, not main Q weighting'))
    write('goal_discrimination_summary.csv', summaries)
    fact_summaries = []
    for key in sorted({(r['campaign'],r['model'],r.get('method',''),r.get('seed','')) for r in fact_goal_rows}):
        rows = [r for r in fact_goal_rows if (r['campaign'],r['model'],r.get('method',''),r.get('seed','')) == key]
        strong = [r for r in rows if r['both_fact_types_at_least_95']]
        fact_summaries.append(dict(zip(['campaign','model','method','seed'],key),
            n_rows=len(rows), n_goal_prompts=len({r['goal_prompt'] for r in rows}),
            goal_Q=avg(rows,'goal_accuracy'), fact_U=avg(rows,'fact_accuracy'),
            n_high_fact_rows=len(strong), high_fact_goal_Q=avg(strong,'goal_accuracy'),
            n_high_fact_low_goal_rows=sum(r['goal_accuracy'] <= .5 for r in strong),
            caveat='Independent calls; no observation of a within-trajectory knowledge-action transition.'))
    write('matched_fact_goal_summary.csv',fact_summaries)
    audit = dict(input_sha256=hashlib.sha256((BASE/'inputs.jsonl').read_bytes()).hexdigest(),
        verified_goal_reference_count=len(oracle_checks), unique_goals=len(case_for_goal),
        expected_unique_goals=1410, goal_oracle_all_passed=True, completed_model_configurations=len(inventory),
        completed_frozen_responses=sum(r['rows'] for r in inventory),
        generated_files=[p.name for p in OUT.glob('*.csv')],
        scope='Retrospective fixed benchmark diagnostics; does not create a new confirmation set.')
    assert len(case_for_goal) == 1410
    (OUT/'existing_evidence_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    main()
