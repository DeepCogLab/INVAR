"""Describe fixed V3-pair versus remaining-pair composition using V4 scores."""
import csv
import collections
import hashlib
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    screen = R / 'evaluations/full92/qwen3_8b/screen'
    marker = json.loads((screen / 'scoring_complete.json').read_text(encoding='utf-8'))
    assert marker['manifest_sha256'] == sha(screen / 'analysis_manifest.json')
    manifest = json.loads((screen / 'analysis_manifest.json').read_text(encoding='utf-8'))
    old_manifest = R.parent / 'methods_v3/precision_manifest.json'
    old = set(json.loads(old_manifest.read_text(encoding='utf-8'))['selected_pairs'])
    methods = marker['methods']
    path = screen / 'diagnostics/behavior_pair_means.csv'
    diagnosis = screen / 'diagnostics/audit.json'
    diagnosis_identity = json.loads(diagnosis.read_text(encoding='utf-8'))
    assert next(v for k, v in diagnosis_identity['input_sha256'].items()
                if k.endswith('/scoring_complete.json')) == sha(screen / 'scoring_complete.json'), 'Wait for matching completed diagnostics'
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    lookup = {(r['method'], r['pair_id']): r for r in rows}
    pairs = set(manifest['selected_pairs'])
    assert len(pairs) == 92 and len(old) == 12 and old < pairs
    assert set(lookup) == {(m, p) for m in methods for p in pairs}
    baseline = {p: lookup['baseline', p] for p in pairs}
    groups = {'all92': pairs, 'prior_fixed12': old, 'remaining80': pairs - old,
              'nonPGG_posthoc_sensitivity': {p for p in pairs if baseline[p]['source_id'] != 'PGG'}}
    for source in sorted({r['source_id'] for r in baseline.values()}):
        groups['source:' + source] = {p for p in pairs if baseline[p]['source_id'] == source}
    out = []
    for group, selected in groups.items():
        for method in methods:
            for metric in ['E', 'G', 'R']:
                values = [float(lookup[method, p][metric]) for p in sorted(selected)]
                delta = [float(lookup[method, p][metric]) - float(baseline[p][metric]) for p in sorted(selected)]
                row = dict(group=group, method=method, metric=metric, n_pairs=len(selected),
                           mean=statistics.mean(values), median=statistics.median(values),
                           mean_delta_vs_baseline=statistics.mean(delta))
                if method != 'baseline' and metric != 'R':
                    row.update(improved=sum(d < -1e-12 for d in delta),
                               worsened=sum(d > 1e-12 for d in delta),
                               ties=sum(abs(d) <= 1e-12 for d in delta))
                out.append(row)
    source_groups = [g for g in groups if g.startswith('source:')]
    assert len(source_groups) == 4
    for method in methods:
        for metric in ['E', 'G', 'R']:
            source_rows = [r for r in out if r['group'] in source_groups
                           and r['method'] == method and r['metric'] == metric]
            whole = next(r for r in out if r['group'] == 'all92'
                         and r['method'] == method and r['metric'] == metric)
            assert abs(sum(r['mean'] * r['n_pairs'] for r in source_rows) / 92 - whole['mean']) < 1e-12
            out.append(dict(group='source_macro', method=method, metric=metric,
                            n_pairs=92, n_sources=4, mean=statistics.mean(r['mean'] for r in source_rows),
                            mean_delta_vs_baseline=statistics.mean(r['mean_delta_vs_baseline'] for r in source_rows)))
    destination = R / 'legacy_expansion_diagnostics'
    destination.mkdir(exist_ok=True)
    comparisons = [(m, 'baseline') for m in methods if m != 'baseline']
    if {'anchor_js', 'anchor_sft'} <= set(methods):
        comparisons.append(('anchor_js', 'anchor_sft'))
    changes = []
    for method, reference in comparisons:
        for pair in sorted(pairs):
            left, right = lookup[method, pair], lookup[reference, pair]
            change = dict(comparison=method + ' minus ' + reference, pair_id=pair,
                          source=left['source_id'], task=left['task'])
            change.update({metric + '_delta': float(left[metric])-float(right[metric])
                           for metric in ['E', 'G', 'R']})
            changes.append(change)
    if changes:
        with (destination / 'paired_point_changes.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(changes[0]))
            writer.writeheader()
            writer.writerows(changes)
    fields = list(dict.fromkeys(k for row in out for k in row))
    with (destination / 'fixed_subset_and_source.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    # Descriptive guards only; entropy/variation do not establish a correct
    # free-behavior preference or fidelity to humans.
    input_path = R.parent / 'baseline_v2/inputs.jsonl'
    inputs = [json.loads(line) for line in input_path.open(encoding='utf-8') if line.strip()]
    originals = {p['prompt_id'] for p in inputs if p['task_type'] == 'behavior'
                 and any(u['variant'] == 'base' for u in p['uses'])}
    prompt_path = screen / 'scores/prompts.csv'
    prompts = list(csv.DictReader(prompt_path.open(encoding='utf-8-sig')))
    guards = []
    for method in methods:
        behavior = [p for p in prompts if p['method'] == method and p['task_type'] == 'behavior']
        assert len(behavior) == len({p['prompt_id'] for p in behavior}) == 1240
        entropy = [float(p['entropy_normalized']) for p in behavior]
        means = [float(p['mean_normalized_or_correct']) for p in behavior if p['prompt_id'] in originals]
        assert len(means) == len(originals)
        guards.append(dict(method=method, unique_behavior_prompts=1240,
                           normalized_entropy_mean=statistics.mean(entropy),
                           entropy_le_0_01=sum(h <= .01 for h in entropy),
                           entropy_ge_0_99=sum(h >= .99 for h in entropy),
                           unique_original_prompts=len(means),
                           original_normalized_means_sd=statistics.pstdev(means),
                           original_normalized_means_range=max(means)-min(means)))
    with (destination / 'descriptive_output_guards.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(guards[0]))
        writer.writeheader()
        writer.writerows(guards)
    menu_groups = collections.defaultdict(list)
    input_lookup = {p['prompt_id']: p for p in inputs}
    for row in prompts:
        if row['task_type'] != 'behavior' or row['method'] not in methods:
            continue
        menu = json.dumps(input_lookup[row['prompt_id']]['action_space'], sort_keys=True)
        for source in json.loads(row['source_ids']):
            menu_groups[row['method'], source, menu].append(row)
    menu_guards = []
    for (method, source, menu), selected in sorted(menu_groups.items()):
        distributions = [json.loads(p['action_or_answer_distribution']) for p in selected]
        canonical = {json.dumps(d, sort_keys=True) for d in distributions}
        point_actions = {next(iter(d)) for d in distributions if len(d) == 1 and next(iter(d.values())) == 1}
        all_point = all(len(d) == 1 and next(iter(d.values())) == 1 for d in distributions)
        menu_guards.append(dict(method=method, source=source, menu=menu,
                                n_unique_prompts=len(selected), n_responses=sum(int(p['n']) for p in selected),
                                normalized_entropy_mean=statistics.mean(float(p['entropy_normalized']) for p in selected),
                                n_unique_empirical_distributions=len(canonical),
                                exact_single_point_action=next(iter(point_actions)) if all_point and len(point_actions) == 1 else '',
                                literal_constant_output=all_point and len(point_actions) == 1))
    with (destination / 'source_menu_output_guards.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(menu_guards[0]))
        writer.writeheader()
        writer.writerows(menu_guards)
    audit = dict(status='descriptive_completed_scores_only', methods=methods,
                 input_sha256={str(p.relative_to(R.parent.parent)): sha(p)
                               for p in [path, diagnosis, prompt_path, input_path, old_manifest, screen / 'scoring_complete.json',
                                         screen / 'analysis_manifest.json', Path(__file__)]},
                 limitations=['Same V4 inference version for every contrast; no old V3 raw-score subtraction',
                              'Prior12 versus remaining80 is fixed selection composition, not training exposure',
                              'Existing legacy adapter, one training seed; not the new source-fold matrix',
                              'Directions are descriptive point estimates, not significance decisions',
                              'Unique prompt entropy and normalized mean dispersion are descriptive guards, not behavioral validity tests',
                              'NonPGG exclusion is a posthoc sensitivity after observing constant PGG outputs; full92 remains primary',
                              'R has no preferred direction; source/task/menu may be confounded'])
    (destination / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(methods=methods, rows=len(out))))


if __name__ == '__main__':
    run()
