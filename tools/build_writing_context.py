"""Extract traceable writing tables from accepted reports; no resampling or inference.

Writes only writing/tables/. Existing experimental reports are never changed.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'writing/tables'
INPUTS = {}


def source(path):
    p = ROOT / path
    rel = p.relative_to(ROOT).as_posix()
    INPUTS[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def rows(path):
    with source(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load(path):
    return json.loads(source(path).read_text(encoding='utf-8-sig'))


def write(name, values):
    assert values
    fields = list(dict.fromkeys(k for r in values for k in r))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_table = []
    inventory = rows('results/20261111_短文补实验/verified_model_inventory.csv')
    assert len(inventory) == 26
    for entry in inventory:
        report = Path(entry['path']).relative_to(ROOT).as_posix()
        behavior = next(r for r in rows(report + '/behavior_summary.csv') if r['group'] == 'all')
        controls = {r['value']: r for r in rows(report + '/control_summary.csv') if r['group'] == 'type'}
        audit = load(report + '/completion_audit.json')
        assert audit['status'] == 'passed' and audit['raw_sha256'] == entry['raw_sha256']
        assert int(audit['response_rows']) == 421504
        assert math.isclose(float(behavior['E']), float(entry['E']), abs_tol=1e-12)
        assert math.isclose(float(behavior['G']), float(entry['G']), abs_tol=1e-12)
        model_table.append(dict(campaign=entry['campaign'], model=entry['model'],
            setting='original_structured_interface', weighting='pair_equal', rows=421504,
            pairs=92, behavior_quartets=701, E=behavior['E'], G=behavior['G'], R=behavior['R'],
            changed_fact_U=controls['changed_fact']['accuracy'],
            stable_fact_U=controls['stable_fact']['accuracy'], goal_Q=controls['goal']['accuracy'],
            raw_sha256=audit['raw_sha256'], accepted_report=report))
    for r in rows('results/20261111_短文补实验/new_model_main_summary.csv'):
        if r['subset'] != 'all':
            continue
        report = f"experiments/shortpaper_20261111/remote_records/runs/{r['job']}/main/scoring/reports/{r['job']}"
        audit = load(report + '/completion_audit.json')
        assert audit['status'] == 'passed' and audit['response_rows'] == 421504
        model_table.append(dict(campaign='shortpaper_20261111_main', model=r['job'],
            setting='original_structured_interface', weighting='pair_equal', rows=421504,
            pairs=92, behavior_quartets=701, E=r['E'], G=r['G'], R=r['R'],
            changed_fact_U=r['changed_fact_U'], stable_fact_U=r['stable_fact_U'],
            goal_Q=r['goal_Q'], raw_sha256=audit['raw_sha256'], accepted_report=report))
    assert len(model_table) == 28
    write('all_model_configurations.csv', model_table)

    goal = rows('results/20261112_确认补实验/accepted_goal.csv')
    behavior = rows('results/20261112_确认补实验/accepted_behavior.csv')
    joined = []
    for r in behavior:
        g = next(x for x in goal if x['job'] == r['job'] and x['stage'] == 'canonical'
                 and x['arm'] == r['arm'] and x['weighting'] == r['weighting'])
        joined.append(dict(**r, Q_all_attempts=g['Q_all_attempts'], worst_side_Q=g['worst_side_Q'],
            scope='canonical_429_behavior_prompts_1410_goal_prompts',
            E_G_denominator='legal_outputs_conditional', Q_denominator='all_attempts'))
    assert len(joined) == 12
    write('native_confirmation.csv', joined)
    write('parameter_confirmation.csv', [x for x in goal if x['stage'] == 'parameter'])
    write('training_summary.csv', rows('results/20261029_全部实验总结/三种子完整汇总.csv'))

    contrasts = rows('results/20261113_训练机制补实验/accepted_JS_minus_SFT_all_checkpoints.csv')
    assert len(contrasts) == 72
    means = []
    for model in sorted({r['model'] for r in contrasts}):
        for step in (50, 100, 150):
            for stratum in ('train', 'heldout_source', 'heldout_variant'):
                selected = [r for r in contrasts if r['model'] == model
                            and int(r['step']) == step and r['stratum'] == stratum]
                assert len(selected) == 4
                record = dict(model=model, step=step, stratum=stratum, source_panels=4,
                              edges_per_panel=16, training_seed=20261026)
                for metric in ('delta_proxy_W1', 'delta_proxy_JS', 'delta_decoder_W1_conditional'):
                    record[metric] = math.fsum(float(x[metric]) for x in selected) / 4
                means.append(record)
    write('mechanism_checkpoint_means.csv', means)

    outputs = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
               for p in sorted(OUT.glob('*.csv'))}
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        status='extracted_from_accepted_reports', date='2026-11-13',
        operation='deterministic extraction/join/four-panel arithmetic mean only; no new scientific sampling',
        inputs=INPUTS, outputs=outputs,
        counts=dict(model_configurations=28, native_confirmation_rows=12,
                    training_rows=12, mechanism_checkpoint_rows=18)),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', inputs=len(INPUTS), outputs=len(outputs), models=len(model_table))))


if __name__ == '__main__':
    main()
