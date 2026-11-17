"""Frozen V3 mathematics, explicit source views and V4-specific reporting."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

R = Path(__file__).resolve().parent
V3 = Path(os.environ.get('INVAR_METHODS_V3', R.parent/'methods_v3'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def baseline_view(parent, destination, manifest, model):
    """Copy exact parent lines once per source, then hard-link across seeds.

    This is an analysis view, explicitly not a newly generated run. Config and
    sampling identities remain those of the original full92 generation.
    """
    source = parent/model/'baseline.jsonl'
    marker = json.loads(source.with_suffix('.complete.json').read_text(encoding='utf-8'))
    assert sha(source) == marker['responses_sha256']
    identity = json.loads(source.with_suffix('.run_identity.json').read_text(encoding='utf-8'))
    selected = set(manifest['selected_prompt_ids'])
    assert selected <= set(identity['plan']['selected_prompt_ids'])
    scope = manifest['scope']
    cache = R/'baseline_views'/model/scope
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache/'baseline.jsonl'
    projection = dict(kind='exact_raw_line_projection_not_new_generation', parent_raw=str(source.resolve()),
                      parent_raw_sha256=marker['responses_sha256'], selected_prompt_ids_sha256=hashlib.sha256(
                          json.dumps(sorted(selected)).encode()).hexdigest(), rows=manifest['calls_per_model_method'])
    record_path = cache/'projection.json'
    if raw.exists():
        assert json.loads(record_path.read_text(encoding='utf-8')) == projection
        child_marker = json.loads(raw.with_suffix('.complete.json').read_text(encoding='utf-8'))
        assert sha(raw) == child_marker['responses_sha256']
    else:
        seen = set()
        temp = raw.with_suffix('.tmp')
        with source.open('rb') as src, temp.open('wb') as dst:
            for line in src:
                row = json.loads(line)
                if row['prompt_id'] not in selected:
                    continue
                key = (row['prompt_id'], row['replicate'])
                assert key not in seen and row['config_sha256'] == marker['config_sha256']
                seen.add(key)
                dst.write(line)
        assert len(seen) == projection['rows']
        temp.replace(raw)
        child_marker = dict(marker, rows=len(seen), responses_sha256=sha(raw), analysis_projection=projection)
        child_plan = dict(identity['plan'], selected_prompt_ids=sorted(selected),
                          planned_rows=len(seen), unique_inputs=len(selected))
        if 'inputs_by_type' in manifest:
            child_plan['inputs_by_type'] = manifest['inputs_by_type']
        for key in ['source_partitions', 'provenance_sources']:
            if key in child_plan:
                child_plan[key] = {pid:value for pid,value in child_plan[key].items() if pid in selected}
        child_plan.pop('content_sha256', None)
        child_plan['content_sha256'] = hashlib.sha256(json.dumps(child_plan, ensure_ascii=False, sort_keys=True,
                                                                separators=(',', ':')).encode()).hexdigest()
        child_identity = dict(identity, plan=child_plan, analysis_projection=projection)
        atomic(raw.with_suffix('.complete.json'), child_marker)
        atomic(raw.with_suffix('.run_identity.json'), child_identity)
        atomic(record_path, projection)
    target = destination/model
    target.mkdir(parents=True, exist_ok=True)
    for name in ['baseline.jsonl','baseline.complete.json','baseline.run_identity.json']:
        p = target/name
        if p.exists():
            assert sha(p) == sha(cache/name), 'Existing baseline view differs'
        else:
            os.link(cache/name, p)
    atomic(target/'baseline.projection.json', projection)


def run(args):
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.parent_baseline:
        baseline_view(args.parent_baseline, args.out, manifest, args.model)
    methods = [m for m in ['baseline','anchor_sft','anchor_js']
               if (args.out/args.model/(m+'.complete.json')).is_file()]
    assert methods and 'baseline' in methods
    analysis = dict(manifest, initial_models=[args.model], model_methods={args.model:methods},
                    parent_manifest_sha256=sha(args.manifest), analysis_methods=methods)
    atomic(args.out/'analysis_manifest.json', analysis)
    a = SimpleNamespace(baseline=args.baseline, manifest=args.out/'analysis_manifest.json', out=args.out,
                        methods=','.join(methods), bootstrap=500)
    score = module('invar_v4_frozen_v3_score', V3/'score_methods.py')
    score.report(a)
    finite = module('invar_v4_frozen_v3_finite', V3/'finite_sample_audit.py')
    finite.run(a)
    # Independent standard-library scalar implementation, with no math edits.
    verifier = Path(os.environ.get('INVAR_VERIFY_METHODS', R.parent.parent/'tools/verify_methods.py'))
    subprocess.run([sys.executable,str(verifier),'--baseline',str(args.baseline),'--manifest',str(a.manifest),
                    '--out',str(args.out)],check=True)
    audit = json.loads((args.out/'independent_audit/audit.json').read_text(encoding='utf-8'))
    assert audit['checked_methods'] == len(methods)
    rows = list(csv.DictReader((args.out/'scores/summary.csv').open(encoding='utf-8-sig')))
    lookup = {(r['method'],r['kind'],r['metric']):float(r['estimate']) for r in rows if r['group']=='all' and r['status']=='complete'}
    lines = ['# INVAR 全量与来源迁移比较','',
             '本报告仅含已完成并独立复算的方法；原有数据已被观察，是开发与回溯迁移。',
             f'模型：{args.model}；范围：{manifest["scope"]}；{len(manifest["selected_pairs"])}组条件；完成方法：{", ".join(methods)}。','',
             '| 方法 | E↓ | G↓ | R（描述） | 应变事实U | 未变事实U | 目标Q |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        vals = [lookup.get((method,k,m)) for k,m in [('behavior','E'),('behavior','G'),('behavior','R'),
                 ('changed_fact','accuracy'),('stable_fact','accuracy'),('goal','accuracy')]]
        lines.append('| '+method+' | '+' | '.join('不适用' if v is None else f'{v:.6f}' for v in vals)+' |')
    if set(['anchor_sft','anchor_js']) <= set(methods):
        lines += ['', '主比较JS−同曝光SFT（固定题目点估计）：']
        for kind, metric in [('behavior','E'),('behavior','G'),('changed_fact','accuracy'),('stable_fact','accuracy'),('goal','accuracy')]:
            keys = [(m,kind,metric) for m in ['anchor_sft','anchor_js']]
            if all(k in lookup for k in keys):
                lines.append(f'- {kind}:{metric}：{lookup[keys[1]]-lookup[keys[0]]:+.6f}')
    lines += ['', 'E2/G2另见finite_sample_audit；熵、来源/任务/变体/菜单分项见scores。R没有统一越大越好的解释。',
              '原模型若为留来源投影，复用同一全量原始回答，不能当作多个独立种子；投影记录给出父文件SHA。',
              '控制准确率点态Hoeffding界见independent_audit；不代表新情境或多重比较保证。']
    (args.out/'development_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic(args.out/'scoring_complete.json',dict(status='independently_verified',model=args.model,
           methods=methods,raw_rows=audit['checked_raw_rows'],manifest_sha256=sha(a.manifest),
           code_hashes={p.name:sha(p) for p in [Path(__file__),V3/'score_methods.py',V3/'finite_sample_audit.py',verifier]}))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--parent-baseline',type=Path)
    run(p.parse_args())
