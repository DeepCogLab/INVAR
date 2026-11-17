"""Describe all frozen source/seed/model results without selecting winners.

Only controller-verified jobs enter this table. Three training seeds are
descriptive repeats; shared baseline responses are never independent repeats.
"""
import argparse
import collections
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

R=Path(__file__).resolve().parent
SOURCES=('HPVBG','IGVOFO','PGG','PIQ572')
METHODS=('baseline','anchor_sft','anchor_js')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csv_rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save_json(p,value):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def save_csv(p,rows,fields):
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    t.replace(p)


def weighted(points,aggregation):
    assert points and all(p['weight']>0 for p in points)
    assert aggregation in ('pair_weighted','source_macro')
    weights=[p['weight'] if aggregation=='pair_weighted' else 1 for p in points]
    return sum(p['estimate']*w for p,w in zip(points,weights))/sum(weights)


def collect_screen(root,model,source,seed):
    screen=root/'evaluations/transfer'/model/source/f'seed{seed}'/'screen'
    marker=screen/'scoring_complete.json'
    if not marker.is_file():return None
    before=sha(marker);m=read(marker)
    if m['status']!='independently_verified' or m['model']!=model:return None
    # A new scoring pass publishes its analysis manifest before replacing the
    # old completion marker. Do not read tables during that interval.
    if sha(screen/'analysis_manifest.json')!=m['manifest_sha256']:return None
    independent=read(screen/'independent_audit/audit.json')
    if independent['checked_raw_rows']!=m['raw_rows'] or independent['checked_methods']!=len(m['methods']):return None
    verified=[]
    for method in ['anchor_sft','anchor_js']:
        jobid=f'{model}_{source}_{seed}_{method}'
        proof=root/'status'/(jobid+'_verified.json')
        if proof.is_file() and method in m['methods']:
            p=read(proof)
            assert p['status']=='generated_and_independently_verified'
            assert (p['job']['model'],p['job']['scope'],p['job']['training_seed'],p['job']['method'])==(model,source,seed,method)
            verified.append(method)
    if not verified:return None
    baseline=read(screen/model/'baseline.complete.json')
    full=read(root/'evaluations/full92'/model/'screen'/model/'baseline.complete.json')
    assert baseline['analysis_projection']['parent_raw_sha256']==full['responses_sha256'], 'Baseline is not the shared full92 reference'
    selected=set(verified+['baseline'])
    values=[]
    for row in csv_rows(screen/'scores/summary.csv'):
        if row['model']!=model or row['method'] not in selected or row['group']!='all' or row['status']!='complete':continue
        n=int(row['n_prompts'] if row['kind']=='role' else row['n_pairs'])
        values.append(dict(model=model,source=source,training_seed=seed,method=row['method'],kind=row['kind'],metric=row['metric'],
                           estimate=float(row['estimate']),weight=n,baseline_reused=row['method']=='baseline'))
    for row in csv_rows(screen/'finite_sample_audit/summary.csv'):
        if row['model']!=model or row['method'] not in selected or row['group']!='all':continue
        for metric in ['E2_unbiased','G2_unbiased']:
            values.append(dict(model=model,source=source,training_seed=seed,method=row['method'],kind='behavior',metric=metric,
                               estimate=float(row[metric]),weight=int(row['n_pairs']),baseline_reused=row['method']=='baseline'))
    if sha(marker)!=before:return None
    # Baseline reuse is byte-identical across training-seed analyses.
    if len(verified)==2:
        parents=[root/'learning'/model/source/f'seed{seed}'/method for method in ['anchor_sft','anchor_js']]
        configs=[read(p/'training_config.json') for p in parents]
        for field in ['source_pool_sha256','exposure_schedule_sha256','input_sha256']:
            assert configs[0][field]==configs[1][field],f'Training arms differ in {field}'
        assert sha(parents[0]/'actual_exposures.json')==sha(parents[1]/'actual_exposures.json'), 'Actual exposure differs'
    return dict(values=values,verified=verified,baseline_hash=baseline['responses_sha256'],
                scoring_marker=str(marker),marker_sha256=before)


def summarize_seed(folds,model,seed):
    # No grand mean until both arms are independently verified in every fold.
    if set(folds)!=set(SOURCES) or any(set(v['verified'])!={'anchor_sft','anchor_js'} for v in folds.values()):return []
    groups=collections.defaultdict(list)
    for fold in folds.values():
        for row in fold['values']:groups[(row['method'],row['kind'],row['metric'])].append(row)
    result=[]
    for (method,kind,metric),points in sorted(groups.items()):
        assert len({p['source'] for p in points})==len(points)
        # PGG has no goals: absence is not a zero, and source-macro Q has three sources.
        for aggregation in ['pair_weighted','source_macro']:
            result.append(dict(model=model,training_seed=seed,method=method,kind=kind,metric=metric,
                aggregation=aggregation,estimate=weighted(points,aggregation),eligible_sources=len(points),
                eligible_units=sum(p['weight'] for p in points),baseline_reused=method=='baseline'))
    return result


def seed_statistics(rows,seeds):
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['model'],row['method'],row['kind'],row['metric'],row['aggregation'])].append(row)
    output=[]
    for key,items in sorted(groups.items()):
        if {r['training_seed'] for r in items}!=set(seeds):continue
        model,method,kind,metric,aggregation=key
        values=[r['estimate'] for r in items]
        if method=='baseline':
            assert max(values)-min(values)<1e-12,'Shared baseline changed across seeds'
            n=0;fitted=1;sd=None;scope='single shared baseline; no training-seed SD'
        else:
            n=len(values);fitted=4*n;sd=statistics.stdev(values);scope='descriptive training-seed variation; four overlapping source folds per seed, not independent studies or CI'
        output.append(dict(model=model,method=method,kind=kind,metric=metric,aggregation=aggregation,n_training_seeds=n,distinct_fitted_models=fitted,
            mean=statistics.mean(values),sd=sd,minimum=min(values),maximum=max(values),scope=scope))
    return output


def differences(rows):
    lookup={(r['model'],r['training_seed'],r['method'],r['kind'],r['metric'],r['aggregation']):r for r in rows}
    out=[]
    for key,row in sorted(lookup.items()):
        model,seed,method,kind,metric,aggregation=key
        if method!='anchor_js':continue
        for comparator in ['anchor_sft','baseline']:
            other=lookup[(model,seed,comparator,kind,metric,aggregation)]
            out.append(dict(model=model,training_seed=seed,comparison='anchor_js minus '+comparator,kind=kind,metric=metric,
                aggregation=aggregation,difference=row['estimate']-other['estimate']))
    return out


def main(root=R):
    plan=read(root/'stage_manifest.json');dest=root/'aggregate';dest.mkdir(exist_ok=True)
    source_rows=[];seed_rows=[];proofs=[];baseline_hashes={}
    for model in plan['models']:
        for seed in plan['training_seeds']:
            folds={}
            for source in SOURCES:
                result=collect_screen(root,model,source,seed)
                if not result:continue
                key=(model,source)
                if key in baseline_hashes:assert baseline_hashes[key]==result['baseline_hash']
                baseline_hashes[key]=result['baseline_hash']
                folds[source]=result;source_rows.extend(result['values'])
                proofs.append(dict(model=model,source=source,seed=seed,marker_sha256=result['marker_sha256'],methods=result['verified']))
            seed_rows.extend(summarize_seed(folds,model,seed))
    stats=seed_statistics(seed_rows,plan['training_seeds']);delta=differences(seed_rows)
    save_csv(dest/'source_seed_metrics.csv',source_rows,['model','source','training_seed','method','kind','metric','estimate','weight','baseline_reused'])
    save_csv(dest/'complete_seed_metrics.csv',seed_rows,['model','training_seed','method','kind','metric','aggregation','estimate','eligible_sources','eligible_units','baseline_reused'])
    save_csv(dest/'training_seed_statistics.csv',stats,['model','method','kind','metric','aggregation','n_training_seeds','distinct_fitted_models','mean','sd','minimum','maximum','scope'])
    save_csv(dest/'paired_seed_differences.csv',delta,['model','training_seed','comparison','kind','metric','aggregation','difference'])
    verified=len(list((root/'status').glob('*_verified.json')))
    audit=dict(status='partial_no_final_claim' if verified<52 else 'matrix_complete_pending_final_local_audit',
        verified_jobs=verified,planned_jobs=52,complete_model_seed_blocks=len({(r['model'],r['training_seed']) for r in seed_rows}),
        source_proofs=proofs,baseline_policy='same source baseline hash across all training seeds; never counted as three independent baselines',
        aggregation='pair-weighted primary; equal eligible sources secondary; roles weighted by unique prompts; PGG Q absent',
        script_sha256=sha(Path(__file__)),updated_unix=time.time())
    save_json(dest/'audit.json',audit)
    lines=['# 第四阶段比较汇总','',f'已独立复算生成项目：{verified}/52；完整模型×训练种子组合：{audit["complete_model_seed_blocks"]}/6。',
           '尚未完成四来源双臂的训练种子不进入跨来源总表。所有现有材料已观察，本轮为回溯迁移。','',
           'source_seed_metrics保留逐折结果；complete_seed_metrics按条件对权重及来源宏平均分别汇总；paired_seed_differences保留每种子JS−SFT与JS−baseline。',
           'training_seed_statistics仅在三个训练种子齐备后报告描述性均值、标准差和范围；原模型只是一份共享基线，不赋予三个独立重复。',
           '每个训练种子的跨来源总表拼合四个不同留来源适配器的评价，不代表一个适配器在全部92组上的成绩；四折训练来源重叠。',
           '角色按唯一提示数加权；无Q来源不填0，来源宏平均Q只对适用来源计算。R无越大越好的含义；主E/G及平方辅助E2/G2均保留。','',
           '最终结论仍需联合各screen/diagnostics中的输出熵、互斥目标、条件对尾部和任务/角色曝光覆盖，不能仅以E降低宣告修复。']
    (dest/'比较汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['status','verified_jobs','complete_model_seed_blocks']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=R);a=p.parse_args();main(a.root)
