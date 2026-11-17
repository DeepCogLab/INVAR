"""Fixed-panel Monte Carlo sensitivity; no population or human-behavior inference."""
import collections
import csv
import json
import math
import argparse
import numpy as np
from campaign_common import R, MODELS, arms, jl, load, save, sha, digest, SEED
from behavior_metrics import metrics


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x


def boot_edge(a,b):
    xs=np.unique(np.concatenate([a[np.isfinite(a)],b[np.isfinite(b)]]))
    ca=np.sum(np.isfinite(a),axis=1);cb=np.sum(np.isfinite(b),axis=1)
    valid=(ca>0)&(cb>0);v=np.full(len(a),np.nan)
    if valid.any():
        aa=a[valid];bb=b[valid];grid=xs[:-1]
        fa=(aa[:,:,None]<=grid[None,None,:]).sum(axis=1)/ca[valid,None]
        fb=(bb[:,:,None]<=grid[None,None,:]).sum(axis=1)/cb[valid,None]
        v[valid]=np.sum(np.abs(fa-fb)*np.diff(xs),axis=1)
    return v


def boot_metrics(q,v):
    def edge(a,b):return np.zeros(len(v[q[a]])) if q[a]==q[b] else boot_edge(v[q[a]],v[q[b]])
    e=(edge('A','B')+edge('C','D'))/2
    coeff=collections.Counter()
    for k,c in [('A',1),('B',-1),('C',-1),('D',1)]:coeff[q[k]]+=c
    means={pid:np.divide(np.nansum(v[pid],axis=1),np.sum(np.isfinite(v[pid]),axis=1),
        out=np.full(len(v[pid]),np.nan),where=np.sum(np.isfinite(v[pid]),axis=1)>0) for pid,c in coeff.items() if c}
    g=np.abs(sum(c*means[pid] for pid,c in coeff.items() if c))
    if not means:g=np.zeros(len(e))
    return dict(E_conditional=e,G_conditional=g)


def agg(qs,values,key=None):
    pairs=collections.defaultdict(list)
    for q,v in zip(qs,values):pairs[q['pair_id']].append(v)
    metadata={q['pair_id']:q for q in qs};groups=collections.defaultdict(list)
    for pid,vs in pairs.items():groups[metadata[pid][key] if key else 'all'].append(np.mean(vs,axis=0))
    return np.mean([np.mean(vs,axis=0) for vs in groups.values()],axis=0)


def ranges(a):
    finite=a[np.isfinite(a)]
    return dict(defined_resamples=len(finite),total_resamples=len(a),
        empirical_2_5_97_5=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None)


def audit_vectorization(qs,vals,boots):
    # Compare the independent vector CDF implementation with legacy scalar W1
    # and its quantile-grid cross-check, including shared prompt references.
    for i in range(min(3,len(next(iter(vals.values()))))):
        scalar_values={pid:[float(x) for x in a[i] if np.isfinite(x)] for pid,a in vals.items()}
        for q,b in zip(qs,boots):
            s=metrics(q,scalar_values,next(iter(vals.values())).shape[1])
            for name in ['E_conditional','G_conditional']:
                assert (s[name] is None and not np.isfinite(b[name][i])) or (s[name] is not None and abs(s[name]-b[name][i])<1e-10)


def analyze_job(job,B):
    d=R/'runs'/job;cfg=load(R/'freeze.json');summary=load(d/'SCORING_COMPLETE.json')
    assert summary['raw_sha256']==sha(d/'responses.jsonl') and summary['freeze_sha256']==sha(R/'freeze.json')
    rows=list(csv.DictReader((d/'attempt_scores.csv').open(encoding='utf-8-sig',newline='')))
    qs0=jl(R/'frozen/core/behavior_quartets.jsonl');ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl') if p['task_type']=='behavior'}
    n=cfg['protocol']['repeats']['behavior'];samples={};result={}
    for arm in arms(job):
        for lang in ['zh','en']:
            qs=[q for q in qs0 if q['language']==lang];vv={pid:np.full(n,np.nan) for pid,p in ps.items() if p['language']==lang}
            for row in rows:
                if row['arm']==arm and row['prompt_id'] in vv and row['valid']=='True':
                    vv[row['prompt_id']][int(row['replicate'])]=float(row['normalized_outcome'])
            rng=np.random.default_rng(int(digest([SEED,'mc',job,arm,lang])[:16],16))
            vals={pid:a[rng.integers(0,n,size=(B,n))] for pid,a in vv.items()}
            boots=[boot_metrics(q,vals) for q in qs];audit_vectorization(qs,vals,boots)
            for weight,key in [('pair_equal',None),('source_equal','source_id'),('task_equal','task')]:
                for m in ['E_conditional','G_conditional']:
                    value=agg(qs,[b[m] for b in boots],key);samples[lang,arm,weight,m]=value
                    result['__'.join([lang,arm,weight,m])]=ranges(value)
    contrasts={}
    for lang in ['zh','en']:
        for w in ['pair_equal','source_equal','task_equal']:
            for m in ['E_conditional','G_conditional']:
                delta=samples[lang,arms(job)[1],w,m]-samples[lang,arms(job)[0],w,m]
                contrasts['__'.join([lang,'second_minus_first',w,m])]=ranges(delta)
    for w in ['pair_equal','source_equal','task_equal']:
        for m in ['E_conditional','G_conditional']:
            delta=(samples['en',arms(job)[1],w,m]-samples['en',arms(job)[0],w,m])-(samples['zh',arms(job)[1],w,m]-samples['zh',arms(job)[0],w,m])
            contrasts['__'.join(['en_minus_zh_intervention_difference',w,m])]=ranges(delta)
    result=dict(status='independent_vector_scalar_agreement',job=job,resamples=B,seed=SEED,levels=result,contrasts=contrasts,
        raw_sha256=summary['raw_sha256'],freeze_sha256=sha(R/'freeze.json'),
        scope='Empirical plug-in resampling ranges conditional on this fixed panel and observed draws. Shared prompt IDs reuse the same resample within each arm. Arms/languages use independent random streams. W1 has finite-draw bias; these are not calibrated confidence intervals, a debiasing procedure, or evidence about unseen studies. Zero bootstrap spread does not imply certainty. Missing-output worst-case bounds are in SCORING_COMPLETE.json.')
    save(d/'MONTE_CARLO_SENSITIVITY.json',clean(result));return result


def report():
    cfg=load(R/'freeze.json');completed=[];table=[]
    for job in cfg['protocol']['job_order']:
        p=R/'runs'/job/'SCORING_COMPLETE.json'
        if not p.exists():continue
        s=load(p);completed.append(job)
        for r in s['summaries']:
            delivery=r['delivery_and_cost'];b=r['behavior_pair_equal']
            table.append(dict(model=job,family=MODELS[job]['family'],kind=MODELS[job]['kind'],role=MODELS[job]['role'],
                language=r['language'],arm=r['arm'],E_conditional=b['E_conditional'],E_missing_low=b['E_low'],E_missing_high=b['E_high'],
                G_conditional=b['G_conditional'],Q=r['goal_pair_equal_all_attempt_Q'],Q_source_equal=r['goal_source_equal_all_attempt_Q'],
                U_changed=r['changed_fact_pair_equal_all_attempt_U'],U_stable=r['stable_fact_pair_equal_all_attempt_U'],
                parameter_Q=r['parameter_pair_equal_all_attempt_Q'],parameter_joint=r['parameter_disjoint_joint_all_four_correct'],
                behavior_delivery=delivery['core_behavior']['delivery_rate'],goal_delivery=delivery['core_goal']['delivery_rate'],
                behavior_mean_tokens=delivery['core_behavior']['mean_completion_tokens'],goal_mean_tokens=delivery['core_goal']['mean_completion_tokens']))
    out=R/'results';out.mkdir(exist_ok=True)
    if table:
        with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    status=load(R/'science_queue_status.json') if (R/'science_queue_status.json').exists() else None
    save(out/'coverage.json',dict(completed_scored_jobs=completed,planned_jobs=cfg['protocol']['job_order'],queue=status,freeze_sha256=sha(R/'freeze.json')))
    lines=['# 新增稳健性实验：当前结果', '',f'已评分 {len(completed)}/{len(cfg["protocol"]["job_order"])} 个检查点。此文件由已验收的结果生成，未完成项不补零。', '',
        '范围：中英双语；行为与事实为四来源各8对；目标检查为13对（来源不均衡）；参数检查为20对互斥最优切片。英文经AI翻译及结构核验，未获人工语义认证。', '',
        '| 模型 | 语言 | 设置 | E（条件有效） | Q（全部尝试） | 参数四题全对 | 行为交付率 |',
        '|---|---|---|---:|---:|---:|---:|']
    def fmt(x):return '未定义' if x is None else f'{x:.4f}'
    for r in table:lines.append('| '+' | '.join([r['model'],r['language'],r['arm'],fmt(r['E_conditional']),fmt(r['Q']),fmt(r['parameter_joint']),fmt(r['behavior_delivery'])])+' |')
    lines+=['','解读要求：原生开关、推理强度、提示推理三类对照分别分析；基模单列。E较小不等于目标执行正确，也不证明模型关注或忽略了材料。先检查交付率、缺失界和同题抽样波动，再解释稳定性。来源加权的方向差异只能在明确口径下描述。',
        '', '详细数字见 summary.csv、各模型 SCORING_COMPLETE.json 及 MONTE_CARLO_SENSITIVITY.json。后者为固定题库的抽样敏感性范围，不是对新来源的置信区间。']
    (out/'CURRENT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job');a=ap.parse_args()
    if a.job:analyze_job(a.job,load(R/'freeze.json')['protocol']['mc_resamples'])
    report()
