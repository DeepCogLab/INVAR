"""Independent complete-plan accounting and scalar scoring, with all failures retained."""
import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path
from common import R, jl, load, save, sha, digest, seed, tasks, repetitions
sys.path.insert(0,str(R/'vendor'))
from audit_new_runs import parse_final
from score_joint_reasoning_v2 import metrics, aggregate


def table(p, rows):
    if not rows: return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def goal_aggregate(rows, key=None):
    pairs=collections.defaultdict(list)
    for r in rows:pairs[r['pair_id']].append(r)
    pp=[dict(pair_id=pid,source_id=rs[0]['source_id'],task=rs[0]['task'],Q=statistics.mean(r['Q_four_sides'] for r in rs),
        worst_side=statistics.mean(r['worst_side'] for r in rs)) for pid,rs in pairs.items()]
    groups=collections.defaultdict(list)
    for p in pp:groups[p[key] if key else 'all'].append(p)
    return {m:statistics.mean(statistics.mean(p[m] for p in ps) for ps in groups.values()) for m in ['Q','worst_side']}


def score(job,stage):
    cfg=load(R/'freeze.json');d=R/'runs'/job/stage
    assert not (d/'SCORING_COMPLETE.json').exists(),'Preserve completed scoring'
    for name,h in cfg['files'].items():assert sha(R/name)==h
    ps,plan=tasks(R,stage);expected=set(plan)
    labels={p['prompt_id']:p for p in jl(R/'frozen'/stage/'evaluation.jsonl')}
    identity=load(d.parent/'identity.json');complete=load(d/'COMPLETE.json')
    assert identity['protocol_sha256']==sha(R/'freeze.json') and identity['runner_sha256']==cfg['runtime_sha256']['worker.py']
    assert complete['rows']==len(plan) and sha(d/'responses.jsonl')==complete['sha256']
    assert load(d/'plan.json')['plan_sha256']==digest(plan)
    seen=set();values=collections.defaultdict(lambda:collections.defaultdict(list));correct=collections.defaultdict(list)
    counts=collections.defaultdict(collections.Counter);attempts=[]
    for row in jl(d/'responses.jsonl'):
        pid,arm,rep=row['prompt_id'],row['batch'],row['replicate'];key=(pid,arm,rep)
        assert key in expected and key not in seen;seen.add(key)
        assert row['model']==job and row['stage']==stage and row['status']=='ok'
        assert row['config_sha256']==digest(identity) and row['messages_sha256']==digest(ps[pid]['messages'])
        assert row['sampling_seed']==seed(stage,pid,rep) and 0<=row['completion_tokens']<=cfg['max_tokens']
        obj,error=parse_final(row['response_text'],row['finish_reason'],ps[pid]['output_schema'])
        lab=labels[pid];action=None;x=None;is_correct=None
        if obj is not None:
            action=obj['action'];space=lab['action_space']
            legal=any(type(action) is type(v) and action==v for v in space['values']) if 'values' in space else type(action) is int and space['min']<=action<=space['max'] and (action-space['min'])%space.get('step',1)==0
            if not legal:obj=None;error='menu'
        count=counts[arm,ps[pid]['task_type']];count['n']+=1;count['valid']+=obj is not None;count['truncated']+=row['finish_reason']=='length';count['completion_tokens']+=row['completion_tokens']
        if ps[pid]['task_type']=='goal':
            is_correct=obj is not None and action in lab['expected_actions'];correct[arm,pid].append(int(is_correct));count['correct']+=is_correct
        elif obj is not None:
            outcome=lab['outcome'];x=float(action==outcome['target_action']) if outcome['kind']=='probability' else action/outcome['denominator']
            assert 0<=x<=1;values[arm][pid].append(x)
        attempts.append(dict(prompt_id=pid,arm=arm,replicate=rep,task_type=ps[pid]['task_type'],valid=obj is not None,error=error,
            action=action,normalized_outcome=x,correct=is_correct,completion_tokens=row['completion_tokens']))
    assert seen==expected
    behavioral=[];goals=[];summaries=[]
    goal_quartets=jl(R/'frozen'/stage/'goal_quartets.jsonl')
    for arm in cfg['arms']:
        summary=dict(arm=arm,counts={typ:dict(count) for (a,typ),count in counts.items() if a==arm})
        if stage=='canonical':
            bq=jl(R/'frozen/canonical/behavior_quartets.jsonl')
            rs=[dict(arm=arm,**q,**metrics(q,values[arm],cfg['behavior_repeats'])) for q in bq]
            behavioral.extend(rs)
            summary['behavior']={name:aggregate(rs,key) for name,key in [('pair_equal',None),('source_equal','source_id'),('task_equal','task')]}
        qr=[]
        for q in goal_quartets:
            rates=[statistics.mean(correct[arm,q[k]]) for k in 'ABCD']
            # Quartet references are not independent prompts or trajectories.
            qr.append(dict(arm=arm,**q,Q_four_sides=statistics.mean(rates),worst_side=min(rates),
                Q_A=rates[0],Q_B=rates[1],Q_C=rates[2],Q_D=rates[3]))
        goals.extend(qr)
        summary['goal']={name:goal_aggregate(qr,key) for name,key in [('pair_equal',None),('source_equal','source_id'),('task_equal','task')]}
        summary['goal']['prompt_equal_Q']=statistics.mean(statistics.mean(v) for (a,p),v in correct.items() if a==arm)
        if stage=='parameter':
            summary['goal']['by_transition']={cat:goal_aggregate([q for q in qr if q['category']==cat]) for cat in sorted({q['category'] for q in qr})}
        summaries.append(summary)
    result=dict(status='passed_independent_scalar',job=job,stage=stage,rows=len(seen),raw_sha256=complete['sha256'],
        freeze_sha256=sha(R/'freeze.json'),scorer_sha256=sha(Path(__file__)),all_plan_seeds_messages_identity_verified=True,summaries=summaries,
        scope='All attempts in Q. Behavior conditional metrics exclude missing; separate missing-quality bounds are not sampling confidence intervals. Pair/source/task weighting and shared prompts retained. R has no preferred direction.')
    if stage=='canonical':
        off,on=[s['behavior']['pair_equal'] for s in summaries]
        result['delta_all_attempt_bounds']={m:[on[m+'_low']-off[m+'_high'],on[m+'_high']-off[m+'_low']] for m in ['E','G','R']}
    table(d/'attempt_scores.csv',attempts);table(d/'behavior_quartet_scores.csv',behavioral);table(d/'goal_quartet_scores.csv',goals)
    save(d/'SCORING_COMPLETE.json',result);print(job,stage,'independent scoring passed',len(seen))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--stage',required=True,choices=['canonical','parameter']);a=ap.parse_args();score(a.job,a.stage)
