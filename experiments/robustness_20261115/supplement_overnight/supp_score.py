"""CPU scoring with independent line parsing, transport and weighted checks."""
import argparse
import collections
import csv
import json
import re
import statistics
from supp_common import *
from parsing_v3 import parse_final
from behavior_metrics import metrics,aggregate,w1
import independent_acceptance_v3 as independent


def mean(v):return statistics.mean(v) if v and all(x is not None for x in v) else None


def controls(qs,truth,n,key=None):
    pairs=collections.defaultdict(list);meta={q['pair_id']:q for q in qs}
    for q in qs:pairs[q['pair_id']].append(sum(truth[q[k],r] for k in 'ABCD' for r in range(n))/(4*n))
    groups=collections.defaultdict(list)
    for pid,values in pairs.items():groups[meta[pid][key] if key else 'all'].append(mean(values))
    return mean([mean(v) for v in groups.values()])


def equal(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-10),(a,b)


def score(job):
    cfg=verify_freeze();d=S/'runs'/job;done=load(d/'COMPLETE.json');identity=load(d/'identity.json')
    assert sha(d/'responses.jsonl')==done['raw_sha256']
    assert identity['freeze_sha256']==sha(S/'freeze.json') and identity['job']==job
    assert identity['model']==JOBS[job]['model']
    assert done['identity_sha256']==digest(identity)
    ps={p['prompt_id']:p for p in jl(S/'frozen/model_inputs.jsonl')};labels={p['prompt_id']:p for p in jl(S/'frozen/evaluation.jsonl')}
    plan=jl(S/'plans'/(job+'.jsonl'));assert plan==make_plan(job,list(ps.values()))
    assert identity['trial_plan_sha256']==digest(plan) and len(plan)==identity['expected_rows']==done['rows']
    expected={(t['prompt_id'],cell(t),t['replicate']):t for t in plan};assert len(expected)==len(plan)
    seen=set();attempts=[];truth=collections.defaultdict(dict);values=collections.defaultdict(lambda:collections.defaultdict(list))
    stats=collections.defaultdict(list);identity_hash=digest(identity);kind=MODELS[JOBS[job]['model']]['kind'];independent_values=collections.defaultdict(lambda:collections.defaultdict(list));independent_truth=collections.defaultdict(dict)
    with (d/'responses.jsonl').open(encoding='utf-8') as raw_stream:
        for line in raw_stream:
            r=json.loads(line);c=cell(r);k=(r['prompt_id'],c,r['replicate']);assert k in expected and k not in seen;seen.add(k)
            assert all(r[x]==v for x,v in expected[k].items())
            assert r['job']==job and r['model']==JOBS[job]['model'] and r['identity_sha256']==identity_hash
            ik=input_key(r);p=ps[r['prompt_id']];lab=labels[r['prompt_id']]
            assert r['rendered_sha256']==identity['rendered_sha256'][ik]
            assert r['prompt_token_ids_sha256']==identity['prompt_token_ids_sha256'][ik]
            assert r['messages_sha256']==digest(messages(p,r,job))==identity['messages_sha256'][ik]
            assert r['stage']==p['stage'] and r['task_type']==p['task_type']
            assert r['completion_tokens']==len(r['token_ids']) and r['completion_tokens']<=r['max_tokens']
            obj,error,strict=parse_final(r['response_text'],r['response_with_special_tokens'],r['finish_reason'],p['output_schema'],kind,r['arm'],identity['prefilled_think'][ik])
            alt=independent.delivered(r,p['output_schema'],kind,identity['prefilled_think'][ik]);assert obj==alt,k
            good=obj is not None and p['task_type']=='goal' and obj['action'] in lab['expected_actions']
            independent_good=alt is not None and p['task_type']=='goal' and alt['action'] in lab['expected_actions']
            truth[c][p['prompt_id'],r['replicate']]=good;independent_truth[c][p['prompt_id'],r['replicate']]=independent_good
            x=None
            if obj is not None and p['task_type']=='behavior':
                outcome=lab['outcome'];x=float(obj['action']==outcome['target_action']) if outcome['kind']=='probability' else obj['action']/outcome['denominator']
                values[c][p['prompt_id']].append(x)
                independent_values[c][p['prompt_id']].append(float(alt['action']==outcome['target_action']) if outcome['kind']=='probability' else alt['action']/outcome['denominator'])
            starts=list(re.finditer(r'(?m)^[^\S\r\n]*(?=\{)',r['response_text'].strip())) if obj is not None else []
            prefix=r['response_text'].strip()[:starts[-1].start()].strip() if starts else None
            letters=[ch for ch in prefix if ch.isalpha()] if prefix is not None else []
            a=dict(prompt_id=p['prompt_id'],cell=c,replicate=r['replicate'],language=p['language'],stage=p['stage'],task_type=p['task_type'],
                valid=obj is not None,error=error,strict=strict,correct=bool(good) if p['task_type']=='goal' else None,
                normalized_outcome=x,completion_tokens=r['completion_tokens'],truncated=r['finish_reason']=='length',
                visible_prefix_chars=len(prefix) if prefix is not None else None,
                visible_prefix_cjk_fraction=sum('\u4e00'<=ch<='\u9fff' for ch in letters)/len(letters) if letters else None)
            attempts.append(a);stats[c,p['stage'],p['task_type']].append(a)
    assert seen==set(expected)
    bqs=jl(S/'frozen/core/behavior_quartets.jsonl');gqs=jl(S/'frozen/core/goal_quartets.jsonl');pqs=jl(S/'frozen/parameter/goal_quartets.jsonl')
    summaries=[];checks=0
    for c in sorted(truth):
        sample=next(t for t in plan if cell(t)==c);lang=sample['language']
        row=dict(job=job,experiment=JOBS[job]['experiment'],model=JOBS[job]['model'],cell=c,**{k:sample[k] for k in ['language','instruction_language','arm','max_tokens']})
        for stage,qs in [('core',gqs),('parameter',pqs)]:
            if 'goal' not in JOBS[job]['task_types']:continue
            qs=[q for q in qs if q['language']==lang]
            for q in qs:
                assert set(labels[q['A']]['expected_actions'])==set(labels[q['B']]['expected_actions'])
                assert set(labels[q['C']]['expected_actions'])==set(labels[q['D']]['expected_actions'])
            vals=[sum(independent_truth[c][q[k],r] for k in 'ABCD' for r in range(2))/8 for q in qs]
            for weight,key in [('pair',None),('source','source_id'),('task','task')]:
                value=controls(qs,truth[c],2,key);equal(value,independent.weighted(qs,vals,key));checks+=1
                row[stage+'_Q_'+weight+'_equal']=value
            switch=[q for q in qs if set(labels[q['A']]['expected_actions']).isdisjoint(labels[q['C']]['expected_actions'])]
            joint=mean([sum(all(truth[c][q[k],r] for k in 'ABCD') for r in range(2))/2 for q in switch])
            altjoint=mean([sum(all(independent_truth[c][q[k],r] for k in 'ABCD') for r in range(2))/2 for q in switch])
            equal(joint,altjoint);checks+=1;row[stage+'_joint_all_four_correct']=joint
        if 'behavior' in JOBS[job]['task_types']:
            qs=[q for q in bqs if q['language']==lang];bm=[dict(**q,**metrics(q,values[c],4)) for q in qs]
            altmetrics=[independent.behavior(q,independent_values[c],4) for q in qs]
            for weight,key in [('pair',None),('source','source_id'),('task','task')]:
                result=aggregate(bm,key)
                for metric in ['E_conditional','E_low','E_high','G_conditional','G_low','G_high']:
                    equal(result[metric],independent.weighted(qs,[v[metric] for v in altmetrics],key));checks+=1
                    row['behavior_'+weight+'_'+metric]=result[metric]
            half=[]
            for pid,p in ps.items():
                if p['language']!=lang or p['task_type']!='behavior':continue
                rs=[a for a in stats[c,'core','behavior'] if a['prompt_id']==pid]
                half.append(w1(*[[a['normalized_outcome'] for a in rs if a['valid'] and a['replicate']%2==h] for h in [0,1]]))
            row['behavior_same_prompt_split_half_W1_conditional']=mean(half)
        for stage,typ in ([('core','goal'),('parameter','goal')] if 'goal' in JOBS[job]['task_types'] else [])+([('core','behavior')] if 'behavior' in JOBS[job]['task_types'] else []):
            a=stats[c,stage,typ];assert a
            row[stage+'_'+typ+'_attempts']=len(a)
            row[stage+'_'+typ+'_delivery']=sum(x['valid'] for x in a)/len(a)
            row[stage+'_'+typ+'_truncated']=sum(x['truncated'] for x in a)/len(a)
            row[stage+'_'+typ+'_mean_completion_tokens']=mean([x['completion_tokens'] for x in a])
            row[stage+'_'+typ+'_strict_format']=sum(x['strict'] for x in a)/len(a)
            prefix=[x['visible_prefix_chars'] for x in a if x['visible_prefix_chars'] is not None]
            row[stage+'_'+typ+'_prefix_evaluable_count']=len(prefix)
            row[stage+'_'+typ+'_mean_visible_prefix_chars_conditional']=mean(prefix)
            row[stage+'_'+typ+'_mean_visible_prefix_cjk_fraction_conditional']=mean([x['visible_prefix_cjk_fraction'] for x in a if x['visible_prefix_cjk_fraction'] is not None])
        summaries.append(row)
    with (d/'attempt_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(attempts[0]));w.writeheader();w.writerows(attempts)
    result=dict(status='complete_independently_checked',job=job,rows=len(seen),raw_sha256=done['raw_sha256'],freeze_sha256=sha(S/'freeze.json'),
        summaries=summaries,scope='Separate post-hoc follow-up; complete jobs only. Fixed translated panel, no calibrated uncertainty or unseen-source inference. Behavior n4 diagnostic only; invalid Q answers retain all-attempt denominator.')
    save(d/'SCORING_COMPLETE.json',result)
    save(d/'INDEPENDENT_ACCEPTANCE.json',dict(status='passed_linewise_final_parser_and_monotone_transport_checks',rows=len(seen),summary_values_checked=checks,
        raw_sha256=done['raw_sha256'],freeze_sha256=sha(S/'freeze.json'),scoring_sha256=sha(d/'SCORING_COMPLETE.json'),
        evaluator_code_sha256=sha(Path(__file__)),independent_helper_sha256=sha(S/'upstream/independent_acceptance_v3.py'),
        scope='Different parsing and W1/weighting algorithms applied to every raw answer; software-level independent check, not an independent human review or empirical source.'))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True,choices=list(JOBS));a=ap.parse_args();s=score(a.job);print(json.dumps(dict(job=a.job,rows=s['rows'],status=s['status'])))
