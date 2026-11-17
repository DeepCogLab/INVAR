"""Audit every planned attempt; invalid answers remain in all-attempt accuracy."""
import argparse
import collections
import csv
import json
import statistics
from campaign_common import R, MODELS, arms, digest, sha, load, jl, save, messages_for
from science_worker import plan
from parsing import parse_final
from behavior_metrics import metrics, aggregate, w1


def mean(xs):return statistics.mean(xs) if xs and all(x is not None for x in xs) else None


def controls(qs,counts,n,key=None):
    pairs=collections.defaultdict(list)
    for q in qs:
        pairs[q['pair_id']].append(mean([counts[q[k]]['correct']/n for k in 'ABCD']))
    groups=collections.defaultdict(list)
    metadata={q['pair_id']:q for q in qs}
    for pid,v in pairs.items():groups[metadata[pid][key] if key else 'all'].append(mean(v))
    return mean([mean(v) for v in groups.values()])


def score(job):
    cfg=load(R/'freeze.json');protocol=cfg['protocol'];d=R/'runs'/job
    for name,h in cfg['files'].items():assert sha(R/name)==h,name
    ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')}
    labels={p['prompt_id']:p for p in jl(R/'frozen/evaluation.jsonl')}
    trials=plan(job,list(ps.values()),protocol);expected={(t['prompt_id'],t['arm'],t['replicate']):t for t in trials}
    identity=load(d/'identity.json');complete=load(d/'COMPLETE.json');raw=d/'responses.jsonl'
    identity_hash=digest(identity)
    assert identity['freeze_sha256']==sha(R/'freeze.json') and identity['job']==job
    assert identity['trial_plan_sha256']==digest(trials) and identity['expected_rows']==len(trials)
    assert sha(raw)==complete['raw_sha256'] and complete['rows']==len(trials)
    seen=set();counts=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    values=collections.defaultdict(lambda:collections.defaultdict(list));attempts=[];truth={};token_counts=collections.defaultdict(list)
    with raw.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);pid,arm,rep=r['prompt_id'],r['arm'],r['replicate'];key=pid,arm,rep
            assert key in expected and key not in seen;seen.add(key)
            assert r['model']==job and r['identity_sha256']==identity_hash
            assert r['sampling_seed']==expected[key]['sampling_seed']
            assert r['messages_sha256']==digest(messages_for(ps[pid],MODELS[job]['kind'],arm))
            assert r['rendered_sha256']==identity['rendered_sha256'][pid+'__'+arm]
            assert r['prompt_token_ids_sha256']==identity['prompt_token_ids_sha256'][pid+'__'+arm]
            assert r['completion_tokens']==len(r['token_ids'])
            obj,error,strict=parse_final(r['response_text'],r['response_with_special_tokens'],r['finish_reason'],ps[pid]['output_schema'],MODELS[job]['kind'],arm,identity['prefilled_think'][pid+'__'+arm])
            p=ps[pid];label=labels[pid];c=counts[arm][pid]
            c['attempts']+=1;c['valid']+=obj is not None;c['strict_format']+=strict
            c['truncated']+=r['finish_reason']=='length';c['completion_tokens']+=r['completion_tokens']
            c['error_'+str(error)]+=error is not None
            x=None;correct=False
            if obj is not None:
                if p['task_type']=='behavior':
                    outcome=label['outcome'];action=obj['action']
                    x=float(action==outcome['target_action']) if outcome['kind']=='probability' else action/outcome['denominator']
                    assert 0<=x<=1;values[arm][pid].append(x)
                elif p['task_type']=='goal':correct=obj['action'] in label['expected_actions']
                else:correct=obj['answer']==label['expected_answer']
            c['correct']+=correct;truth[key]=correct
            token_counts[arm,p['language'],p['stage'],p['task_type']].append(r['completion_tokens'])
            attempts.append(dict(prompt_id=pid,arm=arm,replicate=rep,valid=obj is not None,
                strict_format=strict,error=error,correct=correct if p['task_type']!='behavior' else None,
                action_json=json.dumps(obj['action']) if obj is not None and p['task_type']=='behavior' else None,
                normalized_outcome=x,completion_tokens=r['completion_tokens']))
    assert seen==set(expected)
    quartet_rows=[];summaries=[]
    bqs=jl(R/'frozen/core/behavior_quartets.jsonl');gqs=jl(R/'frozen/core/goal_quartets.jsonl')
    fqs=jl(R/'frozen/core/fact_quartets.jsonl');pqs=jl(R/'frozen/parameter/goal_quartets.jsonl')
    for arm in arms(job):
        for lang in ('zh','en'):
            n=protocol['repeats']['behavior'];qs=[q for q in bqs if q['language']==lang]
            br=[dict(arm=arm,**q,**metrics(q,values[arm],n)) for q in qs];quartet_rows.extend(br)
            summary=dict(arm=arm,language=lang,behavior_pair_equal=aggregate(br),behavior_source_equal=aggregate(br,'source_id'),behavior_task_equal=aggregate(br,'task'))
            # The new panel is source-balanced; legacy composition is explicitly a reweighting sensitivity.
            by_source={s:aggregate([r for r in br if r['source_id']==s]) for s in ['HPVBG','IGVOFO','PGG','PIQ572']}
            old_weights={'HPVBG':8,'IGVOFO':8,'PGG':22,'PIQ572':54}
            summary['behavior_legacy_source_mix_sensitivity']={k:sum(by_source[s][k]*w for s,w in old_weights.items())/92 if all(by_source[s][k] is not None for s in old_weights) else None for k in by_source['HPVBG']}
            cqs=[q for q in gqs if q['language']==lang]
            summary['goal_pair_equal_all_attempt_Q']=controls(cqs,counts[arm],protocol['repeats']['goal'])
            summary['goal_source_equal_all_attempt_Q']=controls(cqs,counts[arm],protocol['repeats']['goal'],'source_id')
            summary['goal_task_equal_all_attempt_Q']=controls(cqs,counts[arm],protocol['repeats']['goal'],'task')
            for typ in ('changed_fact','stable_fact'):
                qs2=[q for q in fqs if q['language']==lang and q['probe_type']==typ]
                summary[typ+'_pair_equal_all_attempt_U']=controls(qs2,counts[arm],protocol['repeats'][typ])
            for stage,qs2 in [('core',cqs),('parameter',[q for q in pqs if q['language']==lang])]:
                disjoint=[q for q in qs2 if set(labels[q['A']]['expected_actions']).isdisjoint(labels[q['C']]['expected_actions'])]
                ng=protocol['repeats']['parameter' if stage=='parameter' else 'goal']
                summary[stage+'_disjoint_goal_quartets']=len(disjoint)
                summary[stage+'_disjoint_joint_all_four_correct']=mean([mean([float(all(truth[q[k],arm,rep] for k in 'ABCD')) for rep in range(ng)]) for q in disjoint])
                if stage=='parameter':summary['parameter_pair_equal_all_attempt_Q']=controls(qs2,counts[arm],ng)
            details={}
            for stage,typ in [('core',t) for t in ['behavior','goal','changed_fact','stable_fact']]+[('parameter','goal')]:
                ids=[pid for pid,p in ps.items() if p['stage']==stage and p['task_type']==typ and p['language']==lang]
                cc=sum((counts[arm][pid] for pid in ids),collections.Counter());tc=token_counts[arm,lang,stage,typ]
                details[stage+'_'+typ]=dict(cc,delivery_rate=cc['valid']/cc['attempts'],
                    unique_prompt_all_attempt_accuracy=cc['correct']/cc['attempts'] if typ!='behavior' else None,
                    mean_completion_tokens=statistics.mean(tc),p95_completion_tokens=sorted(tc)[min(len(tc)-1,int(.95*len(tc)))])
            summary['delivery_and_cost']=details
            # Finite-draw diagnostic: split each prompt's draws into even/odd replicates.
            bypid={p['prompt_id']:[a for a in attempts if a['prompt_id']==p['prompt_id'] and a['arm']==arm] for p in ps.values() if p['task_type']=='behavior' and p['language']==lang}
            noise=[]
            for pid,rs in bypid.items():
                halves=[[r['normalized_outcome'] for r in rs if r['replicate']%2==h and r['valid']] for h in (0,1)]
                noise.append(w1(*halves))
            summary['same_prompt_split_half_W1_conditional']=mean(noise)
            fixed={}
            for task in sorted({labels[pid]['task'] for pid in bypid}):
                rows=[r for pid,rs in bypid.items() if labels[pid]['task']==task for r in rs if r['valid']]
                ac=collections.Counter(r['action_json'] for r in rows)
                fixed[task]=dict(valid=len(rows),dominant_action=ac.most_common(1),
                    fraction_of_valid=ac.most_common(1)[0][1]/len(rows) if rows else None)
            summary['behavior_action_concentration_by_task']=fixed
            summary['split_half_scope']='Within-prompt sampling-noise diagnostic at half draw count; not a bias correction or a calibrated confidence interval.'
            summaries.append(summary)
    for name,rows in [('attempt_scores.csv',attempts),('behavior_quartet_scores.csv',quartet_rows)]:
        with (d/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='passed_complete_trial_audit',job=job,rows=len(seen),raw_sha256=sha(raw),
        freeze_sha256=sha(R/'freeze.json'),summaries=summaries,
        scope='Fixed 32-pair source-balanced panel and 20-pair parametric diagnostic; AI translated English. Conditional W1 excludes invalid outputs, all-attempt missingness bounds and accuracy retain them. Pair/source equality is by construction. No inference to unseen studies or human behavior.')
    save(d/'SCORING_COMPLETE.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args();result=score(a.job)
    print(json.dumps(dict(job=a.job,status=result['status'],rows=result['rows'])))
