"""Score new trials and conservatively parse completed native reasoning traces."""
import argparse
import collections
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from run_api import R,load,jl,save,digest,sha,valid

BASE=Path('/workspace/project_social_simulation/invar_reliability_runs/20261024_v2')


def native_final(text,schema,finish):
    if finish=='length':
        return None
    text=text.strip()
    if text.startswith('<think>'):
        if text.count('</think>')!=1:
            return None
        text=text.split('</think>',1)[1].strip()
    # Parse only the final response outside a completed reasoning block.
    # Never select an arbitrary JSON fragment from the reasoning itself.
    if re.fullmatch(r'<Answer>\s*.*?\s*</Answer>',text,re.S|re.I):
        text=re.sub(r'^<Answer>\s*|\s*</Answer>$','',text,flags=re.I)
    if re.fullmatch(r'```(?:json)?\s*.*?\s*```',text,re.S):
        text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
    return text.strip() if valid(text.strip(),schema,finish) else None


def write(p,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def weighted(rows,key):
    groups=collections.defaultdict(list)
    for r in rows:
        groups[r['pair_id']].append(r[key])
    return sum(sum(a)/len(a) for a in groups.values())/len(groups) if groups else None


def diagnostic(job,stage):
    d=R/'runs'/job/stage
    complete=load(d/'COMPLETE.json')
    raw=d/'responses.jsonl'
    assert sha(raw)==complete['sha256']
    ps={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')}
    allrows=jl(raw)
    stats=collections.defaultdict(lambda:dict(n=0,valid=0,correct=0,truncated=0))
    for row in allrows:
        p=ps[row['prompt_id']]
        final=native_final(row['response_text'],p['output_schema'],row['finish_reason']) if stage=='native' else row['response_text'] if row['schema_valid'] else None
        s=stats[(row['prompt_id'],row['batch'])]
        s['n']+=1;s['valid']+=final is not None;s['truncated']+=row['finish_reason']=='length'
        if final is not None:
            value=json.loads(final)['action']
            s['correct']+=value in p['expected_actions']
    prompt_rows=[dict(prompt_id=pid,arm=arm,**s,accuracy_all_attempts=s['correct']/s['n']) for (pid,arm),s in sorted(stats.items())]
    write(d/'prompt_scores.csv',prompt_rows)
    qrows=[]
    for q in jl(BASE/'goal_quartets.jsonl'):
        for arm in sorted({key[1] for key in stats}):
            if all((q[t],arm) in stats for t in 'ABCD'):
                ac=[stats[(q[t],arm)]['correct']/stats[(q[t],arm)]['n'] for t in 'ABCD']
                qrows.append(dict(pair_id=q['pair_id'],quartet_id=q['quartet_id'],task=q['task'],policy=q['policy'],variant=q['variant_id'],arm=arm,Q=sum(ac)/4,worst_side=min(ac)))
    write(d/'quartet_scores.csv',qrows)
    summaries=[]
    for arm in sorted({key[1] for key in stats}):
        selected=[r for r in prompt_rows if r['arm']==arm]
        qs=[q for q in qrows if q['arm']==arm]
        n=sum(r['n'] for r in selected)
        summaries.append(dict(job=job,stage=stage,arm=arm,n=n,n_prompts=len(selected),
            pair_equal_Q=weighted(qs,'Q'),pair_equal_worst_side=weighted(qs,'worst_side'),
            prompt_equal_Q=sum(r['accuracy_all_attempts'] for r in selected)/len(selected),
            valid=sum(r['valid'] for r in selected),truncated=sum(r['truncated'] for r in selected),
            invalid_or_truncated_as_failure=True,conditional_parse_accuracy=sum(r['correct'] for r in selected)/max(1,sum(r['valid'] for r in selected))))
    write(d/'summary.csv',summaries)
    save(d/'SCORING_COMPLETE.json',dict(rows=len(allrows),raw_sha256=sha(raw),scorer_sha256=sha(Path(__file__)),
        scope='All attempts counted; native sample is prompt-weighted diagnostic, not full-benchmark Q.',summaries=summaries))


def main_score(job):
    d=R/'runs'/job/'main'
    complete=load(d/'COMPLETE.json')
    assert not complete['invalid'],'Invalid outputs retained; formal score blocked'
    assert sha(d/'responses.jsonl')==complete['sha256']
    s=d/'scoring';s.mkdir(exist_ok=True)
    for n in ['inputs.jsonl','fact_quartets.jsonl','goal_quartets.jsonl','INVAR_dataset']:
        target=s/n
        if not target.exists():
            target.symlink_to(BASE/n,target_is_directory=(n=='INVAR_dataset'))
    output=s/'outputs/main';output.mkdir(parents=True,exist_ok=True)
    projected=output/f'{job}.jsonl'
    plan=s/'main_plan.jsonl'
    # A fresh deterministic projection is derived from immutable generated rows.
    with projected.open('w',encoding='utf-8') as dst,plan.open('w',encoding='utf-8') as pf:
        for row in jl(d/'responses.jsonl'):
            row['trial_id']=digest([job,row['prompt_id'],row['batch'],row['replicate']])[:24]
            dst.write(json.dumps(row,ensure_ascii=False)+'\n')
            pf.write(json.dumps({k:row[k] for k in ['model','prompt_id','batch','replicate','trial_id']})+'\n')
    save(s/'frozen_manifest.json',dict(input_sha256=sha(BASE/'inputs.jsonl'),models=[job],plan_sha256=sha(plan)))
    sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'INVAR_dataset'))
    spec=importlib.util.spec_from_file_location('frozen_score',BASE/'score_v2.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    scorer.R=s
    sys.argv=['score_v2.py','--model',job]
    scorer.main()
    save(d/'SCORING_COMPLETE.json',dict(rows=complete['rows'],scorer_sha256=sha(BASE/'score_v2.py'),
        report=str(s/'reports'/job),scope='Frozen original scorer used without scientific modifications.'))


def rescore_old_native():
    old=Path('/workspace/project_social_simulation/invar_social_cognition_runs/20261104')
    ps={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')}
    summaries=[]
    details=[]
    for file in sorted((old/'runs').glob('*/native/responses.jsonl')):
        job=file.parent.parent.name
        groups=collections.defaultdict(lambda:dict(n=0,strict_valid=0,revised_valid=0,correct=0,truncated=0))
        for row in jl(file):
            p=ps[row['prompt_id']]
            final=native_final(row['response_text'],p['output_schema'],row['finish_reason'])
            s=groups[p['task_type']]
            s['n']+=1;s['strict_valid']+=row['schema_valid'];s['revised_valid']+=final is not None
            s['truncated']+=row['finish_reason']=='length'
            correct=None
            if final is not None and p['task_type']!='behavior':
                v=next(iter(json.loads(final).values()))
                correct=v in p['expected_actions'] if p['task_type']=='goal' else v==p['expected_answer']
                s['correct']+=correct
            details.append(dict(job=job,prompt_id=p['prompt_id'],task_type=p['task_type'],strict_valid=row['schema_valid'],
                revised_valid=final is not None,correct=correct,final=final,finish=row['finish_reason']))
        for typ,s in groups.items():
            summaries.append(dict(job=job,task_type=typ,**s,accuracy_all_attempts=s['correct']/s['n'] if typ!='behavior' else None,
                scope='12 fixed prompts per type, one call each; exploratory native diagnostic only',raw_sha256=sha(file)))
    out=R/'old_native_rescore';out.mkdir(exist_ok=True)
    write(out/'summary.csv',summaries);write(out/'details.csv',details)
    save(out/'audit.json',dict(rows=len(details),models=len({r['job'] for r in details}),
        scorer_sha256=sha(Path(__file__)),old_files_modified=0,
        change='Parse final exact JSON/codefence/Answer after one completed initial think block. No arbitrary JSON extraction.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job');ap.add_argument('--stage',choices=['main','goal_diagnostic','native','old_native'],required=True)
    a=ap.parse_args()
    if a.stage=='old_native':
        rescore_old_native()
    elif a.stage=='main':
        main_score(a.job)
    else:
        diagnostic(a.job,a.stage)
