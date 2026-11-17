"""CPU diagnostics of verified V4 outputs: tails, goal discrimination, coverage.

No new questions or model calls. Run --all-completed after each scoring update.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

R=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonl(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def table(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def quantile(values,p):
    values=sorted(values);x=(len(values)-1)*p;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def run(base,out):
    marker=json.loads((out/'scoring_complete.json').read_text(encoding='utf-8'))
    assert marker['status']=='independently_verified'
    files=[out/'scoring_complete.json',out/'scores/quartets.csv',out/'scores/prompts.csv',
           base/'inputs.jsonl',base/'goal_quartets.jsonl',base/'INVAR_dataset/data/pairs.jsonl']
    identity={str(p):sha(p) for p in files}
    dest=out/'diagnostics';dest.mkdir(exist_ok=True)
    ap=dest/'audit.json'
    if ap.exists():
        old=json.loads(ap.read_text(encoding='utf-8'))
        if old['input_sha256']==identity and old['script_sha256']==sha(Path(__file__)):return
    model=marker['model'];methods=marker['methods']
    inputs={p['prompt_id']:p for p in jsonl(base/'inputs.jsonl')}
    pairs={p['pair_id']:p for p in jsonl(base/'INVAR_dataset/data/pairs.jsonl')}
    rows=[r for r in table(out/'scores/quartets.csv') if r['model']==model and r['kind']=='behavior' and r['status']=='complete']
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['method'],row['pair_id'])].append(row)
    means=[]
    for (method,pair),group in groups.items():
        means.append(dict(method=method,pair_id=pair,source_id=group[0]['source_id'],task=group[0]['task'],
                          **{key:statistics.mean(float(r[key]) for r in group) for key in ['E','G','R']}))
    save(dest/'behavior_pair_means.csv',means)
    tails=[]
    for method in methods:
        subset=[r for r in means if r['method']==method]
        for metric in ['E','G']:
            xs=[r[metric] for r in subset]
            if not xs:continue
            tails.append(dict(method=method,metric=metric,n_pairs=len(xs),mean=statistics.mean(xs),
                median=quantile(xs,.5),p90=quantile(xs,.9),p95=quantile(xs,.95),maximum=max(xs),
                **{f'count_ge_{t}':sum(x>=t for x in xs) for t in [.05,.1,.2]}))
    save(dest/'behavior_tail.csv',tails)
    changes=[];idx={(r['method'],r['pair_id']):r for r in means}
    for left,right in [('baseline','anchor_sft'),('anchor_sft','anchor_js'),('baseline','anchor_js')]:
        if left not in methods or right not in methods:continue
        for metric in ['E','G']:
            ds=[idx[(right,p)][metric]-idx[(left,p)][metric] for m,p in idx if m==left and (right,p) in idx]
            changes.append(dict(comparison=right+' minus '+left,metric=metric,n_pairs=len(ds),
                improved=sum(d < -1e-12 for d in ds),worsened=sum(d > 1e-12 for d in ds),
                ties=sum(abs(d)<=1e-12 for d in ds),mean_delta=statistics.mean(ds)))
    save(dest/'behavior_change_counts.csv',changes)
    prompts={(r['method'],r['prompt_id']):r for r in table(out/'scores/prompts.csv')
             if r['model']==model and r['status']=='complete'}
    goal_groups=collections.defaultdict(dict)
    for q in jsonl(base/'goal_quartets.jsonl'):
        for tag,condition,variant in [('A',0,'base'),('B',0,q['variant_id']),('C',1,'base'),('D',1,q['variant_id'])]:
            pid=q[tag]
            if ('baseline',pid) not in prompts:continue
            key=(q['pair_id'],condition,variant)
            assert q['policy'] not in goal_groups[key] or goal_groups[key][q['policy']]==pid
            goal_groups[key][q['policy']]=pid
    contrasts=[]
    for (pair,condition,variant),policies in goal_groups.items():
        for a,b in itertools.combinations(sorted(policies),2):
            pa,pb=policies[a],policies[b]
            assert inputs[pa]['output_schema']==inputs[pb]['output_schema']
            if set(map(str,inputs[pa]['expected_actions'])) & set(map(str,inputs[pb]['expected_actions'])):continue
            for method in methods:
                x,y=prompts[(method,pa)],prompts[(method,pb)]
                ca,cb=float(x['mean_normalized_or_correct']),float(y['mean_normalized_or_correct'])
                contrasts.append(dict(method=method,pair_id=pair,condition=condition,variant_id=variant,
                    policy_a=a,policy_b=b,prompt_a=pa,prompt_b=pb,mean_correct=(ca+cb)/2,
                    worse_side=min(ca,cb),independent_calls_both_correct=ca*cb))
    save(dest/'goal_policy_contrasts.csv',contrasts)
    goal_summary=[]
    for method in methods:
        subset=[r for r in contrasts if r['method']==method]
        if not subset:continue
        grouped=collections.defaultdict(list)
        for row in subset:grouped[row['pair_id']].append(row)
        goal_summary.append(dict(method=method,n_pairs=len(grouped),contrast_references=len(subset),
            unique_prompt_pairs=len({(r['prompt_a'],r['prompt_b']) for r in subset}),
            **{k:statistics.mean(statistics.mean(r[k] for r in g) for g in grouped.values())
               for k in ['mean_correct','worse_side','independent_calls_both_correct']}))
    save(dest/'goal_policy_summary.csv',goal_summary)
    # Actual exposure and prepared-pool task/role coverage, not assumed source coverage.
    coverage=[]
    for method in methods:
        if method=='baseline':continue
        identity_path=out/model/(method+'.run_identity.json')
        run_identity=json.loads(identity_path.read_text(encoding='utf-8'))
        training=Path(run_identity['config']['adapter']['path']).parent
        if not training.is_dir():continue
        pool=jsonl(training/'supervised_inputs.jsonl')+jsonl(training/'free_inputs.jsonl')
        exposure=json.loads((training/'actual_exposures.json').read_text(encoding='utf-8'))['prompt_exposure_counts']
        pool_pairs={p for r in pool for p in r['pair_references']}
        seen_pairs={p for r in pool if exposure.get(r['prompt_id'],0)>0 for p in r['pair_references']}
        for row in means:
            if row['method']!=method:continue
            test=pairs[row['pair_id']]
            coverage.append(dict(method=method,pair_id=row['pair_id'],source_id=test['source_id'],task=test['task'],role=test['role'],
                prepared_pool_same_task=any(pairs[p]['task']==test['task'] for p in pool_pairs),
                actual_exposure_same_task=any(pairs[p]['task']==test['task'] for p in seen_pairs),
                actual_exposure_same_task_role=any((pairs[p]['task'],pairs[p]['role'])==(test['task'],test['role']) for p in seen_pairs)))
    save(dest/'actual_task_coverage.csv',coverage)
    audit=dict(status='verified_score_diagnostics',input_sha256=identity,script_sha256=sha(Path(__file__)),
               model=model,methods=methods,goal_applicable=bool(contrasts),
               scope='Fixed observed inputs; policy products are independent-call estimates, not measured trajectories; contrasts may share endpoints',
               no_new_model_calls=True)
    ap.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(diagnosed=str(out),goal_applicable=bool(contrasts),methods=methods)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--all-completed',action='store_true')
    a=p.parse_args()
    if a.all_completed:
        for marker in sorted((R/'evaluations').rglob('scoring_complete.json')):run(a.baseline,marker.parent)
    else:
        assert a.out;run(a.baseline,a.out)
