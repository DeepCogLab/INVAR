"""Cross-check all completed point estimates against a separate scalar evaluator."""
import argparse,collections,csv,json,math,statistics,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'INVAR_dataset'))
from evaluate import w1,normalize

def jl(path):
    with path.open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def csvrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def same(a,b,label):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),label

ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);args=ap.parse_args()
out=R/'reports'/args.model
assert json.loads((out/'completion_audit.json').read_text())['status']=='passed'
inputs={x['prompt_id']:x for x in jl(R/'inputs.jsonl')}
values=collections.defaultdict(list);correct=collections.defaultdict(list);counts=collections.Counter()
with (R/'outputs/main'/(args.model+'.jsonl')).open(encoding='utf-8') as f:
    for line in f:
        row=json.loads(line);p=inputs[row['prompt_id']];obj=json.loads(row['response_text']);counts[row['batch']]+=1
        if p['task_type']=='behavior':
            if row['batch']=='main':values[p['prompt_id']].append(normalize(obj['action'],p['outcome']))
        else:correct[p['prompt_id']].append(obj['action'] in p['expected_actions'] if p['task_type']=='goal' else obj['answer']==p['expected_answer'])
points={x['quartet_id']:x for x in csvrows(out/'behavior_quartets.csv')}
qs=jl(R/'INVAR_dataset/data/quartets.jsonl');by_pair=collections.defaultdict(list)
for q in qs:
    a,b,c,d=[values[q[tag]] for tag in 'ABCD']
    assert all(len(v)==256 for v in (a,b,c,d))
    metrics=dict(E=(w1(a,b)+w1(c,d))/2,R=(w1(a,c)+w1(b,d))/2,
                 delta_original=statistics.mean(c)-statistics.mean(a),delta_variant=statistics.mean(d)-statistics.mean(b))
    metrics['G']=abs(metrics['delta_original']-metrics['delta_variant'])
    for key,value in metrics.items():same(value,points[q['quartet_id']][key],q['quartet_id']+':'+key)
    by_pair[q['pair_id']].append(metrics)
overall=next(x for x in csvrows(out/'behavior_summary.csv') if x['group']=='all' and x['value']=='all')
for key in ('E','R','G','delta_original','delta_variant'):
    same(statistics.mean(statistics.mean(x[key] for x in group) for group in by_pair.values()),overall[key],'overall:'+key)
control_points={x['quartet_id']:x for x in csvrows(out/'control_quartets.csv')}
acc={pid:sum(v)/len(v) for pid,v in correct.items()};ncontrols=0
for filename in ('fact_quartets.jsonl','goal_quartets.jsonl'):
    for q in jl(R/filename):
        p=[acc[q[tag]] for tag in 'ABCD'];s=control_points[q['quartet_id']]
        checks=dict(accuracy=sum(p)/4,worst_side=min(p),dual_original=p[0]*p[2],dual_variant=p[1]*p[3],
                    min_original=min(p[0],p[2]),min_variant=min(p[1],p[3]))
        for key,value in checks.items():same(value,s[key],q['quartet_id']+':'+key)
        ncontrols+=1
report=dict(model=args.model,status='passed',behavior_quartets_checked=len(qs),control_quartets_checked=ncontrols,
            response_rows_scanned=sum(counts.values()),method='Scalar CDF W1 implementation and direct arithmetic over raw responses; no reuse of vectorized point estimates')
report['completion_audit_sha256']=hashlib.sha256((out/'completion_audit.json').read_bytes()).hexdigest()
report['raw_sha256']=json.loads((out/'completion_audit.json').read_text())['raw_sha256']
(out/'independent_recompute.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
