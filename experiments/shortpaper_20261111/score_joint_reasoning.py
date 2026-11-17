"""Audit the fixed native behavior experiment, retaining missing-output uncertainty."""
import argparse
import collections
import csv
import statistics
from pathlib import Path
from audit_new_runs import R,jl,load,sha,save,digest,parse_final


def w1(a,b):
    if not a or not b:return None
    ca,cb=collections.Counter(a),collections.Counter(b);xs=sorted(set(ca)|set(cb))
    fa=fb=value=0.
    for x,y in zip(xs,xs[1:]):
        fa+=ca[x]/len(a);fb+=cb[x]/len(b);value+=abs(fa-fb)*(y-x)
    return value


def w1_check(a,b):
    # Independent quantile-grid calculation, including unequal parse counts.
    if not a or not b:return None
    a=sorted(a);b=sorted(b);cuts=sorted({i/len(a) for i in range(len(a)+1)}|{i/len(b) for i in range(len(b)+1)})
    return sum((y-x)*abs(a[min(int(((x+y)/2)*len(a)),len(a)-1)]-b[min(int(((x+y)/2)*len(b)),len(b)-1)]) for x,y in zip(cuts,cuts[1:]))


def distance(a,b,n):
    point=w1(a,b)
    if point is None:return None,0.,1.
    assert abs(point-w1_check(a,b))<1e-10
    missing=2-(len(a)+len(b))/n
    return point,max(0.,point-missing),min(1.,point+missing)


def metrics(q,values,n):
    v={k:values[q[k]] for k in 'ABCD'}
    result={}
    for label,edges in [('E',['AB','CD']),('R',['AC','BD'])]:
        ds=[distance(v[a],v[b],n) if q[a]!=q[b] else (0.,0.,0.) for a,b in edges]
        result[label+'_conditional']=statistics.mean(d[0] for d in ds) if all(d[0] is not None for d in ds) else None
        for i,suffix in [(1,'low'),(2,'high')]:result[label+'_'+suffix]=statistics.mean(d[i] for d in ds)
    coeff=collections.Counter()
    for k,c in [('A',1),('B',-1),('C',-1),('D',1)]:coeff[q[k]]+=c
    lo=hi=0.
    for pid,c in coeff.items():
        vl=sum(values[pid])/n;vh=vl+(n-len(values[pid]))/n
        lo+=c*(vl if c>=0 else vh);hi+=c*(vh if c>=0 else vl)
    result['G_low']=0. if lo<=0<=hi else min(abs(lo),abs(hi))
    result['G_high']=min(max(abs(lo),abs(hi)),2*result['E_high'])
    result['G_conditional']=abs(sum(c*statistics.mean(values[pid]) for pid,c in coeff.items() if c)) if all(values[pid] for pid,c in coeff.items() if c) else None
    if result['G_conditional'] is not None:assert result['G_conditional']<=2*result['E_conditional']+1e-10
    return result


def aggregate(rows,key=None):
    # First average variants within each condition pair; sensitivity weights pairs
    # equally within source or task, then groups equally.
    pairs=collections.defaultdict(list)
    for r in rows:pairs[r['pair_id']].append(r)
    cols=[k for k in rows[0] if k.startswith(('E_','G_','R_'))]
    pp=[]
    for rs in pairs.values():
        p={k:rs[0][k] for k in ['pair_id','source_id','task']}
        p.update({k:statistics.mean(r[k] for r in rs) if all(r[k] is not None for r in rs) else None for k in cols});pp.append(p)
    groups=collections.defaultdict(list)
    for p in pp:groups[p[key] if key else 'all'].append(p)
    return {k:statistics.mean(statistics.mean(p[k] for p in ps) for ps in groups.values()) if all(p[k] is not None for p in pp) else None for k in cols}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    cfg=load(R/'joint_reasoning_freeze.json');d=R/'runs'/a.job/'joint_reasoning'
    for name,h in cfg['files'].items():assert sha(R/'joint_frozen'/name)==h
    ps={p['prompt_id']:p for p in jl(R/'joint_frozen/model_inputs.jsonl')}
    labels={p['prompt_id']:p for p in jl(R/'joint_frozen/evaluation.jsonl')}
    qs=jl(R/'joint_frozen/quartets.jsonl');n=cfg['repeats']
    expected={(pid,arm,rep) for pid in ps for arm in cfg['arms'] for rep in range(n)}
    identity=load(d/'identity.json');complete=load(d/'COMPLETE.json');raw=d/'responses.jsonl'
    assert identity['protocol']==cfg and identity['job']==a.job
    assert identity['runner_sha256']==cfg['runtime_sha256']['offline_joint_reasoning.py']
    assert sha(raw)==complete['sha256'] and complete['rows']==len(expected)
    values=collections.defaultdict(lambda:collections.defaultdict(list));seen=set();counts=collections.defaultdict(collections.Counter);attempts=[]
    for r in jl(raw):
        pid,arm,rep=r['prompt_id'],r['batch'],r['replicate'];key=pid,arm,rep
        assert key in expected and key not in seen;seen.add(key)
        assert r['model']==a.job and r['status']=='ok' and r['config_sha256']==digest(identity)
        assert r['messages_sha256']==digest(ps[pid]['messages'])
        assert r['sampling_seed']==int(digest([20261111,'joint',pid,rep])[:8],16)%(2**31-1)
        obj,error=parse_final(r['response_text'],r['finish_reason'],ps[pid]['output_schema'])
        g=counts[arm];g['n']+=1;g['valid']+=obj is not None;g['truncated']+=r['finish_reason']=='length';g['completion_tokens']+=r['completion_tokens']
        x=None
        if obj is not None:
            action=obj['action'];o=labels[pid]['outcome'];space=labels[pid]['action_space']
            assert ('values' in space and any(type(action) is type(v) and action==v for v in space['values'])) or ('min' in space and type(action) is int and space['min']<=action<=space['max'])
            x=float(action==o['target_action']) if o['kind']=='probability' else action/o['denominator']
            assert 0<=x<=1;values[arm][pid].append(x)
        attempts.append(dict(prompt_id=pid,arm=arm,replicate=rep,valid=obj is not None,error=error,normalized_outcome=x,completion_tokens=r['completion_tokens']))
    assert seen==expected
    quartet_rows=[];summaries=[]
    for arm in cfg['arms']:
        rs=[dict(arm=arm,**q,**metrics(q,values[arm],n)) for q in qs];quartet_rows.extend(rs)
        summaries.append(dict(arm=arm,**counts[arm],parse_rate=counts[arm]['valid']/counts[arm]['n'],pair_equal=aggregate(rs),source_equal=aggregate(rs,'source_id'),task_equal=aggregate(rs,'task')))
    delta={k:summaries[1]['pair_equal'][k]-summaries[0]['pair_equal'][k] if summaries[1]['pair_equal'][k] is not None and summaries[0]['pair_equal'][k] is not None else None for k in summaries[0]['pair_equal']}
    result=dict(status='passed_independent_scalar',stage='joint_reasoning',job=a.job,rows=len(seen),raw_sha256=complete['sha256'],scorer_sha256=sha(Path(__file__)),trial_plan_identity_seed_messages_verified=True,
        summaries=summaries,thinking_minus_no_thinking_conditional_and_endpoint_differences=delta,
        scope='15 fixed condition pairs, two strict variants, 87 unique prompts, 128 draws each. Conditional metrics exclude missing responses; low/high bound all attempted outcomes on [0,1]. Bounds do not quantify finite-sampling error. R has no preferred direction; endpoint differences are not a bound on the difference.',
        delta_all_attempt_bounds={m:[summaries[1]['pair_equal'][m+'_low']-summaries[0]['pair_equal'][m+'_high'],summaries[1]['pair_equal'][m+'_high']-summaries[0]['pair_equal'][m+'_low']] for m in ['E','G','R']})
    for name,rows in [('attempt_scores.csv',attempts),('quartet_scores.csv',quartet_rows)]:
        with (d/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(d/'SCORING_COMPLETE.json',result);save(d/'independent_scalar_audit.json',result)
    print(__import__('json').dumps(result))


if __name__=='__main__':main()
