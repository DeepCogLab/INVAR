"""Add a descriptive conditional permutation reference without altering primary scores."""
import argparse
import collections
import csv
import random
import statistics
from pathlib import Path
from audit_new_runs import R,load,jl,sha,save,digest
from score_joint_reasoning_v2 import w1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    d=R/'runs'/a.job/'joint_reasoning';out=d/'conditional_noise_reference.json'
    assert not out.exists(),'Keep an existing derived reference immutable'
    audit=load(d/'independent_scalar_audit.json');cfg=load(R/'joint_reasoning_freeze_v2.json')
    assert audit['status']=='passed_independent_scalar' and audit['raw_sha256']==load(d/'COMPLETE.json')['sha256']
    qs=jl(R/'joint_frozen/quartets.jsonl');values=collections.defaultdict(lambda:collections.defaultdict(list));seen=set()
    rows=list(csv.DictReader((d/'attempt_scores.csv').open(encoding='utf-8-sig')))
    for r in rows:
        key=(r['prompt_id'],r['arm'],int(r['replicate']));assert key not in seen;seen.add(key)
        if r['valid']=='True':values[r['arm']][r['prompt_id']].append(float(r['normalized_outcome']))
    ids={q[s] for q in qs for s in 'ABCD'}
    assert seen=={(pid,arm,rep) for pid in ids for arm in cfg['arms'] for rep in range(cfg['repeats'])}
    summaries=[]
    for arm in cfg['arms']:
        cache={};pairs=collections.defaultdict(list)
        for q in qs:
            points=[];refs=[]
            for edge in ['AB','CD']:
                key=tuple(sorted(q[s] for s in edge))
                if key not in cache:
                    x,y=(values[arm][pid] for pid in key);point=w1(x,y)
                    if point is None:cache[key]=(None,None)
                    elif key[0]==key[1]:cache[key]=(0.,0.)
                    else:
                        pool=x+y;ds=[];rng=random.Random(int(digest([20261111,'joint_E_noise',a.job,arm,key])[:16],16))
                        for _ in range(199):
                            rng.shuffle(pool);ds.append(w1(pool[:len(x)],pool[len(x):]))
                        cache[key]=(point,statistics.mean(ds))
                point,ref=cache[key];points.append(point);refs.append(ref)
            pairs[q['pair_id']].append((statistics.mean(points) if all(x is not None for x in points) else None,
                statistics.mean(refs) if all(x is not None for x in refs) else None))
        complete=all(x is not None for rs in pairs.values() for row in rs for x in row)
        point=statistics.mean(statistics.mean(x[0] for x in rs) for rs in pairs.values()) if complete else None
        ref=statistics.mean(statistics.mean(x[1] for x in rs) for rs in pairs.values()) if complete else None
        recorded=next(s['pair_equal']['E_conditional'] for s in audit['summaries'] if s['arm']==arm)
        assert (point is None and recorded is None) or abs(point-recorded)<1e-10
        summaries.append(dict(arm=arm,E_conditional=point,E_conditional_permutation_reference=ref,n_pairs=len(pairs),unique_edges=len(cache)))
    result=dict(status='completed_descriptive_reference',job=a.job,summaries=summaries,resamples=199,
        input_attempt_scores_sha256=sha(d/'attempt_scores.csv'),primary_audit_sha256=sha(d/'independent_scalar_audit.json'),script_sha256=sha(Path(__file__)),
        scope='Pair-equal pooled-label permutation mean for E on parseable outputs only; descriptive finite-draw reference, not a p value, unbiased correction, confidence interval, or protection against missing-output selection. Shared prompt IDs and repeated edges are cached, not treated as independent N.')
    save(out,result);print(__import__('json').dumps(result))


if __name__=='__main__':main()
