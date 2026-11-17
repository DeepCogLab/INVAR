"""Check all accepted behavior extension aggregates and expose every source/task slice."""
import collections
import csv
import json
import math
from pathlib import Path
from analyze_existing import OUT,write
from audit_new_runs import sha


def mean(values):
    return sum(values)/len(values) if values and all(v is not None for v in values) else None


def main():
    root=Path('artifacts/shortpaper_20261111');slices=[];proof=[]
    index=json.loads((root/'mirror_index.json').read_text(encoding='utf-8'))
    for stage in index['stages']:
        if Path(stage).name!='joint_reasoning':continue
        job=Path(stage).parts[1];d=root/stage
        ap=root/'local_scalar_audits'/(job+'_joint_reasoning.json')
        if not ap.exists():continue
        audit=json.loads(ap.read_text(encoding='utf-8'));assert audit['status']=='passed_independent_scalar'
        assert audit['raw_sha256']==index['files'][stage+'/responses.jsonl']['sha256']
        p=d/'quartet_scores.csv';assert sha(p)==index['files'][stage+'/quartet_scores.csv']['sha256']
        rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
        cols=[k for k in rows[0] if k.startswith(('E_','G_','R_'))]
        for r in rows:
            for k in cols:r[k]=float(r[k]) if r[k] else None
        points={}
        for arm in ['native_no_thinking','native_thinking']:
            pairs=collections.defaultdict(list)
            for r in rows:
                if r['arm']==arm:pairs[r['pair_id']].append(r)
            assert len(pairs)==15 and all(len(v)==2 for v in pairs.values())
            pp=[dict(pair_id=pid,source_id=rs[0]['source_id'],task=rs[0]['task'],**{k:mean([r[k] for r in rs]) for k in cols}) for pid,rs in pairs.items()]
            for dim in ['all','source_id','task']:
                groups=collections.defaultdict(list)
                for r in pp:groups[r[dim] if dim!='all' else 'all'].append(r)
                means={key:{k:mean([r[k] for r in rs]) for k in cols} for key,rs in groups.items()}
                recorded=next(s for s in audit['summaries'] if s['arm']==arm)[{'all':'pair_equal','source_id':'source_equal','task':'task_equal'}[dim]]
                for k in cols:
                    value=mean([r[k] for r in means.values()]);old=recorded[k]
                    assert (value is None and old is None) or math.isclose(value,old,abs_tol=1e-12),(job,dim,arm,k)
                for key,m in sorted(means.items()):points[arm,dim,key]=(m,len(groups[key]))
        for dim,key in sorted({(dim,key) for arm,dim,key in points}):
            off,n=points['native_no_thinking',dim,key];on,n2=points['native_thinking',dim,key];assert n==n2
            r=dict(job=job,group=dim,value=key,n_condition_pairs=n)
            for metric in ['E','G','R']:
                for mode,values in [('off',off),('on',on)]:
                    for suffix in ['conditional','low','high']:r[f'{mode}_{metric}_{suffix}']=values[f'{metric}_{suffix}']
                r[f'delta_{metric}_conditional']=on[metric+'_conditional']-off[metric+'_conditional'] if on[metric+'_conditional'] is not None and off[metric+'_conditional'] is not None else None
                r[f'delta_{metric}_low']=on[metric+'_low']-off[metric+'_high']
                r[f'delta_{metric}_high']=on[metric+'_high']-off[metric+'_low']
            slices.append(r)
        proof.append(dict(job=job,quartet_scores_sha256=sha(p),local_scalar_audit_sha256=sha(ap),all_pair_source_task_aggregates_rechecked=True))
    if slices:
        write('joint_reasoning_all_slices.csv',slices)
        (OUT/'joint_slice_arithmetic_audit.json').write_text(json.dumps(dict(status='verified_all_accepted_slices',models=proof,script_sha256=sha(Path(__file__)),scope='All source/task slices retained, pair first weighting, missing-output intervals are not sampling confidence intervals; R has no preferred direction.'),indent=2),encoding='utf-8')
    print('Checked all joint pair/source/task aggregates:',len(proof),'accepted models')


if __name__=='__main__':main()
