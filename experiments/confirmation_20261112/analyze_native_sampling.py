"""Predeclared fixed-input diagnostics for independently accepted confirmation stages."""
import argparse,collections,csv,itertools,json,math,sys
from pathlib import Path
import numpy as np
from common import R,load,jl,save,sha

ROOT=R.parents[1]
OUT=ROOT/'results/20261112_确认补实验/native_statistics'
ARMS=['native_no_thinking','native_thinking']


def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def table(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def finite(x):return float(x) if np.isfinite(x) else None
def span(draws):
    bad=int(np.sum(~np.isfinite(draws)))
    return dict(empirical_low=finite(np.quantile(draws,.025)) if not bad else None,
                empirical_high=finite(np.quantile(draws,.975)) if not bad else None,undefined_draws=bad)


def weights(qs,kind):
    bypair=collections.defaultdict(list)
    for q in qs:bypair[q['pair_id']].append(q)
    bygroup=collections.defaultdict(list)
    key={'pair_equal':None,'source_equal':'source_id','task_equal':'task'}[kind]
    for pair,rs in bypair.items():bygroup[rs[0][key] if key else 'all'].append(pair)
    pairw={p:1/len(bygroup)/len(ps) for ps in bygroup.values() for p in ps}
    return [pairw[q['pair_id']]/len(bypair[q['pair_id']]) for q in qs]


class Panel:
    """Draw the legal outcomes and missing bin together; never impute action zero."""
    def __init__(self,arrays,rng,draws,n=None):
        self.data={};self.edges={};self.draws=draws
        for pid,arr in sorted(arrays.items()):
            valid=np.array([x for x in arr if x is not None],dtype=float)
            xs,counts=np.unique(valid,return_counts=True)
            allcounts=np.append(counts,len(arr)-len(valid)).astype(np.int64)
            sampled=rng.multinomial(len(arr),allcounts/len(arr),size=draws) if n is None else rng.multivariate_hypergeometric(allcounts,n,size=draws)
            sampled=sampled[:,:-1];validn=sampled.sum(axis=1)
            with np.errstate(divide='ignore',invalid='ignore'):
                cdf=np.cumsum(sampled,axis=1)/validn[:,None]
                mean=(sampled@xs)/validn
            self.data[pid]=(xs,counts/len(valid) if len(valid) else counts.astype(float),
                float(valid.mean()) if len(valid) else np.nan,cdf,mean)
    def means(self,pid):return self.data[pid][2],self.data[pid][4]
    def distance(self,a,b):
        if a==b:return 0.,np.zeros(self.draws)
        edge=tuple(sorted([a,b]))
        if edge in self.edges:return self.edges[edge]
        xa,pa,ma,ba,bma=self.data[a];xb,pb,mb,bb,bmb=self.data[b]
        if not len(xa) or not len(xb):return np.nan,np.full(self.draws,np.nan)
        grid=np.union1d(xa,xb);ia=np.searchsorted(xa,grid[:-1],side='right')-1;ib=np.searchsorted(xb,grid[:-1],side='right')-1
        ca=np.where(ia>=0,np.cumsum(pa)[np.maximum(ia,0)],0.);cb=np.where(ib>=0,np.cumsum(pb)[np.maximum(ib,0)],0.)
        ac=np.where(ia[None,:]>=0,ba[:,np.maximum(ia,0)],0.);bc=np.where(ib[None,:]>=0,bb[:,np.maximum(ib,0)],0.)
        point=float(np.abs(ca-cb)@np.diff(grid));draws=np.abs(ac-bc)@np.diff(grid)
        # Even a one-point grid is undefined if a resample has no legal output.
        draws[~np.isfinite(bma)|~np.isfinite(bmb)]=np.nan
        self.edges[edge]=point,draws;return point,draws


def behavior(qs,panel,kind,verify=False):
    result={m:0. for m in ['E','G','R']};draws={m:np.zeros(panel.draws) for m in result}
    for q,w in zip(qs,weights(qs,kind)):
        A,B,C,D=[q[k] for k in 'ABCD']
        ab,abb=panel.distance(A,B);cd,cdb=panel.distance(C,D);ac,acb=panel.distance(A,C);bd,bdb=panel.distance(B,D)
        ma,mab=panel.means(A);mb,mbb=panel.means(B);mc,mcb=panel.means(C);md,mdb=panel.means(D)
        coeff=collections.Counter()
        for side,c in [('A',1),('B',-1),('C',-1),('D',1)]:coeff[q[side]]+=c
        signed=sum(c*panel.means(pid)[0] for pid,c in coeff.items() if c)
        signed_draws=sum((c*panel.means(pid)[1] for pid,c in coeff.items() if c),np.zeros(panel.draws))
        p=dict(E=(ab+cd)/2,R=(ac+bd)/2,G=abs(signed))
        b=dict(E=(abb+cdb)/2,R=(acb+bdb)/2,G=np.abs(signed_draws))
        if verify:
            for m in p:
                expected=float(q[m+'_conditional']) if q[m+'_conditional'] else np.nan
                assert np.isclose(p[m],expected,atol=1e-9,equal_nan=True),(q['quartet_id'],m)
        for m in p:result[m]+=w*p[m];draws[m]+=w*b[m]
    return result,draws


def goal(qs,rates,kind):
    coeff=collections.Counter()
    for q,w in zip(qs,weights(qs,kind)):
        for side in 'ABCD':coeff[q[side]]+=w/4
    mass=math.fsum(coeff.values());assert math.isclose(mass,1.,abs_tol=1e-12)
    coeff={p:c/mass for p,c in coeff.items()}
    point=math.fsum(c*float(np.mean(rates[p])) for p,c in coeff.items())
    assert -1e-12<=point<=1+1e-12
    return max(0.,min(1.,point)),coeff


def permutation_reference(qs,arrays,rng,kind):
    edges={};total=0.
    def edge(a,b):
        if a==b:return 0.
        key=tuple(sorted([a,b]))
        if key in edges:return edges[key]
        va=[x for x in arrays[a] if x is not None];vb=[x for x in arrays[b] if x is not None]
        if not va or not vb:return np.nan
        xs,ct=np.unique(va+vb,return_counts=True);na,nb=len(va),len(vb)
        sample=rng.multivariate_hypergeometric(ct,na,size=199)
        diff=np.cumsum(sample/na-(ct[None,:]-sample)/nb,axis=1)[:,:-1]
        edges[key]=float(np.mean(np.abs(diff)@np.diff(xs)));return edges[key]
    for q,w in zip(qs,weights(qs,kind)):total+=w*(edge(q['A'],q['B'])+edge(q['C'],q['D']))/2
    return finite(total)


def analyze(job,stage,mirror,out,draws=500,subdraws=100):
    d=mirror/'runs'/job/stage;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'Preserve prior statistics'
    index=load(mirror/'mirror_index.json');local=load(mirror/'local_scalar_audits'/f'{job}_{stage}.json')
    assert local['status']=='passed_independent_scalar' and local['raw_sha256']==index['files'][f'runs/{job}/{stage}/responses.jsonl']['sha256']
    paths=[d/'attempt_scores.csv',d/'goal_quartet_scores.csv',d/'SCORING_COMPLETE.json',mirror/'frozen'/stage/'evaluation.jsonl']
    if stage=='canonical':paths.append(d/'behavior_quartet_scores.csv')
    for p in paths:assert sha(p)==index['files'][str(p.relative_to(mirror)).replace('\\','/')]['sha256']
    audit=load(d/'SCORING_COMPLETE.json');labels={p['prompt_id']:p for p in jl(mirror/'frozen'/stage/'evaluation.jsonl')}
    attempts=read(d/'attempt_scores.csv');gq=read(d/'goal_quartet_scores.csv')
    bq=read(d/'behavior_quartet_scores.csv') if stage=='canonical' else []
    behavior_arrays={arm:collections.defaultdict(list) for arm in ARMS};goal_arrays={arm:collections.defaultdict(list) for arm in ARMS}
    failures=collections.Counter();catalog=[];entropy=[]
    for row in attempts:
        arm,pid=row['arm'],row['prompt_id'];valid=row['valid']=='True'
        if row['task_type']=='behavior':behavior_arrays[arm][pid].append(float(row['normalized_outcome']) if valid else None)
        else:goal_arrays[arm][pid].append(int(row['correct']=='True'))
        if not valid:failures[arm,pid,row['error']]+=1
    for (arm,pid,error),n in sorted(failures.items()):catalog.append(dict(arm=arm,prompt_id=pid,task=labels[pid]['task'],error=error,count=n))
    results=[];curves=[];refs=[];goal_slices=[];discrimination=[]
    qpanels={};rng=np.random.default_rng(20261112+stage.count('a')*19+list(load(mirror/'freeze.json')['models']).index(job)*1000)
    for arm in ARMS:
        for pid,arr in behavior_arrays[arm].items():
            assert len(arr)==128
            ct=collections.Counter(x for x in arr if x is not None);n=sum(ct.values())
            h=-sum((c/n)*math.log(c/n) for c in ct.values()) if n else None
            entropy.append(dict(arm=arm,prompt_id=pid,n_attempts=len(arr),n_legal=n,observed_outcome_values=len(ct),entropy_nats=h,single_observed_outcome=len(ct)==1))
        for arr in goal_arrays[arm].values():assert len(arr)==8
        qpanels[arm]={p:rng.binomial(len(v),np.mean(v),size=draws)/len(v) for p,v in sorted(goal_arrays[arm].items())}
    if bq:
        panels={arm:Panel(behavior_arrays[arm],rng,draws) for arm in ARMS}
        for kind in ['pair_equal','source_equal','task_equal']:
            computed={arm:behavior([q for q in bq if q['arm']==arm],panels[arm],kind,True) for arm in ARMS}
            for metric in ['E','G','R']:
                off,on=[computed[arm][0][metric] for arm in ARMS];delta=computed[ARMS[1]][1][metric]-computed[ARMS[0]][1][metric]
                recorded=[s['behavior'][kind][metric+'_conditional'] for s in audit['summaries']]
                assert np.allclose([off,on],[np.nan if x is None else x for x in recorded],atol=1e-9,equal_nan=True)
                results.append(dict(kind='behavior',weighting=kind,metric=metric,off_point=finite(off),on_point=finite(on),delta=finite(on-off),
                    **span(delta),Q_pointwise_bound_low=None,Q_pointwise_bound_high=None))
            for arm in ARMS:refs.append(dict(arm=arm,weighting=kind,conditional_E_permutation_mean=permutation_reference([q for q in bq if q['arm']==arm],behavior_arrays[arm],rng,kind),resamples=199))
        for n in [16,32,64,128]:
            computed={arm:behavior([q for q in bq if q['arm']==arm],Panel(behavior_arrays[arm],rng,subdraws,n),'pair_equal') for arm in ARMS}
            for metric in ['E','G','R']:
                delta=computed[ARMS[1]][1][metric]-computed[ARMS[0]][1][metric]
                curves.append(dict(n_all_attempts_per_prompt=n,metric=metric,delta_mean=finite(np.mean(delta)),**span(delta)))
    for kind in ['pair_equal','source_equal','task_equal']:
        calculated={arm:goal([q for q in gq if q['arm']==arm],goal_arrays[arm],kind) for arm in ARMS}
        off,on=[calculated[arm][0] for arm in ARMS]
        assert np.allclose([off,on],[s['goal'][kind]['Q'] for s in audit['summaries']],atol=1e-10)
        samples={arm:sum(c*qpanels[arm][p] for p,c in calculated[arm][1].items()) for arm in ARMS}
        variance=sum(c*c/len(goal_arrays[arm][p]) for arm in ARMS for p,c in calculated[arm][1].items())
        rad=math.sqrt(.5*math.log(40)*variance)
        results.append(dict(kind='goal',weighting=kind,metric='Q',off_point=off,on_point=on,delta=on-off,**span(samples[ARMS[1]]-samples[ARMS[0]]),
            Q_pointwise_bound_low=max(-1.,on-off-rad),Q_pointwise_bound_high=min(1.,on-off+rad)))
    for arm in ARMS:
        qq=[q for q in gq if q['arm']==arm]
        for q in qq:
            sa,sb=map(lambda p:set(labels[p]['expected_actions']),[q['A'],q['C']])
            q['transition']='same_optimum_set' if sa==sb else 'disjoint_optimum_sets' if not sa&sb else 'changed_but_overlapping'
        for dimension in ['transition','policy','task','source_id']:
            for value in sorted({q[dimension] for q in qq}):
                sub=[q for q in qq if q[dimension]==value];point,_=goal(sub,goal_arrays[arm],'pair_equal')
                goal_slices.append(dict(arm=arm,dimension=dimension,value=value,n_quartet_references=len(sub),n_pairs=len({q['pair_id'] for q in sub}),Q=point))
        buckets=collections.defaultdict(dict)
        for q in qq:
            for side in 'ABCD':buckets[q['pair_id'],q.get('variant_id','list_table'),side][q['policy']]=q[side]
        seen=set()
        for (pair,variant,side),policies in sorted(buckets.items()):
            for a,b in itertools.combinations(sorted(policies),2):
                pa,pb=policies[a],policies[b];key=tuple(sorted([pa,pb]))
                if key in seen or set(labels[pa]['expected_actions'])&set(labels[pb]['expected_actions']):continue
                seen.add(key);qa,qb=np.mean(goal_arrays[arm][pa]),np.mean(goal_arrays[arm][pb])
                discrimination.append(dict(arm=arm,pair_id=pair,policy1=a,policy2=b,prompt1=pa,prompt2=pb,Q1=qa,Q2=qb,worst_policy=min(qa,qb),both_correct_independent_estimate=qa*qb))
    for name,rs in [('sampling_summary.csv',results),('sample_size_curves.csv',curves),('conditional_noise_reference.csv',refs),('prompt_entropy_and_missing.csv',entropy),('failure_catalog.csv',catalog),('goal_transition_policy_source_task.csv',goal_slices),('goal_discrimination.csv',discrimination)]:table(out/name,rs)
    result=dict(status='completed_native_sampling_diagnostics',job=job,stage=stage,source_raw_sha256=local['raw_sha256'],
        input_sha256={str(p.relative_to(mirror)):sha(p) for p in paths},draws=draws,subsample_draws=subdraws,missing_attempts=len([r for r in attempts if r['valid']!='True']),
        scope='Fixed inputs only. Bootstrap resamples all attempts including missing; E/G/R are conditional within draws. Undefined draws make ranges unavailable. Empirical ranges are not calibrated confidence intervals; zero-width ranges do not prove invariance. Missing-quality bounds remain in primary audit. Q pointwise Hoeffding bounds assume independent bounded draws and combine shared prompt coefficients, not multiple-comparison/source-population guarantees. R has no preferred direction. Goal products are independent-call estimates, not trajectories.',script_sha256=sha(Path(__file__)))
    save(out/'COMPLETE.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--stage',required=True,choices=['canonical','parameter']);ap.add_argument('--synthetic-test',action='store_true');a=ap.parse_args()
    protocol=dict(draws=500,subsample_draws=100,subsample_n=[16,32,64,128],conditional_permutations=199,
        script_sha256=sha(Path(__file__)),seed_base=20261112,goal_weights=['pair_equal','source_equal','task_equal'],
        scope='Predeclared before reading new confirmation results. Fixed observed canonical inputs and separate parameter diagnostic. No new source or calibrated bootstrap coverage claims.')
    p=OUT/'protocol.json';OUT.mkdir(exist_ok=True,parents=True)
    if p.exists():assert load(p)==protocol
    else:save(p,protocol)
    mirror=Path('artifacts/confirmation_20261112')
    if a.synthetic_test:
        raise SystemExit('Use validate_native_sampling.py to build a separate synthetic verification fixture; no scientific outputs are fabricated.')
    receipt=load(R/'local_mirror_acceptance.json')
    assert any(x['job']==a.job and x['stage']==a.stage for x in receipt['accepted']),'Stage not independently accepted locally'
    print(json.dumps(analyze(a.job,a.stage,mirror,OUT/a.job/a.stage)))


if __name__=='__main__':main()
