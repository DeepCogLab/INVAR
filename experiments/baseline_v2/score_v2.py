"""Fixed-item V2 analysis with shared-prompt Monte Carlo resampling."""
import argparse,collections,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'INVAR_dataset'))
from evaluate import parse_response,normalize

def read(path):
    with path.open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def save(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def filehash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class Distributions:
    def __init__(self,values,seed=20261024,draws=1000):
        self.values=values;self.draws=draws;self.rng=np.random.default_rng(seed)
        self.data={};self.edges={}
        for pid,arr in sorted(values.items()):
            support,counts=np.unique(arr,return_counts=True);n=len(arr);probs=counts/n
            boots=self.rng.multinomial(n,probs,size=draws)/n
            self.data[pid]=(support,probs,np.cumsum(boots,axis=1),boots@support)
    def mean(self,pid):
        x,p,_,m=self.data[pid];return float(p@x),m
    def distance(self,a,b):
        key=tuple(sorted([a,b]))
        if key in self.edges:return self.edges[key]
        xa,pa,ba,_=self.data[a];xb,pb,bb,_=self.data[b]
        grid=np.union1d(xa,xb)
        if len(grid)<2:
            result=(0.,np.zeros(self.draws),0.)
        else:
            ia=np.searchsorted(xa,grid[:-1],side='right')-1
            ib=np.searchsorted(xb,grid[:-1],side='right')-1
            acdf=np.where(ia>=0,np.cumsum(pa)[np.maximum(ia,0)],0.)
            bcdf=np.where(ib>=0,np.cumsum(pb)[np.maximum(ib,0)],0.)
            aboot=np.where(ia[None,:]>=0,ba[:,np.maximum(ia,0)],0.)
            bboot=np.where(ib[None,:]>=0,bb[:,np.maximum(ib,0)],0.)
            widths=np.diff(grid)
            point=float(np.abs(acdf-bcdf)@widths);bootstrap=np.abs(aboot-bboot)@widths
            # Within-edge pooled permutation reference. It is not a p-value.
            pooled=np.concatenate([self.values[a],self.values[b]])
            xp,ct=np.unique(pooled,return_counts=True)
            na,nb=len(self.values[a]),len(self.values[b])
            sample=self.rng.multivariate_hypergeometric(ct,na,size=199)
            pcdf=np.cumsum(sample/na-(ct[None,:]-sample)/nb,axis=1)[:,:-1]
            reference=float(np.mean(np.abs(pcdf)@np.diff(xp)))
            result=(point,bootstrap,reference)
        self.edges[key]=result;return result

def pair_weights(rows):
    counts=collections.Counter(x['pair_id'] for x in rows)
    return np.array([1/len(counts)/counts[x['pair_id']] for x in rows])

def summarized(rows,boot,metrics,group_name='all',group_value='all'):
    if not rows:return []
    weights=pair_weights(rows)
    record=dict(group=group_name,value=group_value,n_pairs=len(set(x['pair_id'] for x in rows)),n_quartets=len(rows))
    for metric in metrics:
        record[metric]=float(weights@np.array([x[metric] for x in rows]))
        draws=weights@np.stack([boot[x['quartet_id']][metric] for x in rows])
        low,high=np.quantile(draws,[.025,.975])
        record[metric+'_mc_low']=float(low);record[metric+'_mc_high']=float(high)
    if 'accuracy' in metrics:
        coefficients=collections.defaultdict(float);ns={}
        for weight,row in zip(weights,rows):
            for pid,n in boot[row['quartet_id']]['_accuracy_sources']:
                coefficients[pid]+=float(weight)/4;ns[pid]=n
        # Weighted independent bounded trials; shared prompt references combine
        # before computing their coefficients. Non-degenerate at all-correct.
        radius=float(np.sqrt(.5*np.log(40)*sum(c*c/ns[pid] for pid,c in coefficients.items())))
        record['accuracy_bound95_low']=max(0.,record['accuracy']-radius)
        record['accuracy_bound95_high']=min(1.,record['accuracy']+radius)
    return [record]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args()
    manifest=json.loads((R/'frozen_manifest.json').read_text())
    assert filehash(R/'inputs.jsonl')==manifest['input_sha256']
    if 'plan_sha256' in manifest:assert filehash(R/'main_plan.jsonl')==manifest['plan_sha256']
    inputs={p['prompt_id']:p for p in read(R/'inputs.jsonl')}
    expected={}
    with (R/'main_plan.jsonl').open() as f:
        for line in f:
            p=json.loads(line)
            if p['model']==a.model:
                assert p['trial_id'] not in expected
                expected[p['trial_id']]=(p['prompt_id'],p['replicate'],p['batch'])
    values=collections.defaultdict(list);repeat=collections.defaultdict(list);correct=collections.defaultdict(list)
    seen=set();configs=set();counts=collections.Counter();errors=[]
    raw=R/'outputs/main'/f'{a.model}.jsonl'
    with raw.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);trial=row['trial_id'];pid=row['prompt_id'];p=inputs[pid]
            assert row['model']==a.model and trial not in seen
            assert expected[trial]==(pid,row['replicate'],row['batch'])
            seen.add(trial);configs.add(row['config_sha256']);counts[row['batch']]+=1
            obj=json.loads(row['response_text']);field=next(iter(p['output_schema']['properties']));value=obj[field]
            spec=p['output_schema']['properties'][field]
            valid=set(obj)=={field} and type(value) is (int if spec['type']=='integer' else str)
            valid=valid and ('enum' not in spec or value in spec['enum'])
            valid=valid and ('minimum' not in spec or spec['minimum']<=value<=spec['maximum'])
            if not valid or row['status']!='ok' or row['finish_reason']=='length':errors.append(trial)
            if p['task_type']=='behavior':
                action,error=parse_response(row['response_text'],p['action_space']);assert error is None
                (repeat if row['batch']=='repeat' else values)[pid].append(normalize(action,p['outcome']))
            else:
                correct[pid].append(int(value in p['expected_actions'] if p['task_type']=='goal' else value==p['expected_answer']))
    assert seen==set(expected) and len(configs)==1 and not errors,(len(seen),len(expected),errors[:3])
    identity_path=raw.with_suffix('.identity.json')
    if identity_path.exists():
        identity=json.loads(identity_path.read_text())
        assert configs=={identity['config_sha256']}
        assert identity['config']['inputs_sha256']==manifest['input_sha256']
    for pid,arr in values.items():assert len(arr)==256,pid
    out=R/'reports'/a.model;out.mkdir(parents=True,exist_ok=True)
    seed=20261024+manifest['models'].index(a.model)
    distributions=Distributions(values,seed=seed)
    behavior=[];boots={}
    qs=read(R/'INVAR_dataset/data/quartets.jsonl')
    for q in qs:
        A,B,C,D=[q[t] for t in 'ABCD']
        ab,abb,abn=distributions.distance(A,B);cd,cdb,cdn=distributions.distance(C,D)
        ac,acb,acn=distributions.distance(A,C);bd,bdb,bdn=distributions.distance(B,D)
        ma,mab=distributions.mean(A);mb,mbb=distributions.mean(B);mc,mcb=distributions.mean(C);md,mdb=distributions.mean(D)
        delta0=mc-ma;delta1=md-mb
        row={k:v for k,v in q.items() if k not in 'ABCD'}
        row.update(E=(ab+cd)/2,R=(ac+bd)/2,delta_original=delta0,delta_variant=delta1,G=abs(delta1-delta0),
                   E_permutation_reference=(abn+cdn)/2,R_permutation_reference=(acn+bdn)/2,
                   mean_A=ma,mean_B=mb,mean_C=mc,mean_D=md,n_per_side=256)
        behavior.append(row)
        boots[q['quartet_id']]=dict(E=(abb+cdb)/2,R=(acb+bdb)/2,
                                    delta_original=mcb-mab,delta_variant=mdb-mbb,G=np.abs((mdb-mbb)-(mcb-mab)))
    metrics=('E','R','G','delta_original','delta_variant')
    summaries=summarized(behavior,boots,metrics)
    for key in ('task','role','variant_id','intervention_kind','analysis_tier','source_id'):
        for v in sorted(set(x[key] for x in behavior)):
            summaries+=summarized([x for x in behavior if x[key]==v],boots,metrics,key,v)
    for source in sorted(set(x['source_id'] for x in behavior)):
        summaries+=summarized([x for x in behavior if x['source_id']!=source],boots,metrics,'leave_source_out',source)
    summaries+=summarized([x for x in behavior if x['variant_id']!='V6_relation'],boots,metrics,'sensitivity','without_V6')
    save(out/'behavior_quartets.csv',behavior);save(out/'behavior_summary.csv',summaries)
    # Fixed six-task macro average is supplementary; source population is not sampled.
    task_rows=[x for x in summaries if x['group']=='task']
    macro={metric:float(np.mean([x[metric] for x in task_rows])) for metric in metrics}
    (out/'task_macro.json').write_text(json.dumps(macro,indent=2))
    repeat_rows=[]
    for pid,r in repeat.items():
        assert len(r)==256
        # Bootstrap objects contain main responses only; repetition is a distinct batch.
        x,c=np.unique(np.concatenate([values[pid],r]),return_counts=True)
        ca=np.searchsorted(np.sort(values[pid]),x[:-1],side='right')/256
        cb=np.searchsorted(np.sort(r),x[:-1],side='right')/256
        repeat_rows.append(dict(prompt_id=pid,n_main=256,n_repeat=256,W1=float(np.abs(ca-cb)@np.diff(x))))
    save(out/'same_prompt_repeat.csv',repeat_rows)
    rng=np.random.default_rng(seed+1000);acc={};accboot={}
    for pid,arr in sorted(correct.items()):
        acc[pid]=float(np.mean(arr));accboot[pid]=rng.binomial(len(arr),acc[pid],size=1000)/len(arr)
    controls=[];control_boot={}
    for file,typ in [('fact_quartets.jsonl','fact'),('goal_quartets.jsonl','goal')]:
        for q in read(R/file):
            p=[acc[q[t]] for t in 'ABCD'];bp=[accboot[q[t]] for t in 'ABCD']
            row={k:v for k,v in q.items() if k not in 'ABCD'}
            row.update(type=typ,accuracy=float(np.mean(p)),worst_side=min(p),dual_original=p[0]*p[2],dual_variant=p[1]*p[3],
                       min_original=min(p[0],p[2]),min_variant=min(p[1],p[3]),accuracy_A=p[0],accuracy_B=p[1],accuracy_C=p[2],accuracy_D=p[3])
            controls.append(row)
            control_boot[q['quartet_id']]=dict(accuracy=np.mean(bp,axis=0),worst_side=np.min(bp,axis=0),
                                               dual_original=bp[0]*bp[2],dual_variant=bp[1]*bp[3],
                                               min_original=np.minimum(bp[0],bp[2]),min_variant=np.minimum(bp[1],bp[3]),
                                               _accuracy_sources=[(q[tag],len(correct[q[tag]])) for tag in 'ABCD'])
    cm=('accuracy','worst_side','dual_original','dual_variant','min_original','min_variant')
    cs=[]
    for kind in ('changed_fact','stable_fact','goal'):
        subset=[x for x in controls if x.get('probe_type',x['type'])==kind]
        cs+=summarized(subset,control_boot,cm,'type',kind)
        for key in ('task','variant_id','factor','policy'):
            for v in sorted(set(x[key] for x in subset if key in x)):
                cs+=summarized([x for x in subset if x.get(key)==v],control_boot,cm,kind+'_'+key,v)
    for policy in ('own','total','equality'):
        subset=[x for x in controls if x.get('policy')==policy and x.get('unique_optimum_switch')]
        cs+=summarized(subset,control_boot,cm,'goal_switch',policy)
    # Heterogeneous fact and goal rows need a consistent CSV schema.
    allkeys=list(dict.fromkeys(k for x in controls for k in x))
    save(out/'control_quartets.csv',[{k:x.get(k,'') for k in allkeys} for x in controls])
    save(out/'control_summary.csv',cs)
    role_rows=[dict(prompt_id=pid,variant_id=inputs[pid]['variant_id'],correct=sum(arr),n=len(arr),accuracy=acc[pid]) for pid,arr in correct.items() if inputs[pid]['task_type']=='role']
    save(out/'role_checks.csv',role_rows)
    if role_rows:
        role_accuracy=float(np.mean([x['accuracy'] for x in role_rows]))
        radius=float(np.sqrt(.5*np.log(40)*sum((1/len(role_rows))**2/x['n'] for x in role_rows)))
        (out/'role_summary.json').write_text(json.dumps(dict(accuracy=role_accuracy,accuracy_bound95_low=max(0,role_accuracy-radius),
            accuracy_bound95_high=min(1,role_accuracy+radius),n_prompts=len(role_rows),method='Hoeffding bound; fixed prompts; independent sampled outputs'),indent=2))
    (out/'completion_audit.json').write_text(json.dumps(dict(model=a.model,status='passed',response_rows=len(seen),expected=len(expected),
        raw_sha256=filehash(raw),configuration_sha256=list(configs),counts=dict(counts),interface_failures=0,truncated=0,
        input_sha256=manifest['input_sha256'],ci_scope='fixed items, shared unique-prompt response bootstrap, not source-population inference',
        bootstrap_draws=1000,bootstrap_caveat='Empirical resampling ranges only; non-smooth E/R/G boundaries and degenerate control samples preclude zero-effect testing or perfect-accuracy proofs',
        accuracy_bound_method='Two-sided pointwise Hoeffding 95% bound; shared prompt coefficients merged; fixed prompts, not population inference',
        dual_caveat='Product estimates independent calls both correct, not an observed paired transition',
        permutation_reference_draws=199,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2),encoding='utf-8')
    print(json.dumps(dict(model=a.model,status='scored',rows=len(seen)),ensure_ascii=False),flush=True)

if __name__=='__main__':
    from runtime_guard import guarded_scoring
    guarded_scoring(R,main)
