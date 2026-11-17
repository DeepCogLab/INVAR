"""Fixed-input JS-SFT uncertainty diagnostics from verified V4 histograms, no GPU."""
import collections,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=Path('artifacts/methods_v4_20261028/evaluations/transfer')
OUT=ROOT/'results/20261112_确认补实验/statistics_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(name,rs):
    if not rs:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)


class Distributions:
    def __init__(self,histograms,rng,draws,subsample=None):
        self.data={};self.edges={};self.draws=draws
        for pid,(xs,counts) in sorted(histograms.items()):
            n=int(sum(counts));p=counts/n
            sampled=rng.multinomial(n,p,size=draws) if subsample is None else rng.multivariate_hypergeometric(counts,subsample,size=draws)
            denominator=n if subsample is None else subsample
            self.data[pid]=(xs,np.cumsum(p),p@xs,np.cumsum(sampled/denominator,axis=1),(sampled/denominator)@xs)
    def mean(self,pid):
        x,c,m,b,bm=self.data[pid];return m,bm
    def distance(self,a,b):
        if a==b:return 0.,np.zeros(self.draws)
        key=tuple(sorted([a,b]))
        if key in self.edges:return self.edges[key]
        xa,ca,ma,ba,bma=self.data[a];xb,cb,mb,bb,bmb=self.data[b]
        grid=np.union1d(xa,xb);ia=np.searchsorted(xa,grid[:-1],side='right')-1;ib=np.searchsorted(xb,grid[:-1],side='right')-1
        ac=np.where(ia>=0,ca[np.maximum(ia,0)],0);bc=np.where(ib>=0,cb[np.maximum(ib,0)],0)
        ab=np.where(ia[None,:]>=0,ba[:,np.maximum(ia,0)],0);bbb=np.where(ib[None,:]>=0,bb[:,np.maximum(ib,0)],0)
        result=float(np.abs(ac-bc)@np.diff(grid)),np.abs(ab-bbb)@np.diff(grid)
        self.edges[key]=result;return result


def summarize(qs,dist,verify=True):
    pairs=collections.Counter(q['pair_id'] for q in qs);point={m:0. for m in ['E','G','R']};boot={m:np.zeros(dist.draws) for m in point}
    for q in qs:
        A,B,C,D=[q[k] for k in 'ABCD']
        ab,abb=dist.distance(A,B);cd,cdb=dist.distance(C,D);ac,acb=dist.distance(A,C);bd,bdb=dist.distance(B,D)
        ma,mab=dist.mean(A);mb,mbb=dist.mean(B);mc,mcb=dist.mean(C);md,mdb=dist.mean(D)
        p=dict(E=(ab+cd)/2,R=(ac+bd)/2,G=abs((md-mb)-(mc-ma)))
        b=dict(E=(abb+cdb)/2,R=(acb+bdb)/2,G=np.abs((mdb-mbb)-(mcb-mab)))
        if verify:
            for m in p:assert abs(p[m]-float(q[m]))<1e-9,(q['quartet_id'],m,p[m],q[m])
        weight=1/len(pairs)/pairs[q['pair_id']]
        for m in p:point[m]+=weight*p[m];boot[m]+=weight*b[m]
    return point,boot


def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'COMPLETE.json').exists(),'Preserve completed statistics'
    protocol=dict(draws=500,subsample_draws=100,subsample_n=[16,32,64,128],seed=20261112,alpha=.05,
        scope='Previously observed fixed inputs. Shared prompt histogram draws reused across quartets. Empirical resampling ranges are descriptive, not calibrated confidence intervals; degenerate distributions flagged. Q Hoeffding bounds assume independent bounded draws and combine shared-prompt coefficients. No source-population or training-seed-population inference.',script_sha256=sha(Path(__file__)))
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    inputs={p['prompt_id']:p for p in map(json.loads,(ROOT/'experiments/baseline_v2/inputs.jsonl').read_text(encoding='utf-8').splitlines())}
    sources={};results=[];curves=[];signs=[]
    files=sorted(DATA.glob('*/*/seed*/screen/scores/prompts.csv'));assert len(files)==24,len(files)
    for file_index,p in enumerate(files):
        screen=p.parents[1];marker=screen/'scoring_complete.json';assert json.loads(marker.read_text())['status']=='independently_verified'
        qp=p.with_name('quartets.csv');pr=rows(p);qr=rows(qp)
        for path in [p,qp,marker]:sources[str(path)]=sha(path)
        model,source,training_seed=p.relative_to(DATA).parts[:3]
        panels={};qstats={};histograms={};degenerate={}
        for method in ['anchor_sft','anchor_js']:
            hist={};qmeans={};ns={}
            for r in pr:
                if r['method']!=method:continue
                pid=r['prompt_id'];n=int(r['n']);distribution=json.loads(r['action_or_answer_distribution']);spec=inputs[pid]
                counts=np.array([round(float(v)*n) for v in distribution.values()],dtype=np.int64);assert counts.sum()==n
                keys=list(distribution)
                if r['task_type']=='behavior':
                    o=spec['outcome'];xs=np.array([float(k==o['target_action']) if o['kind']=='probability' else int(k)/o['denominator'] for k in keys])
                    grouped=collections.Counter()
                    for x,c in zip(xs,counts):grouped[float(x)]+=int(c)
                    xs=np.array(sorted(grouped));counts=np.array([grouped[x] for x in xs],dtype=np.int64)
                    assert abs(xs@counts/n-float(r['mean_normalized_or_correct']))<1e-9
                    hist[pid]=xs,counts
                elif r['task_type']=='goal':
                    truth={str(x) for x in spec['expected_actions']};qmeans[pid]=sum(c for k,c in zip(keys,counts) if k in truth)/n;ns[pid]=n
                    assert abs(qmeans[pid]-float(r['mean_normalized_or_correct']))<1e-9
            bqs=[q for q in qr if q['method']==method and q['kind']=='behavior']
            rng=np.random.default_rng(protocol['seed']+file_index*11+(method=='anchor_js'))
            histograms[method]=hist;degenerate[method]=sum(len(x)==1 for x,c in hist.values())
            panels[method]=summarize(bqs,Distributions(hist,rng,protocol['draws']))
            gqs=[q for q in qr if q['method']==method and q['kind']=='goal'];coef=collections.Counter()
            pair_counts=collections.Counter(q['pair_id'] for q in gqs)
            for q in gqs:
                w=1/len(pair_counts)/pair_counts[q['pair_id']]/4
                for side in 'ABCD':coef[q[side]]+=w
            qstats[method]=dict(mean=sum(c*qmeans[pid] for pid,c in coef.items()),variance_proxy=sum(c*c/ns[pid] for pid,c in coef.items()),applicable=bool(gqs))
            # Simultaneous fixed-prompt mean bands across 48 fitted model panels.
            M=len(hist);radii={pid:math.sqrt(math.log(2*M*48/protocol['alpha'])/(2*sum(c))) for pid,(x,c) in hist.items()}
            dist=Distributions(hist,rng,1)
            for q in bqs:
                intervals=[]
                for positive,negative in [('C','A'),('D','B')]:
                    coeff=collections.Counter();coeff[q[positive]]+=1;coeff[q[negative]]-=1
                    mu=sum(c*dist.mean(pid)[0] for pid,c in coeff.items());rad=sum(abs(c)*radii[pid] for pid,c in coeff.items())
                    intervals.append((mu-rad,mu+rad))
                certified=(intervals[0][1]<0<intervals[1][0]) or (intervals[1][1]<0<intervals[0][0])
                signs.append(dict(model=model,source=source,seed=training_seed,method=method,quartet_id=q['quartet_id'],simultaneous_fixed_prompt_reversal=bool(certified)))
        for metric in ['E','G','R']:
            point=panels['anchor_js'][0][metric]-panels['anchor_sft'][0][metric]
            draws=panels['anchor_js'][1][metric]-panels['anchor_sft'][1][metric]
            results.append(dict(model=model,source=source,seed=training_seed,metric=metric,JS_minus_SFT=point,
                empirical_resampling_low=float(np.quantile(draws,.025)),empirical_resampling_high=float(np.quantile(draws,.975)),
                Q_bound95_low=None,Q_bound95_high=None,SFT_degenerate_prompts=degenerate['anchor_sft'],JS_degenerate_prompts=degenerate['anchor_js']))
        if qstats['anchor_sft']['applicable']:
            point=qstats['anchor_js']['mean']-qstats['anchor_sft']['mean'];rad=math.sqrt(.5*math.log(2/protocol['alpha'])*(qstats['anchor_js']['variance_proxy']+qstats['anchor_sft']['variance_proxy']))
            results.append(dict(model=model,source=source,seed=training_seed,metric='Q',JS_minus_SFT=point,empirical_resampling_low=None,empirical_resampling_high=None,
                Q_bound95_low=max(-1.,point-rad),Q_bound95_high=min(1.,point+rad),SFT_degenerate_prompts=degenerate['anchor_sft'],JS_degenerate_prompts=degenerate['anchor_js']))
        for n in protocol['subsample_n']:
            sample_panels={}
            for method in ['anchor_sft','anchor_js']:
                rng=np.random.default_rng(protocol['seed']+file_index*101+n+(method=='anchor_js'))
                bqs=[q for q in qr if q['method']==method and q['kind']=='behavior']
                sample_panels[method]=summarize(bqs,Distributions(histograms[method],rng,protocol['subsample_draws'],n),verify=False)[1]
            for metric in ['E','G','R']:
                delta=sample_panels['anchor_js'][metric]-sample_panels['anchor_sft'][metric]
                curves.append(dict(model=model,source=source,seed=training_seed,n_per_prompt=n,metric=metric,delta_mean=float(delta.mean()),delta_low=float(np.quantile(delta,.025)),delta_high=float(np.quantile(delta,.975))))
        print('Verified histogram/resampling',file_index+1,'/24',model,source,training_seed,flush=True)
    save('paired_training_sampling.csv',results);save('training_sample_size_curves.csv',curves);save('signed_effect_simultaneous_checks.csv',signs)
    receipt=dict(status='completed_fixed_input_sampling_diagnostics',comparisons=24,input_sha256=sources,protocol=protocol,
        summary_rows=len(results),sample_size_rows=len(curves),signed_quartet_rows=len(signs),certified_reversals=sum(r['simultaneous_fixed_prompt_reversal'] for r in signs))
    (OUT/'COMPLETE.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('Fixed-input sampling diagnostics completed',len(results))


if __name__=='__main__':main()
