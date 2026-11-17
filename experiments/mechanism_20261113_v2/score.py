"""Predeclared same-menu proxy/decoder diagnostics, all attempted answers retained."""
import argparse
import collections
import math
import numpy as np
from common import *


def distribution_metrics(p,q,menu):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    assert abs(p.sum()-1)<1e-8 and abs(q.sum()-1)<1e-8
    m=(p+q)/2
    js=sum(float(x)*math.log(float(x/y)) for a in (p,q) for x,y in zip(a,m) if x>0)/2
    if all(type(x) in (int,float) for x in menu):
        order=np.argsort(menu);xs=np.asarray(menu,dtype=float)[order]
        w=float(np.sum(np.abs(np.cumsum((p-q)[order])[:-1])*np.diff(xs))/(xs[-1]-xs[0]))
    else:w=float(np.abs(p-q).sum()/2)
    return w,js


def decode(row,record):
    if record['finish_reason']=='length':return None,'truncated'
    try: obj=json.loads(record['response_text'])
    except (ValueError,TypeError):return None,'invalid_json'
    if type(obj)!=dict or set(obj)!={row['field']}:return None,'invalid_fields'
    value=obj[row['field']]
    for i,x in enumerate(row['candidates']):
        if type(value) is type(x) and value==x:return i,None
    return None,'illegal_type_or_menu'


def quantiles(values):
    finite=[x for x in values if x is not None and math.isfinite(x)]
    return dict(defined=len(finite),undefined=len(values)-len(finite),
        empirical_range=np.quantile(finite,[.025,.975]).tolist() if len(finite)==len(values) else None)


def main(job):
    verify_freeze();directory=out(job);assert (directory/'proxy/COMPLETE.json').exists() and (directory/'generation/COMPLETE.json').exists()
    target=directory/'analysis';assert not (target/'SCORING_COMPLETE.json').exists(),'Existing scores immutable'
    for stage in ('proxy','generation'):
        completion=load(directory/stage/'COMPLETE.json')
        assert completion['freeze_sha256']==sha(R/'freeze.json')
        for name,h in completion['files_sha256'].items(): assert sha(directory/stage/name)==h
    edge_rows=[];goal_rows=[];bridge_rows=[];histograms=[];contrasts=[];sensitivity=[]
    for source in SOURCES:
        rows={r['prompt_id']:r for r in jl(panel(source)/'model_inputs.jsonl')}
        edges=jl(panel(source)/'edges.jsonl');members=jl(panel(source)/'goal_membership.jsonl')
        labels={r['prompt_id']:r['expected'] for r in jl(panel(source)/'goal_labels.jsonl')}
        all_proxy={};all_counts={}
        for state in STATES:
            proxies={r['prompt_id']:r for r in jl(directory/'proxy'/f'{source}_{state}.jsonl')}
            assert set(proxies)==set(rows)
            records=jl(directory/'generation'/f'{source}_{state}.jsonl');seen=set()
            counts={pid:np.zeros(len(r['candidates'])+1,dtype=int) for pid,r in rows.items()}
            failures=collections.Counter()
            for record in records:
                pid=record['prompt_id'];row=rows[pid];rep=record['replicate'];key=(pid,rep)
                assert key not in seen and 0<=rep<reps(row);seen.add(key)
                assert record['source_panel']==source and record['state']==state and record['job']==job
                assert record['sampling_seed']==seed(job,source,state,pid,rep) and record['lora_binding_ok']
                meta=adapter(job,source,state)
                assert record['adapter_sha256']==(meta['fingerprint']['content_sha256'] if meta else None)
                assert record['messages_sha256']==digest(row['messages'])
                index,error=decode(row,record)
                counts[pid][index if index is not None else -1]+=1
                if error:failures[error]+=1
            assert len(seen)==sum(reps(r) for r in rows.values())
            for pid,row in rows.items():
                assert counts[pid].sum()==reps(row)
                proxy=proxies[pid];assert proxy['input_sha256']==digest(row) and proxy['candidates']==row['candidates']
                z=np.asarray(proxy['complete_logps']);z=np.exp(z-z.max());z/=z.sum()
                assert np.max(np.abs(z-proxy['probabilities']))<1e-10
                p=counts[pid][:-1]/counts[pid][:-1].sum() if counts[pid][:-1].sum() else None
                bridge_rows.append(dict(source=source,state=state,prompt_id=pid,task_type=row['task_type'],
                    proxy_decoder_tv=float(np.abs(z-p).sum()/2) if p is not None else None,
                    proxy_entropy=float(-sum(x*math.log(x) for x in z if x)),
                    decoder_entropy=float(-sum(x*math.log(x) for x in p if x)) if p is not None else None,
                    attempts=int(counts[pid].sum()),failures=int(counts[pid][-1])))
                histograms.append(dict(source=source,state=state,prompt_id=pid,counts=counts[pid].tolist()))
                if pid in labels:
                    idx=[i for i,x in enumerate(row['candidates']) if x in labels[pid]]
                    stratum=next(x['stratum'] for x in members if x['prompt_id']==pid)
                    goal_rows.append(dict(source=source,state=state,prompt_id=pid,stratum=stratum,
                        proxy_correct_set_mass=float(z[idx].sum()),Q=float(counts[pid][idx].sum()/counts[pid].sum()),
                        attempts=int(counts[pid].sum()),failures=int(counts[pid][-1])))
            for edge in edges:
                a,b=edge['left'],edge['right'];menu=rows[a]['candidates'];assert menu==rows[b]['candidates']
                w,j=distribution_metrics(proxies[a]['probabilities'],proxies[b]['probabilities'],menu)
                pa=counts[a][:-1];pb=counts[b][:-1]
                observed=distribution_metrics(pa/pa.sum(),pb/pb.sum(),menu) if pa.sum() and pb.sum() else (None,None)
                missing=sum(counts[p][-1]/counts[p].sum() for p in (a,b))
                edge_rows.append(dict(source=source,state=state,**edge,proxy_W1=w,proxy_JS=j,
                    decoder_W1_conditional=observed[0],decoder_JS_conditional=observed[1],
                    missing_quality_W1_bounds=[max(0,observed[0]-missing),min(1,observed[0]+missing)] if observed[0] is not None else [0,1]))
            all_proxy[state]=proxies;all_counts[state]=counts
        for step in (50,100,150):
            sft=f'anchor_sft_{step}';js=f'anchor_js_{step}'
            for stratum in ('train','heldout_source','heldout_variant'):
                es=[e for e in edges if e['stratum']==stratum]
                gs=[g for g in members if g['stratum']==stratum]
                used=sorted({e[k] for e in es for k in ('left','right')}|{g['prompt_id'] for g in gs})
                rng=np.random.default_rng(int(digest([20261113,job,source,step,stratum])[:16],16))
                draws={'delta_decoder_W1':[],'delta_Q':[]}
                for _ in range(500):
                    boot={st:{pid:rng.multinomial(int(all_counts[st][pid].sum()),all_counts[st][pid]/all_counts[st][pid].sum())
                        for pid in used} for st in (sft,js)}
                    values={}
                    for st in (sft,js):
                        ws=[]
                        for e in es:
                            a,b=e['left'],e['right'];pa=boot[st][a][:-1];pb=boot[st][b][:-1]
                            ws.append(distribution_metrics(pa/pa.sum(),pb/pb.sum(),rows[a]['candidates'])[0] if pa.sum() and pb.sum() else None)
                        qs=[sum(boot[st][g['prompt_id']][i] for i,x in enumerate(rows[g['prompt_id']]['candidates']) if x in labels[g['prompt_id']])/boot[st][g['prompt_id']].sum() for g in gs]
                        values[st]=(math.fsum(ws)/len(ws) if ws and all(w is not None for w in ws) else None,
                                    math.fsum(qs)/len(qs) if qs else None)
                    for i,key in enumerate(draws):
                        draws[key].append(values[js][i]-values[sft][i] if values[js][i] is not None and values[sft][i] is not None else None)
                subset=[e for e in edge_rows if e['source']==source and e['stratum']==stratum]
                d=dict(source=source,step=step,stratum=stratum,edge_count=len(es),goal_prompt_count=len(gs))
                for metric in ('proxy_W1','proxy_JS','decoder_W1_conditional','decoder_JS_conditional'):
                    means={st:[e[metric] for e in subset if e['state']==st] for st in (sft,js)}
                    d['delta_'+metric]=(math.fsum(means[js])/len(es)-math.fsum(means[sft])/len(es)
                        if es and all(x is not None for st in means for x in means[st]) else None)
                for key,values in draws.items(): d[key+'_shared_prompt_bootstrap']=quantiles(values)
                for metric in ('proxy_correct_set_mass','Q'):
                    means={st:[g[metric] for g in goal_rows if g['source']==source and g['stratum']==stratum and g['state']==st] for st in (sft,js)}
                    d['delta_'+metric]=(math.fsum(means[js])/len(gs)-math.fsum(means[sft])/len(gs)) if gs else None
                contrasts.append(d)
        for n in (16,32,64):
            for state in STATES:
                rng=np.random.default_rng(int(digest([20261113,'downsample',job,source,n,state])[:16],16))
                for stratum in ('train','heldout_source','heldout_variant'):
                    es=[e for e in edges if e['stratum']==stratum]; values=[]
                    used=sorted({e[k] for e in es for k in ('left','right')})
                    for _ in range(100):
                        sample={pid:rng.multivariate_hypergeometric(all_counts[state][pid],n) for pid in used}
                        ws=[]
                        for e in es:
                            a,b=e['left'],e['right'];pa=sample[a][:-1];pb=sample[b][:-1]
                            ws.append(distribution_metrics(pa/pa.sum(),pb/pb.sum(),rows[a]['candidates'])[0] if pa.sum() and pb.sum() else None)
                        values.append(math.fsum(ws)/len(ws) if ws and all(w is not None for w in ws) else None)
                    sensitivity.append(dict(source=source,state=state,stratum=stratum,n=n,**quantiles(values)))
    for name,records in [('edges',edge_rows),('goals',goal_rows),('bridge',bridge_rows),('histograms',histograms),
                         ('contrasts',contrasts),('sensitivity',sensitivity)]:write_jl(target/f'{name}.jsonl',records)
    save(target/'SCORING_COMPLETE.json',dict(status='scored_full_raw',job=job,freeze_sha256=sha(R/'freeze.json'),
        inputs_sha256={f'{stage}/{p.name}':sha(p) for stage in ('proxy','generation') for p in (directory/stage).glob('*.jsonl')},
        output_sha256={p.name:sha(p) for p in target.glob('*.jsonl')},
        limitations=['one preselected old seed; retrospective observed-source diagnostic',
        'small menus and fixed hash-selected edges; not the original full92 aggregate',
        'conditional decoder W1/JS; all-attempt Q; separate missing-quality bounds',
        'empirical bootstrap/subsample ranges are not calibrated confidence intervals',
        'complete-sequence proxy differs from token grammar temperature/top-p; no single-component attribution',
        'decoder JS/W1 are finite-sample plug-in estimates; preserve sample-size sensitivity']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();main(a.job)
