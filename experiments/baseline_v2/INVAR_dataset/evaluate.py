"""Score saved responses; never calls model APIs. Invalid outputs remain in denominators."""
from __future__ import annotations
import argparse, csv, json, math, random, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]

def parse_response(text,space):
    try:
        if not isinstance(text,str): return None,'response_not_text'
        obj=json.loads(text)
        if not isinstance(obj,dict) or set(obj)!={'action'}: return None,'invalid_schema'
        a=obj['action']
        if isinstance(a,bool): return None,'invalid_action'
        if 'values' in space:
            # JSON 1 and "1" are not interchangeable.
            if not any(type(a) is type(v) and a==v for v in space['values']): return None,'invalid_action'
        elif not isinstance(a,int) or not space['min']<=a<=space['max']:
            return None,'invalid_action'
        return a,None
    except (ValueError,TypeError): return None,'invalid_json'

def normalize(action,outcome):
    if outcome['kind']=='probability': return float(action==outcome['target_action'])
    return action/outcome['denominator']

def w1(a,b):
    """Exact empirical W1 for possibly unequal sample sizes on the real line."""
    if not a or not b: return None
    ca,cb=Counter(a),Counter(b); xs=sorted(set(ca)|set(cb))
    acc_a=acc_b=area=0.0
    for i,x in enumerate(xs[:-1]):
        acc_a+=ca[x]/len(a); acc_b+=cb[x]/len(b)
        area+=abs(acc_a-acc_b)*(xs[i+1]-x)
    return area

def quartet_metrics(a,b,c,d):
    if not all((a,b,c,d)): return None
    delta0=statistics.mean(c)-statistics.mean(a)
    delta1=statistics.mean(d)-statistics.mean(b)
    return {'E':(w1(a,b)+w1(c,d))/2,'delta_original':delta0,'delta_variant':delta1,'G':abs(delta1-delta0)}

def null_distance(a,b,rng,resamples):
    pool=list(a)+list(b); ds=[]
    for _ in range(resamples):
        rng.shuffle(pool); ds.append(w1(pool[:len(a)],pool[len(a):]))
    return statistics.mean(ds) if ds else None

def save_csv(path,rows):
    if not rows: path.write_text('',encoding='utf-8-sig'); return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader();w.writerows(rows)

def evaluate(data_dir,responses,outdir,resamples=199,seed=20261022,tier='all'):
    prompts={p['prompt_id']:p for p in read_jsonl(data_dir/'prompts.jsonl')}
    qs=read_jsonl(data_dir/'quartets.jsonl')
    if tier!='all':qs=[q for q in qs if q['analysis_tier']==tier]
    selected_ids={q[k] for q in qs for k in 'ABCD'}
    rows=read_jsonl(responses); outputs=defaultdict(list); counts=defaultdict(Counter);seen=set();models=set()
    metadata=defaultdict(set); global_errors=[]
    for n,row in enumerate(rows,1):
        try:
            key=(row['model'],row['prompt_id'],row['replicate'],row.get('batch','main'))
            if key in seen: raise ValueError(f'duplicate model/prompt/replicate/batch at row {n}: {key}')
            seen.add(key)
            if row['prompt_id'] not in prompts: raise ValueError(f'unknown prompt_id at row {n}')
            models.add(row['model'])
            metadata[row['model']].add(json.dumps(row.get('generation_config',{}),sort_keys=True))
            prompt=prompts[row['prompt_id']]; short=(row['model'],row['prompt_id'],row.get('batch','main'))
            counts[short]['attempted']+=1
            if row.get('status','ok')!='ok':
                counts[short]['service_or_refusal_status']+=1;continue
            action,error=parse_response(row.get('response_text'),prompt['action_space'])
            if error: counts[short][error]+=1;continue
            outputs[short].append(normalize(action,prompt['outcome']));counts[short]['valid']+=1
        except KeyError as exc:
            raise ValueError(f'missing required field row {n}: {exc}') from exc
    outdir.mkdir(parents=True,exist_ok=True)
    if any(len(v)>1 for v in metadata.values()):
        raise ValueError('Different generation_config values share a model key. Separate these runs before scoring.')
    rng=random.Random(seed);details=[];validity=[];repeats=[]
    for model in sorted(models):
        for pid in sorted(selected_ids):
            count=counts[(model,pid,'main')]
            validity.append({'model':model,'prompt_id':pid,**dict(count),'attempted':count['attempted'],'valid':count['valid'],'valid_rate':count['valid']/count['attempted'] if count['attempted'] else None})
            r=outputs[(model,pid,'repeat')];a=outputs[(model,pid,'main')]
            if a and r: repeats.append({'model':model,'prompt_id':pid,'main_n':len(a),'repeat_n':len(r),'same_prompt_W1':w1(a,r)})
        for q in qs:
            arrays=[outputs[(model,q[k],'main')] for k in 'ABCD']
            result={k:q[k] for k in ('quartet_id','pair_id','source_id','task','role','analysis_tier','variant_id','variant_group','intervention_kind')}
            result['model']=model
            for tag,arr in zip('ABCD',arrays): result['n_'+tag]=len(arr)
            metrics=quartet_metrics(*arrays)
            if metrics is None:
                result['status']='missing_valid_condition';details.append(result);continue
            rates=[counts[(model,q[k],'main')]['valid']/counts[(model,q[k],'main')]['attempted'] for k in 'ABCD']
            result.update(metrics)
            result['minimum_valid_rate']=min(rates)
            result['status']='validity_below_95pct' if min(rates)<.95 else 'estimable'
            result['min_n']=min(map(len,arrays))
            result['low_sample_warning']=min(map(len,arrays))<32
            result['null_E_mean']=(null_distance(arrays[0],arrays[1],rng,resamples)+null_distance(arrays[2],arrays[3],rng,resamples))/2 if resamples else None
            details.append(result)
    # Aggregate within homogeneous task/role/intervention strata, first per pair.
    groups=defaultdict(list)
    for d in details:
        if d['status']!='estimable':continue
        key=tuple(d[k] for k in ('model','source_id','task','role','analysis_tier','intervention_kind','variant_id'))
        groups[key].append(d)
    summary=[]
    for key,items in sorted(groups.items()):
        by_pair=defaultdict(list)
        for d in items: by_pair[d['pair_id']].append(d)
        rec=dict(zip(('model','source_id','task','role','analysis_tier','intervention_kind','variant_id'),key))
        rec['n_pairs']=len(by_pair);rec['n_quartets']=len(items)
        for metric in ('E','delta_original','delta_variant','G'):
            vals=[statistics.mean(d[metric] for d in ds) for ds in by_pair.values()]
            rec[metric+'_mean']=statistics.mean(vals)
        rec['inference_scope']='descriptive_fixed_items; no independent-source population CI'
        summary.append(rec)
    save_csv(outdir/'quartet_metrics.csv',details);save_csv(outdir/'stratified_summary.csv',summary)
    save_csv(outdir/'output_validity.csv',validity);save_csv(outdir/'same_prompt_repeats.csv',repeats)
    report={'models':sorted(models),'response_rows':len(rows),'tier':tier,'unique_prompts_in_dataset':len(prompts),'selected_prompt_count':len(selected_ids),'quartet_statuses':dict(Counter(d['status'] for d in details)),'configurations_per_model':{k:len(v) for k,v in metadata.items()},'warnings':['No human-correctness score is computed.','Null E mean is a finite-sampling reference, not an adjusted score or p-value.','Low-N estimates are descriptive only.','Repeated measurements and source protocols are dependent.','Missing response records are not assumed to be model refusals. Inspect run plan completeness.','If configurations_per_model > 1, split configurations into distinct model keys and rerun.'],'resamples':resamples,'seed':seed}
    (outdir/'evaluation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('responses',type=Path);ap.add_argument('--data',type=Path,default=ROOT/'data');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resamples',type=int,default=199);ap.add_argument('--seed',type=int,default=20261022)
    ap.add_argument('--tier',choices=['all','source_conditions','researcher_extension'],default='source_conditions')
    args=ap.parse_args()
    if args.resamples<0:ap.error('--resamples must be nonnegative')
    print(json.dumps(evaluate(args.data,args.responses,args.out,args.resamples,args.seed,args.tier),ensure_ascii=False,indent=2))
