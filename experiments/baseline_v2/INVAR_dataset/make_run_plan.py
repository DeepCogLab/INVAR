"""Make deterministic randomized trial plans; this script does not send requests."""
import argparse,hashlib,json,random
from pathlib import Path
from evaluate import read_jsonl
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',required=True,help='Your actual model/version identifiers')
    ap.add_argument('--repeats',type=int,default=32)
    ap.add_argument('--tier',choices=['all','source_conditions','researcher_extension'],default='source_conditions')
    ap.add_argument('--smoke',action='store_true',help='One condition pair per source/task/role; 2 samples by default')
    ap.add_argument('--seed',type=int,default=20261022);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.repeats<1:ap.error('repeats must be positive')
    qs=read_jsonl(ROOT/'data'/'quartets.jsonl');selected=[];seen={}
    for q in qs:
        if a.tier!='all' and q['analysis_tier']!=a.tier:continue
        group=(q['source_id'],q['task'],q['role'])
        if a.smoke:
            seen.setdefault(group,q['pair_id'])
            if seen[group]!=q['pair_id']:continue
        selected.append(q)
    ids=sorted({q[k] for q in selected for k in 'ABCD'})
    rng=random.Random(a.seed);r=2 if a.smoke and a.repeats==32 else a.repeats
    # Select repeat controls within source/task/role strata, then deduplicate shared prompts.
    grouped={}
    for q in selected:
        group=(q['source_id'],q['task'],q['role']);grouped.setdefault(group,set()).update(q[k] for k in 'ABCD')
    repeat_ids=set()
    for group,pids in sorted(grouped.items()):
        pids=sorted(pids);repeat_ids.update(rng.sample(pids,max(1,round(len(pids)*.1))))
    plan=[]
    for model in a.models:
        for pid in ids:
            for rep in range(r):plan.append({'model':model,'prompt_id':pid,'replicate':rep,'batch':'main'})
        for pid in sorted(repeat_ids):
            for rep in range(r):plan.append({'model':model,'prompt_id':pid,'replicate':rep,'batch':'repeat'})
    rng.shuffle(plan)
    for i,row in enumerate(plan): row['trial_id']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20];row['order']=i
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in plan),encoding='utf-8')
    manifest={'model_ids':a.models,'tier':a.tier,'smoke':a.smoke,'unique_prompts':len(ids),'primary_repeats':r,'repeat_control_prompts':len(repeat_ids),'total_planned_calls':len(plan),'seed':a.seed,'dataset_prompt_file_sha256':hashlib.sha256((ROOT/'data'/'prompts.jsonl').read_bytes()).hexdigest(),'execution_status':'not_run','generation_config':'Must be filled by experimenter before API calls; use identical config within each model key.'}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
