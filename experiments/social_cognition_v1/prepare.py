"""Freeze inputs, matched base inventory and full evaluation schedule."""
import hashlib,json,os,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
B=Path('/workspace/project_social_simulation/invar_reliability_runs/20261024_v2')
M=Path('/workspace/models')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    d=R/'frozen';d.mkdir(exist_ok=True)
    for name in ['inputs.jsonl','fact_quartets.jsonl','goal_quartets.jsonl','frozen_manifest.json']:
        dest=d/name
        if dest.exists():assert sha(dest)==sha(B/name)
        else:shutil.copy2(B/name,dest)
    assert sha(d/'inputs.jsonl')==json.loads((d/'frozen_manifest.json').read_text())['input_sha256']
    rows=[json.loads(s) for s in (d/'inputs.jsonl').open()]
    # Use exactly the prior trial multiplicities, including the repeat batch.
    plan=[]
    for s in (B/'main_plan.jsonl').open():
        p=json.loads(s)
        if p['model']=='qwen3_8b':plan.append({k:p[k] for k in ['prompt_id','replicate','batch']})
    save(d/'plan.json',plan)
    groups=[('smoltaur',.36,'SmolLM2-360M','Smoltaur-0.4B-LoRA-r16','plain'),
      ('qwentaur06',.6,'Qwen3-0.6B-Base','Qwentaur-0.6B-LoRA-r16','plain'),
      ('olmotaur',1.,'OLMo-2-0425-1B','Olmotaur-1B-LoRA-r16','plain'),
      ('qwentaur17',1.7,'Qwen3-1.7B-Base','Qwentaur-1.7B-LoRA-r16','plain'),
      ('qwentaur4',4.,'Qwen3-4B-Base','Qwentaur-4B-LoRA-r16','plain'),
      ('socialr1_4',4.,'SocialR1-4B',None,'chat'),
      ('sotopia',7.,'Mistral-7B-Instruct-v0.1','SOTOPIA-pi-Mistral-7B-BC_SR','mistral'),
      ('minitaur',8.,'Meta-Llama-3.1-8B-bnb-4bit','Llama-3.1-Minitaur-8B-adapter','plain'),
      ('qwentaur8',8.,'Qwen3-8B-Base','Qwentaur-8B-LoRA-r16','plain'),
      ('socialr1_8',8.,'SocialR1-8B',None,'chat'),
      ('socio',8.,'Socio-Foundation-8B',None,'chat'),
      ('centaur',70.,'Meta-Llama-3.1-70B-bnb-4bit','Llama-3.1-Centaur-70B-adapter','plain')]
    jobs=[];inventory=[]
    for id,size,base,adapter,render in groups:
        config=json.loads((M/adapter/'adapter_config.json').read_text()) if adapter else None
        inventory.append(dict(id=id,size=size,base=base,adapter=adapter,adapter_config=config,base_present=(M/base/'config.json').exists()))
        # Adapter and its exact parent are evaluated under identical prompt rendering.
        arms=[(id,base,adapter)] if adapter else [(id,base,None)]
        if adapter: arms.append((id+'_base',base,None))
        if id=='socialr1_4':arms.append(('qwen3_4b_control','Qwen3-4B',None))
        if id=='socialr1_8':arms.append(('qwen3_8b_control','Qwen3-8B',None))
        for arm,b,a in arms:
            jobs.append(dict(id=arm,size=size,base=b,adapter=a,render=render,gpu_count=1,
              download_repo='unsloth/'+b if b.endswith('-Base') else None))
    save(R/'jobs.json',jobs);save(R/'inventory.json',inventory)
    save(R/'freeze.json',dict(input_sha256=sha(d/'inputs.jsonl'),plan_sha256=sha(d/'plan.json'),rows_per_full_arm=len(plan),
         prompts=len(rows),protocol='INVAR_social_cognition_v1_20261104',baseline=str(B),
         correctness_labels_given_to_model=False))
    print(json.dumps(dict(jobs=len(jobs),full_rows_per_arm=len(plan),prompts=len(rows))))
if __name__=='__main__':main()
