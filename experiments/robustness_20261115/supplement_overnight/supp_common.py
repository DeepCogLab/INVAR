"""Separate prospective freeze for explicitly post-hoc overnight follow-ups."""
import sys
from pathlib import Path
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S/'upstream'))
from campaign_common import MODELS,GPU_UUID,digest,sha,jl,load,save,messages_for

JOBS={
    'language_glm':dict(model='glm4_9b',experiment='instruction_language',task_types=['goal'],expected_rows=5280),
    'language_llama':dict(model='llama31_8b',experiment='instruction_language',task_types=['goal'],expected_rows=5280),
    'length_goals_qwen':dict(model='qwen3_8b',experiment='output_budget',task_types=['goal'],expected_rows=2640),
    'length_goals_gpt':dict(model='gpt_oss_20b',experiment='output_budget',task_types=['goal'],expected_rows=2640),
    'length_behavior_qwen':dict(model='qwen3_8b',experiment='output_budget',task_types=['behavior'],expected_rows=3024),
    'length_behavior_gpt':dict(model='gpt_oss_20b',experiment='output_budget',task_types=['behavior'],expected_rows=3024),
}


def cell(t):return '__'.join(str(t[k]) for k in ['language','instruction_language','arm','max_tokens'])
def input_key(t):return t['prompt_id']+'__'+cell(t)


def messages(p,t,job):
    # Change system-instruction language only; original user content stays fixed.
    copy=dict(p,language=t['instruction_language'])
    return messages_for(copy,MODELS[JOBS[job]['model']]['kind'],t['arm'])


def render(tokenizer,p,t,job):
    model=JOBS[job]['model'];kind=MODELS[model]['kind'];kwargs=dict(tokenize=False,add_generation_prompt=True)
    if kind=='native_toggle':kwargs['enable_thinking']=t['arm']=='native_on'
    if kind=='native_effort':kwargs['reasoning_effort']=t['arm'].removeprefix('effort_')
    ms=messages(p,t,job)
    return tokenizer.apply_chat_template(ms,**kwargs),digest(ms)


def make_plan(job,prompts):
    cfg=JOBS[job];rows=[]
    for p in prompts:
        if p['task_type'] not in cfg['task_types']:continue
        instructions=['zh','en'] if cfg['experiment']=='instruction_language' else [p['language']]
        arms=['direct','brief_reasoning'] if cfg['experiment']=='instruction_language' else (['native_on'] if cfg['model']=='qwen3_8b' else ['effort_high'])
        caps=[4096] if cfg['experiment']=='instruction_language' else [4096,8192]
        n=4 if p['task_type']=='behavior' else 2
        for language in instructions:
            for arm in arms:
                for cap in caps:
                    for rep in range(n):
                        t=dict(prompt_id=p['prompt_id'],language=p['language'],instruction_language=language,arm=arm,max_tokens=cap,replicate=rep)
                        # New cohort. Independent across settings, including caps;
                        # no assumed identical sampling path or paired token prefix.
                        t['sampling_seed']=int(digest([20261116,'overnight',job,t])[:8],16)%(2**31-1)
                        rows.append(t)
    return sorted(rows,key=lambda t:digest(['overnight_order',job,t]))


def verify_freeze():
    cfg=load(S/'freeze.json')
    for name,h in cfg['files'].items():assert sha(S/name)==h,name
    return cfg
