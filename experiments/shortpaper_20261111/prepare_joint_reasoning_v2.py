"""Freeze a small source/task-stratified behavior test for the reasoning alternative."""
import collections
import hashlib
import json
from pathlib import Path
from analyze_existing import BASE,jl

R=Path(__file__).resolve().parent


def main():
    d=R/'joint_frozen';d.mkdir(exist_ok=True)
    pairs=jl(BASE/'INVAR_dataset/data/pairs.jsonl');qs=jl(BASE/'INVAR_dataset/data/quartets.jsonl')
    groups=collections.defaultdict(list)
    for p in pairs:groups[p['source_id'],p['task']].append(p)
    selected=[]
    for key,rs in sorted(groups.items()):
        ordered=sorted(rs,key=lambda r:hashlib.sha256(('INVAR_joint_reasoning_v1_20261111|'+r['pair_id']).encode()).hexdigest())
        selected.extend(p['pair_id'] for p in ordered[:2])
    quartets=[q for q in qs if q['pair_id'] in selected and q['variant_id'] in ['V1_table','V4_rules_table']]
    ids={q[t] for q in quartets for t in 'ABCD'}
    ps=[p for p in jl(BASE/'inputs.jsonl') if p['prompt_id'] in ids]
    payloads=[{k:p[k] for k in ['prompt_id','task_type','messages','output_schema']} for p in ps]
    evaluation=[{k:p[k] for k in ['prompt_id','action_space','outcome']} for p in ps]
    for name,rows in [('model_inputs.jsonl',payloads),('evaluation.jsonl',evaluation),('quartets.jsonl',quartets)]:
        content=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)
        p=d/name
        if p.exists():assert p.read_text(encoding='utf-8')==content
        else:p.write_text(content,encoding='utf-8')
    assert len(selected)==15 and len(quartets)==30 and len(ids)==87
    cfg=dict(protocol='INVAR_joint_reasoning_behavior_v2_20261111',prelaunch_revision='v1 was never launched; empty/shared-reference metric test found a None comparison; scoring guard corrected without changing sampling or scientific settings',
        reason='Main runs are complete. Goal-only thinking gains do not establish simultaneous presentation stability; test the specific alternative on a fixed small behavior subset.',
        models={'qwen3_14b':dict(path='/mnt/models/models/Qwen3-14B',dtype='bfloat16'),
                'qwen3_30b_a3b_gptq':dict(path='/mnt/models/models/Qwen3-30B-A3B-GPTQ-Int4',dtype='float16',quantization='gptq_marlin')},
        source_task_cells=len(groups),pairs=15,quartets=30,unique_prompts=87,selected_pair_ids=selected,
        selection='At most two pairs per source/task cell, fixed SHA256 ordering; no score selection. Dictator has one pair. V1_table and V4_rules_table only.',
        repeats=128,arms=['native_no_thinking','native_thinking'],max_tokens=4096,max_model_len=8192,
        temperature=.7,top_p=.95,top_k=-1,rows_per_model=22272,total_rows=44544,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()},
        scoring='Report completion/truncation. E/G/R on parseable outputs are conditional; if any output is missing, bound all-attempt E/G/R using missing mass and normalized [0,1] outcomes. No silent dropping or invalid-as-zero actions.',
        inference='Fixed selected prompts, not a representative source-population claim; noise references descriptive; R has no preferred direction. No new social-behavior gold labels.',
        stop_rule='One fixed two-model/two-mode experiment; no expansion based on outcome.',
        runtime_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['offline_joint_reasoning_v2.py','score_joint_reasoning_v2.py','joint_reasoning_campaign_v2.py','audit_new_runs.py','run_api.py','reasoning_pair_campaign.py']})
    p=R/'joint_reasoning_freeze_v2.json';content=json.dumps(cfg,ensure_ascii=False,indent=2)
    if p.exists():assert p.read_text(encoding='utf-8')==content
    else:p.write_text(content,encoding='utf-8')
    print('Frozen',len(ids),'prompts;',cfg['total_rows'],'new behavior calls')


if __name__=='__main__':main()
