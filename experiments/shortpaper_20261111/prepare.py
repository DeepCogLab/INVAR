"""Freeze small supplementary experiment bundle. No old files are changed."""
import collections
import copy
import hashlib
import json
import re
from pathlib import Path
from analyze_existing import ROOT, BASE, jl, csvread

R = Path(__file__).resolve().parent
IDMAP = {'P1':'Q7','P2':'Q4','P3':'Q9','P4':'Q2','P5':'Q8','P6':'Q5'}


def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def write(p, obj):
    if isinstance(obj,list):
        text = ''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in obj)
    else:
        text = json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    if p.exists():
        assert p.read_text(encoding='utf-8') == text, 'Frozen file changed: '+str(p)
    else:
        p.write_text(text,encoding='utf-8')


def main():
    d = R/'frozen'
    d.mkdir(exist_ok=True)
    ps = jl(BASE/'inputs.jsonl')
    byid = {p['prompt_id']:p for p in ps}
    pairs = {p['pair_id']:p for p in jl(BASE/'INVAR_dataset/data/pairs.jsonl')}
    cases = {p['case_id']:p for p in jl(BASE/'INVAR_dataset/data/cases.jsonl')}
    case_by_goal = {}
    for q in jl(BASE/'goal_quartets.jsonl'):
        for t,s in zip('ABCD',[0,0,1,1]):
            c = cases[pairs[q['pair_id']][f'case{s}_id']]
            if q[t] in case_by_goal:
                old = case_by_goal[q[t]]
                assert old['payoffs'] == c['payoffs'] and old['entities'] == c['entities']
            case_by_goal[q[t]] = c
    model_inputs = []
    goal_inputs = []
    for p in ps:
        # Allowlist excludes answer labels and evaluation metadata by construction.
        model_inputs.append({k:p[k] for k in ['prompt_id','task_type','messages','output_schema']})
        if p['task_type'] != 'goal':
            continue
        c = case_by_goal[p['prompt_id']]
        participants = list(c['entities'])
        pays = c['payoffs'].get('by_action',c['payoffs'])
        heading = participants
        if p['variant_id'] == 'V5_ids':
            heading = [IDMAP[k] for k in heading]
        lines = ['【规则所确定的完整行动收益表】',
                 '以下仅展开原有规则中的收益，行动菜单和本次明确目标均保持原样。',
                 '| action | '+' | '.join(heading)+' |',
                 '|---|'+'---|'*len(heading)]
        for a in c['action_space']['values']:
            pay = dict(pays.get(a,pays.get(str(a))))
            if 'self' in pay:
                pay.setdefault(c['focal_id'],pay['self'])
            lines.append('| '+str(a)+' | '+' | '.join(str(pay[k]) for k in participants)+' |')
        original = next(x for x in model_inputs if x['prompt_id'] == p['prompt_id'])
        oracle = copy.deepcopy(original)
        text = oracle['messages'][-1]['content']
        assert text.count('【本次明确目标】') == 1
        oracle['messages'][-1]['content'] = text.replace('【本次明确目标】','\n'.join(lines)+'\n\n【本次明确目标】')
        goal_inputs.append(dict(prompt_id=p['prompt_id'], original=original['messages'],
            payoff_table=oracle['messages'], output_schema=p['output_schema']))
    repeats = [r['prompt_id'] for r in csvread(BASE/'reports/qwen3_8b/same_prompt_repeat.csv')]
    assert len(set(repeats)) == 127
    reps = {'behavior':256,'changed_fact':8,'stable_fact':8,'goal':16,'role':32}
    rows = sum(reps[p['task_type']] for p in ps)+256*len(repeats)
    assert rows == 421504 and len(goal_inputs) == 1410
    jobs = [dict(id='qwen3_14b',port=18240,model='Qwen3-14B',quantized=False),
            dict(id='qwen3_30b_a3b_gptq',port=18242,model='Qwen3-30B-A3B-GPTQ-Int4',quantized=True)]
    write(d/'model_inputs.jsonl',model_inputs)
    write(d/'goal_diagnostics.jsonl',goal_inputs)
    write(d/'repeat_ids.json',dict(prompt_ids=repeats))
    # Native reasoning sample is selected by fixed hash within tasks, before new results.
    buckets = collections.defaultdict(list)
    for p in ps:
        if p['task_type'] == 'goal':
            buckets[p['task']].append(p['prompt_id'])
    native_ids = sorted({pid for task, ids in buckets.items() for pid in sorted(ids,key=lambda x:digest([20261111,x]))[:32]})
    write(d/'native_sample.json',dict(prompt_ids=native_ids,selection='32 goals per applicable task by SHA256; no score selection',repeats=8,max_tokens=1408))
    write(R/'jobs.json',dict(jobs=jobs))
    freeze = dict(protocol='INVAR_shortpaper_supplement_v1_20261111',source_input_sha256=hashlib.sha256((BASE/'inputs.jsonl').read_bytes()).hexdigest(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())},
        main_rows_per_model=rows,goal_diagnostic_rows_per_model=1410*32*2,
        native_rows_per_model=len(native_ids)*8*2,
        main_repeats=reps,goal_repeats=32,temperature=.7,top_p=.95,top_k=-1,max_tokens=128,
        main_decoder='JSON schema; no correct labels supplied',thinking=False,
        chunk_size=32,seed_policy='sha256([20261111,prompt_id,batch,chunk]); same request seed across arms',
        primary_weighting='pair equal, then source equal and task equal sensitivity',
        selection='All 8666 inputs. All 1410 goal inputs in both original and payoff_table conditions.',
        diagnostics='Payoff table is privileged rule expansion, not an end-to-end deployable method or new data source.',
        baseline_comparison='New model configurations; old response seeds and API execution path differ. No parameter-only causal claim.',
        native_scope='Native no-thinking and thinking, unconstrained; separately report truncation/parse rates; 2048 server context; no truncating inputs.',
        prospective_scope='Arms, sample sizes and analyses frozen before new result inspection; benchmark itself was previously observed.')
    write(R/'freeze.json',freeze)
    print(json.dumps({k:v for k,v in freeze.items() if k!='files'},ensure_ascii=False))


if __name__ == '__main__':
    main()
