"""Complete accounting, deterministic case selection and existing audited full scorer."""
import argparse,collections,importlib.util,json,re,sys
from pathlib import Path
from run_model import R,read,save,digest,filehash,valid
def native_final(text,schema,finish):
    if finish=='length':return None
    tagged=re.findall(r'<Answer>\s*(.*?)\s*</Answer>',text,re.S|re.I)
    candidate=tagged[-1].strip() if len(tagged)==1 else text.strip()
    if candidate.startswith('```'):
        candidate=re.sub(r'^```(?:json)?\s*|\s*```$','',candidate).strip()
    return candidate if valid(candidate,schema,finish) else None
def analyze(job,stage):
    d=R/'runs'/job/stage;rows=read(d/'responses.jsonl');inputs={p['prompt_id']:p for p in read(R/'frozen/inputs.jsonl')}
    aggregate=collections.defaultdict(lambda:dict(n=0,valid=0,correct=0));per=collections.defaultdict(list)
    if stage=='native':
        # Author-requested wrappers are allowed only in this separate diagnostic.
        for r in rows:
            final=native_final(r['response_text'],inputs[r['prompt_id']]['output_schema'],r['finish_reason'])
            r['raw_response_text']=r['response_text'];r['strict_json_valid']=r['schema_valid']
            r['schema_valid']=final is not None
            if final is not None:r['response_text']=final
    for r in rows:
        p=inputs[r['prompt_id']];s=aggregate[p['task_type']];s['n']+=1;s['valid']+=r['schema_valid'];per[r['prompt_id']].append(r)
        if p['task_type']!='behavior' and r['schema_valid']:
            value=next(iter(json.loads(r['response_text']).values()))
            s['correct']+=int(value in p['expected_actions'] if p['task_type']=='goal' else value==p['expected_answer'])
    for typ,s in aggregate.items():
        s['valid_rate']=s['valid']/s['n']
        s['accuracy_all_attempts']=s['correct']/s['n'] if typ!='behavior' else None
        s['scope']='trial-weighted descriptive; official pair-weighted comparison requires full scorer'
    save(d/'task_summary.json',aggregate)
    candidates=collections.defaultdict(list);successes=collections.defaultdict(list)
    for pid,rs in per.items():
        p=inputs[pid]
        if p['task_type']=='behavior':continue
        bad=[]
        for r in rs:
            if not r['schema_valid']:bad.append(r);continue
            value=next(iter(json.loads(r['response_text']).values()))
            good=value in p['expected_actions'] if p['task_type']=='goal' else value==p['expected_answer']
            if not good:bad.append(r)
        if bad:
            candidates[p['task_type']].append(dict(prompt_id=pid,task_type=p['task_type'],pair_id=p.get('pair_id'),variant_id=p.get('variant_id'),
              attempts=len(rs),failures=len(bad),failure_rate=len(bad)/len(rs),messages=p['messages'],
              expected=p.get('expected_actions',p.get('expected_answer')),output_schema=p['output_schema'],
              examples=[dict(response_text=r.get('raw_response_text',r['response_text']),replicate=r['replicate'],sampling_seed=r['sampling_seed'],
                   finish_reason=r['finish_reason'],schema_valid=r['schema_valid']) for r in bad[:3]],
              failure_types=dict(format_or_truncation=sum(not r['schema_valid'] for r in bad),
                  objective_answer=sum(r['schema_valid'] for r in bad)),source_raw=str(d/'responses.jsonl')))
        else:
            successes[p['task_type']].append(dict(prompt_id=pid,pair_id=p.get('pair_id'),task_type=p['task_type'],
              messages=p['messages'],expected=p.get('expected_actions',p.get('expected_answer')),attempts=len(rs),success_rate=1.,
              raw_response_example=rs[0].get('raw_response_text',rs[0]['response_text'])))
    selected=[]
    for typ,cs in candidates.items():
        # Spread across pairs; descending repeated failure rate, stable hash tie breaker.
        cs.sort(key=lambda x:(-x['failure_rate'],digest(x['prompt_id'])))
        used=set()
        for c in cs:
            if c['pair_id'] in used:continue
            selected.append(c);used.add(c['pair_id'])
            if len(used)>=5:break
    save(d/'failure_cases.json',dict(stage=stage,selection='up to five different pairs per task, highest repeated failure rate, SHA tie break',
         candidate_prompts={k:len(v) for k,v in candidates.items()},total_attempts=len(rows),
         raw_sha256=filehash(d/'responses.jsonl'),cases=selected,
         interpretation='pilot/native exploratory; behavior variation is not an objective-answer error; no human review yet'))
    success_selection=[]
    for typ,cs in successes.items():
        used=set()
        for c in sorted(cs,key=lambda c:digest(c['prompt_id'])):
            if c['pair_id'] in used:continue
            used.add(c['pair_id']);success_selection.append(c)
            if len(used)>=3:break
    save(d/'success_cases.json',dict(selection='all-correct prompts, three distinct pairs per type, SHA ordering',cases=success_selection,
      scope='descriptive success examples; does not imply true success probability is one'))
    lines=[f'# {job} / {stage} 失败案例记录','',f'总调用：{len(rows)}。以下为按固定规则筛选的候选案例，尚未完成人工核验。',
       '自由行为没有标准答案；本文件只把有客观标签的事实、目标、角色题记作答题错误。原生输出的格式不合规单独标示。','']
    for c in selected:
        lines.extend([f'## {c["task_type"]} — {c["prompt_id"]}',f'条件对：{c["pair_id"]}；变体：{c["variant_id"]}；失败 {c["failures"]}/{c["attempts"]}。',
          f'正确答案：`{json.dumps(c["expected"],ensure_ascii=False)}`；错误分类：{c["failure_types"]}。','',c['messages'][-1]['content'],'',
          '原始输出：','```',c['examples'][0]['response_text'],'```',''])
    (d/'失败案例.md').write_text('\n'.join(lines),encoding='utf-8')
    if stage!='full':
        from report import main as report
        report();return
    complete=json.loads((d/'COMPLETE.json').read_text())
    if complete['invalid']:
        save(d/'SCORING_BLOCKED.json',dict(reason='Invalid/truncated outputs retained; never reroll or count as cognitive errors',invalid=complete['invalid']));return
    # Reuse the prior scorer unchanged in a new isolated directory and plan.
    b=Path(json.loads((R/'freeze.json').read_text())['baseline']);s=d/'scoring';s.mkdir(exist_ok=True)
    for n in ['inputs.jsonl','fact_quartets.jsonl','goal_quartets.jsonl']:
        p=s/n
        if not p.exists():p.symlink_to(R/'frozen'/n)
    if not (s/'INVAR_dataset').exists():(s/'INVAR_dataset').symlink_to(b/'INVAR_dataset',target_is_directory=True)
    op=s/'outputs/main';op.mkdir(parents=True,exist_ok=True)
    if not (op/f'{job}.jsonl').exists():(op/f'{job}.jsonl').symlink_to(d/'responses.jsonl')
    plan=json.loads((R/'frozen/plan.json').read_text())
    with (s/'main_plan.jsonl').open('w') as f:
        for p in plan:f.write(json.dumps(dict(**p,model=job,trial_id=f'{job}|{p["prompt_id"]}|{p["replicate"]}|{p["batch"]}'))+'\n')
    manifest=json.loads((R/'frozen/frozen_manifest.json').read_text());manifest['models']=[job];manifest['plan_sha256']=filehash(s/'main_plan.jsonl')
    save(s/'frozen_manifest.json',manifest)
    sys.path.insert(0,str(b/'INVAR_dataset'))
    spec=importlib.util.spec_from_file_location('prior_audited_scorer',b/'score_v2.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.R=s;sys.argv=['score_v2.py','--model',job];module.main()
    save(d/'scorer_provenance.json',dict(source=str(b/'score_v2.py'),sha256=filehash(b/'score_v2.py'),changes='none; isolated root and new model plan',
      seed_difference='20261024 for each model; previous scorer used model-index offsets; identical estimands and point estimates'))
    import csv
    with (s/'reports'/job/'behavior_quartets.csv').open(encoding='utf-8-sig') as f:qs=list(csv.DictReader(f))
    chosen=[];used=set()
    original={q['quartet_id']:q for q in read(b/'INVAR_dataset/data/quartets.jsonl')}
    for q in sorted(qs,key=lambda q:(-float(q['E']),digest(q['quartet_id']))):
        if q['pair_id'] in used:continue
        used.add(q['pair_id']);sides={}
        for side in 'ABCD':
            pid=original[q['quartet_id']][side];rs=[r for r in per[pid] if r['batch']=='main']
            sides[side]=dict(prompt_id=pid,messages=inputs[pid]['messages'],n=len(rs),
                 raw_response_frequencies=dict(collections.Counter(r['response_text'] for r in rs)))
        chosen.append(dict(metrics=q,sides=sides))
        if len(chosen)>=5:break
    save(d/'behavior_cases.json',dict(selection='largest E, five different pairs, stable SHA tie break',
         interpretation='distribution-level presentation instability; no objective social-choice label',cases=chosen))
    from report import main as report
    report()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--stage',required=True);a=ap.parse_args();analyze(a.job,a.stage)
