"""Build auditable, provider-neutral INVAR prompts. Python standard library only."""
from __future__ import annotations
import argparse, copy, hashlib, importlib, json, re, shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SYSTEM = '请根据题目给定的角色、信息和规则，作出一次决策。只输出题目要求的JSON对象。'
VERSION = '0.2.0'
VARIANTS = {
    'V1_paragraph': ('V1', '个体设定：列表改自然段', 'presentation'),
    'V1_table': ('V1', '个体设定：列表改表格', 'presentation'),
    'V2_rules_first': ('V2', '实验规则提前', 'presentation'),
    'V2_situation_first': ('V2', '当前情况提前', 'presentation'),
    'V3_situation_list': ('V3', '情境自然段改事实清单', 'presentation'),
    'V4_rules_table': ('V4', '规则清单改表格', 'presentation'),
    'V5_ids': ('V5', '无身份含义的参与者编号替换', 'relational'),
    'V6_relation': ('V6', '同一社会关系的对应表述', 'relational'),
}
IDMAP = {'P1':'Q7', 'P2':'Q4', 'P3':'Q9', 'P4':'Q2', 'P5':'Q8', 'P6':'Q5'}

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')

def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def renumber(s):
    return re.sub(r'P\d+',lambda m: IDMAP.get(m[0],m[0]),s)

def bullets(items):
    return '\n'.join('- '+x for x in items)

def role_sentence(fact):
    # Explicit reversible surface rules, not a generative biography summarizer.
    replacements={
        '参与者编号：':'你的参与者编号是',
        '本次角色：填写':'你本次填写',
        '本次角色：':'你本次是',
        '本情境资源：':'你在本情境中拥有',
    }
    for prefix,value in replacements.items():
        if fact.startswith(prefix):return value+fact[len(prefix):]
    return fact

def table(items):
    cells=[x.replace('|','\\|') for x in items]
    return '| 条目 | 内容 |\n|---|---|\n'+'\n'.join(f'| {i+1} | {x} |' for i,x in enumerate(cells))

def output_instruction(case):
    a=case['action_space']; labels=case.get('action_labels',{})
    if 'values' in a:
        choices='；'.join(f'{json.dumps(v,ensure_ascii=False)}'+('（'+labels[str(v)]+'）' if str(v) in labels else '') for v in a['values'])
        desc='action必须是以下合法值之一：'+choices+'。'
    else:
        desc=f'action必须是{a["min"]}至{a["max"]}之间的整数（含端点）。'
    return case['question']+'\n只输出一个JSON对象，且仅包含action字段。'+desc

def render(case, variant='base'):
    facts={k:list(case[k]) for k in ('role_facts','rules','situation')}
    if variant=='V6_relation':
        alt=case['relation_alt']
        assert alt['section'] in facts
        assert 0<=alt['index']<len(facts[alt['section']])
        facts[alt['section']][alt['index']]=alt['text']
    blocks={
        'role_facts':'【个体设定】\n'+(table(facts['role_facts']) if variant=='V1_table' else ''.join(role_sentence(x) for x in facts['role_facts']) if variant=='V1_paragraph' else bullets(facts['role_facts'])),
        'rules':'【实验规则】\n'+(table(facts['rules']) if variant=='V4_rules_table' else bullets(facts['rules'])),
        'situation':'【当前情况】\n'+(bullets(facts['situation']) if variant=='V3_situation_list' else ''.join(facts['situation'])),
    }
    order=['role_facts','rules','situation']
    if variant=='V2_rules_first': order=['rules','role_facts','situation']
    if variant=='V2_situation_first': order=['situation','role_facts','rules']
    text='\n\n'.join(blocks[k] for k in order)+'\n\n【本次决策】\n'+output_instruction(case)
    if variant=='V5_ids': text=renumber(text)
    return [{'role':'system','content':SYSTEM},{'role':'user','content':text}]

def validate_case(c):
    for key in ('role_facts','rules','situation'):
        assert isinstance(c[key],list) and c[key] and all(isinstance(t,str) and t.strip() for t in c[key]), key
    assert c['question'] and c['focal_id'] in c['entities']
    a=c['action_space']; o=c['outcome']
    if 'values' in a:
        assert len(a['values'])>=2 and len(set(a['values']))==len(a['values'])
    else: assert isinstance(a['min'],int) and isinstance(a['max'],int) and a['max']>a['min']
    if o['kind']=='probability': assert o['target_action'] in a['values']
    else: assert o['denominator']>0
    assert c['payoffs'], 'Payoffs/formula required'

def applicable(c0,c1,v):
    if v=='V6_relation': return all(c.get('relation_alt') for c in (c0,c1))
    if v=='V5_ids': return all(re.search(r'P\d+',json.dumps(c,ensure_ascii=False)) for c in (c0,c1))
    if v=='V3_situation_list': return all(len(c['situation'])>=2 for c in (c0,c1))
    return True

def source_manifest():
    specs=[
        ('PIQ572','README.pdf','CC0-1.0','https://doi.org/10.7910/DVN/PIQ572'),
        ('PIQ572','metadata.json','repository metadata','https://doi.org/10.7910/DVN/PIQ572'),
        ('PIQ572','Tab1_sumstat_games.tex','CC0-1.0','https://doi.org/10.7910/DVN/PIQ572'),
        ('IGVOFO','WEP Robust Econographics Survey Document FINAL.pdf','CC0-1.0','https://doi.org/10.7910/DVN/IGVOFO'),
        ('IGVOFO','metadata.json','repository metadata','https://doi.org/10.7910/DVN/IGVOFO'),
        ('PGG','instructions_appendix.pdf','CC-BY-4.0; credit-line exceptions retained','https://doi.org/10.1007/s10683-022-09756-9'),
        ('HPVBG','article_fulltext.xml','CC-BY (article); peer review portions retain their notices','https://doi.org/10.12688/wellcomeopenres.15474.2'),
        ('HPVBG','registration_0.json','repository metadata','https://doi.org/10.17605/OSF.IO/HPVBG'),
    ]
    rows=[]
    for source,name,license_name,url in specs:
        src=WORKSPACE/'research'/'evidence'/source/name
        assert src.exists(),src
        # Public methodological material only. Never copy individual participant tables.
        dst=ROOT/'provenance'/'source_materials'/source/name
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
        rows.append({'source_id':source,'file':str(dst.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'url':url,'license_note':license_name,'checked_date':'2026-10-22','use':'rules_and_provenance','participant_rows_included':False})
    return rows

def main(from_canonical=False):
    pairs=[];candidate_exclusions=[]
    if from_canonical:
        def read(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        cs={c['case_id']:c for c in read(ROOT/'data'/'cases.jsonl')}
        for p in read(ROOT/'data'/'pairs.jsonl'):
            for side in (0,1):
                c=cs[p.pop(f'case{side}_id')].copy()
                for field in ('source_id','pair_id','condition'):c.pop(field,None)
                p[f'case{side}']=c
            pairs.append(p)
        candidate_exclusions=read(ROOT/'audit'/'candidate_exclusions.jsonl')
        sources=read(ROOT/'provenance'/'sources.jsonl')
        for row in sources:
            assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256']
    else:
        for name in ('piq','pgg_tpp','econ'):
            module=importlib.import_module('builders.'+name)
            pairs.extend(module.build_pairs(WORKSPACE))
            candidate_exclusions.extend(getattr(module,'EXCLUSIONS',[]))
        sources=source_manifest()
    pairs.sort(key=lambda p:p['pair_id'])
    assert len({p['pair_id'] for p in pairs})==len(pairs)
    for p in pairs:
        p['task']={'trust_game':'trust','ultimatum_game':'ultimatum'}.get(p['task'],p['task'])
        if p['task']=='trust':
            p['role']={'trustor':'sender','trustee':'receiver'}.get(p['role'],p['role'])
    prompts={}; cases=[]; quartets=[]; certs=[]; exclusions=[]
    def add_prompt(c,p,v):
        messages=render(c,v); pid='p_'+digest(messages)[:20]
        row={'prompt_id':pid,'messages':messages,'action_space':c['action_space'],'outcome':c['outcome'],'content_sha256':digest(messages),'language':'zh-CN','schema_version':VERSION,'uses':[]}
        if pid in prompts:
            assert prompts[pid]['action_space']==row['action_space'] and prompts[pid]['outcome']==row['outcome']
        else: prompts[pid]=row
        prompts[pid]['uses'].append({'pair_id':p['pair_id'],'variant':v,'case_id':c['case_id']})
        return pid
    for p in pairs:
        for idx in (0,1):
            c=p[f'case{idx}']; validate_case(c)
            c['case_id']=p['pair_id']+f'_c{idx}'
            cases.append({**c,'source_id':p['source_id'],'pair_id':p['pair_id'],'condition':idx})
        a,c=p['case0'],p['case1']
        assert a['action_space']==c['action_space'],p['pair_id']
        assert a['outcome']==c['outcome'],p['pair_id']
        a_id=add_prompt(a,p,'base'); c_id=add_prompt(c,p,'base')
        p['analysis_group']=p.get('split_group',p.get('group_id',p['source_id']))
        p['human_direction_scoring_enabled']=False
        p['review_status']='program_checked_and_ai_reviewed_human_review_pending'
        for v,(family,label,group) in VARIANTS.items():
            if not applicable(a,c,v):
                exclusions.append({'pair_id':p['pair_id'],'variant':v,'reason':'not_applicable_or_no_reviewable_alternative'})
                continue
            b_id=add_prompt(a,p,v); d_id=add_prompt(c,p,v)
            assert a_id!=b_id and c_id!=d_id, (p['pair_id'],v,'noop')
            qid=p['pair_id']+'__'+v
            quartets.append({'quartet_id':qid,'pair_id':p['pair_id'],'source_id':p['source_id'],'task':p['task'],'role':p['role'],'analysis_tier':p['analysis_tier'],'intervention_kind':p['intervention']['kind'],'variant_id':v,'variant_family':family,'variant_group':group,'A':a_id,'B':b_id,'C':c_id,'D':d_id})
            # Deterministic rendering checks do not prove pragmatic equivalence.
            if v=='V5_ids':
                for case in (a,c):
                    reverted=render(case,v)[1]['content']
                    for original,new in IDMAP.items(): reverted=reverted.replace(new,original)
                    assert reverted==render(case)[1]['content']
            if v!='V6_relation':
                for case in (a,c):
                    body=render(case,v)[1]['content']
                    for key in ('role_facts','rules','situation'):
                        for fact in case[key]:
                            expected=renumber(fact) if v=='V5_ids' else fact
                            if v=='V1_paragraph' and key=='role_facts':expected=role_sentence(fact)
                            assert expected.replace('|','\\|') in body or expected in body,(qid,key,fact)
            certs.append({'quartet_id':qid,'operation':label,'invariant_checks':['same_case_records','same_action_space','same_outcome_scale','same_system_instruction','same_question_and_output_contract'],'fact_preservation':'exact_strings_and_bijection_checked' if v!='V6_relation' else 'authored_relation_paraphrase_requires_semantic_review','entity_map':IDMAP if v=='V5_ids' else {},'answer_inverse_map':'identity; action values unchanged','human_review':'pending','ai_review':'source specialist + integration audit; not independent human review'})
    prompt_rows=sorted(prompts.values(),key=lambda x:x['prompt_id'])
    for row in prompt_rows:
        row['character_count']=sum(len(m['content']) for m in row['messages'])
    pair_rows=[{k:v for k,v in p.items() if k not in ('case0','case1')}|{'case0_id':p['case0']['case_id'],'case1_id':p['case1']['case_id']} for p in pairs]
    for name,rows in [('cases',cases),('pairs',pair_rows),('quartets',quartets),('prompts',prompt_rows)]: jsonl(ROOT/'data'/f'{name}.jsonl',rows)
    jsonl(ROOT/'audit'/'equivalence_checks.jsonl',certs)
    jsonl(ROOT/'audit'/'variant_exclusions.jsonl',exclusions)
    jsonl(ROOT/'audit'/'candidate_exclusions.jsonl',candidate_exclusions)
    jsonl(ROOT/'provenance'/'sources.jsonl',sources)
    dump(ROOT/'audit'/'build_summary.json',{'version':VERSION,'status':'ready_for_model_pilot_not_human_validated','pairs':len(pairs),'cases':len(cases),'quartets':len(quartets),'expanded_references':4*len(quartets),'unique_prompts':len(prompt_rows),'pairs_by_tier':dict(Counter(p['analysis_tier'] for p in pairs)),'pairs_by_task':dict(Counter(p['task'] for p in pairs)),'quartets_by_variant':dict(Counter(q['variant_id'] for q in quartets)),'human_records_released':0,'direction_labels_enabled':0,'fact_preservation_checks':'passed','pragmatic_equivalence':'AI reviewed; independent human review pending'})
    # Human-readable complete inventory, not a cherry-picked showcase.
    cards=['# INVAR 全部题目浏览\n\n基准题按条件对列出；变体原文见 prompts.jsonl。人类回答不在题目内。\n']
    for p in pairs:
        cards.append(f'## {p["pair_id"]}\n\n来源：{p["source_id"]}；分层：{p["analysis_tier"]}\n')
        for tag in ('case0','case1'):
            cards.append(f'### {tag}\n\n'+render(p[tag])[1]['content']+'\n')
    (ROOT/'samples').mkdir(exist_ok=True)
    (ROOT/'samples'/'全部基准题.md').write_text('\n'.join(cards),encoding='utf-8')
    sample=[]
    seen=set()
    for q in quartets:
        key=(q['source_id'],q['task'],q['role'])
        if key in seen: continue
        seen.add(key)
        sample.append({**q,'full_prompts':{k:prompts[q[k]]['messages'] for k in ('A','B','C','D')}})
    jsonl(ROOT/'samples'/'完整四元组示例.jsonl',sample)
    controls=[];control_groups=set()
    for p in sorted(pairs,key=lambda p:(p['analysis_tier']!='source_conditions',p['pair_id'])):
        group=(p['source_id'],p['task'],p['role'])
        if group in control_groups:continue
        control_groups.add(group)
        c=p['case0']
        for v in ('base','V1_paragraph','V5_ids'):
            messages=render(c,v)
            body=messages[1]['content'].split('【本次决策】')[0]
            focal=renumber(c['focal_id']) if v=='V5_ids' else c['focal_id']
            body+='【事实理解检查】\n请回答：你在本题的参与者编号是什么？不作行为选择。只输出JSON，例如{"answer":"编号"}，将编号换成题目中的实际编号。'
            msgs=[{'role':'system','content':'请阅读给定材料并回答事实问题。只输出指定的JSON对象。'},{'role':'user','content':body}]
            controls.append({'prompt_id':'control_'+digest(msgs)[:20],'messages':msgs,'expected_answer':focal,'control_type':'focal_role_binding','source_id':p['source_id'],'pair_id':p['pair_id'],'variant_id':v,'use':'independent_calls_only_never_prepend_to_main_decision'})
    jsonl(ROOT/'controls'/'role_comprehension.jsonl',controls)
    # A source-only launch set is kept separate; no source tier is inferred from a filename.
    for tier in ('source_conditions','researcher_extension'):
        tier_qs=[q for q in quartets if q['analysis_tier']==tier]
        used={q[k] for q in tier_qs for k in 'ABCD'}
        jsonl(ROOT/'data'/'subsets'/f'{tier}_prompt_ids.jsonl',[{'prompt_id':pid} for pid in sorted(used)])
    dump(ROOT/'audit'/'release_checksums.json',{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').rglob('*.jsonl'))})
    print(json.dumps(json.loads((ROOT/'audit'/'build_summary.json').read_text(encoding='utf-8')),ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-canonical',action='store_true',help='Re-render packaged canonical records; does not require external raw participant tables.')
    main(parser.parse_args().from_canonical)
