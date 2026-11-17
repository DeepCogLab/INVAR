"""Prepare audited V2; all changes are rule-based and independent of behavior scores."""
import copy,collections,hashlib,json,random,subprocess,sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'INVAR_dataset'
sys.path.insert(0,str(D))
import build_dataset as b

def read(path):return [json.loads(s) for s in path.open(encoding='utf-8-sig') if s.strip()]
def write(path,rows):path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

changes=[]
for name in ('cases','pairs'):
    shutil.copyfile(R.parent/'INVAR_dataset/data'/f'{name}.jsonl',D/'data'/f'{name}.jsonl')
cases=read(D/'data/cases.jsonl')
for c in cases:
    if c['case_id'].startswith('HPVBG_UG_'):
        old=c['relation_alt']['text']
        c['relation_alt']['text']=old.replace('按照P1提出的分配，P2将得到','P1提出的分配方案中，给P2分配').replace('P1给自己留下','给P1分配')
        changes.append(dict(case_id=c['case_id'],field='relation_alt.text',before=old,after=c['relation_alt']['text'],reason='Keep allocation proposal conditional on acceptance'))
    if c['case_id'].startswith('IGVOFO_DG_THRESHOLD'):
        for field,value in [('question','你拟定的分配方案中给P2多少点，即g？请输出合法菜单中的点数；最终到账点数按上述执行规则计算。')]:
            changes.append(dict(case_id=c['case_id'],field=field,before=c[field],after=value,reason='Disambiguate proposed allocation from realized payoff'))
            c[field]=value
        for field,value in [('relation_alt.text','现在分配方案中拟划给P2多少点，由P1决定；最终到账点数另按执行规则计算。'),
                            ('situation.0','现在由P1拟定分配方案，选择方案中给P2的点数g；最终到账点数按执行规则计算。')]:
            changes.append(dict(case_id=c['case_id'],field=field,before=c['relation_alt']['text'] if field.startswith('relation') else c['situation'][0],after=value,reason='Disambiguate proposed allocation from realized payoff'))
            if field.startswith('relation'):c['relation_alt']['text']=value
            else:c['situation'][0]=value
    if 'source_game_tree' in c:
        c['original_source_game_tree']=c.pop('source_game_tree')
write(D/'data/cases.jsonl',cases)
b.VERSION='0.2.0';b.main(from_canonical=True)
cases={c['case_id']:c for c in read(D/'data/cases.jsonl')}
pairs=read(D/'data/pairs.jsonl');ps=read(D/'data/prompts.jsonl')
qs=read(D/'data/quartets.jsonl')
assert len(ps)==1240 and len(qs)==701 and len(pairs)==92
(R/'revision_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
models=['qwen3_8b','qwen25_7b_instruct','qwen25_7b_base','qwen25_14b_instruct','glm4_9b','llama31_8b_instruct','mistral7b_instruct_v03','deepseek_llm7b_chat']

def obj_schema(field,schema):
    return {'type':'object','properties':{field:schema},'required':[field],'additionalProperties':False}
def action_schema(space):
    s={'type':'integer' if space['type']=='integer' else 'string'}
    if 'values' in space:s['enum']=space['values']
    else:s.update(minimum=space['min'],maximum=space['max'])
    return obj_schema('action',s)
inputs={}
for p in ps:
    p['output_schema']=action_schema(p['action_space']);p['task_type']='behavior'
    inputs[p['prompt_id']]=p

def leaf(c,action):
    p=c['payoffs'];p=p.get('RIGHT',p)
    return p[action]

def facts(p,a,c):
    f=p['intervention']['factor']
    if f=='automatic_minimum_allocation_threshold':
        return ['若分配方案g=1000，P1最后得到5000点。','若分配方案g=1000，P1最后得到0点。'],['若g=2000，P1最后得到4000点。','若g=2000，P1最后得到0点。']
    if f=='connection_between_display_and_actual_contributions':
        return ['条件表表头来自本组三名X类参与者实际贡献的取整平均数。','条件表表头来自另外三个独立随机数的取整平均数。'],['本组三名X类参与者的实际贡献由计算机独立均匀抽取0至10的整数决定。','本组三名X类参与者的实际贡献由其本人此刻自主决定。']
    if f=='first_mover_action':
        return ['P1已经自主选择LEFT。','P1已经自主选择RIGHT。'],['P2在当前节点可选择X或Y。','P2在当前节点只能选择X。']
    if f=='first_mover_agency':
        return ['当前RIGHT由P1在原可选LEFT或RIGHT的情况下自主选择。','当前RIGHT由程序指定，P1没有选择权。'],['当前已经到达RIGHT之后的节点，轮到P2选择。','当前已经到达LEFT之后的节点，轮到P2选择。']
    if f=='individual_or_group_offer':
        return ['若实施惩罚，有1名提议者的收益受到扣减。','若实施惩罚，有3名提议者的收益分别受到扣减。'],['实施惩罚时，你自己需要支付1点。','实施惩罚时，你自己需要支付3点。']
    if f=='mutual_return_payoff_R':
        r0=leaf(a,'NORTH')['P1'];r1=leaf(c,'NORTH')['P1']
        return [f'若P1选RIGHT且P2选NORTH，双方各得到{r0}点。',f'若P1选RIGHT且P2选NORTH，双方各得到{r1}点。'],['若P1选RIGHT且P2选EAST，双方收益不相等。','若P1选RIGHT且P2选EAST，双方收益相等。']
    if f=='offer_to_receiver':
        # Actual accept payoffs, not a generic ultimatum-game convention.
        x=a['payoffs']['accept'].get('self',a['payoffs']['accept'].get('P2'))
        y=c['payoffs']['accept'].get('self',c['payoffs']['accept'].get('P2'))
        return [f'如果P2接受本次方案，P2得到{x}点。',f'如果P2接受本次方案，P2得到{y}点。'],['P2拒绝本次方案后，双方本题均得到0点。','P2拒绝本次方案后，双方仍按原方案得到点数。']
    if f=='origin_of_type_x_contributions':
        return ['本组三名X类的实际贡献采用各自本人在此前独立任务中的选择。','本组三名X类的实际贡献采用其他组随机选中参与者在此前任务中的选择。'],['条件表表头来自本组三名X类实际贡献的取整平均数。','条件表表头来自与实际贡献独立的另一组随机数。']
    if f=='rejection_right':
        return ['P1选RIGHT后，P2仍可自主选择NORTH或EAST。','P1选RIGHT后直接执行EAST，P2不能选择NORTH。'],['P1仍可自主选择LEFT或RIGHT。','P1只能选择RIGHT。']
    if f=='return_decision_right':
        return ['P1选RIGHT后，P2仍可自主选择NORTH或EAST。','P1选RIGHT后固定执行NORTH，P2不能选择EAST。'],['P1选择LEFT会直接结束本题。','P1选择LEFT后仍由P2选择NORTH或EAST。']
    if f=='return_determination_rule':
        return ['若P1送出2000点，P1在作决定时不能确定P2返还的具体点数。','若P1送出2000点，公开制度会自动令P2返还2000点。'],['若P1送出2000点，P2先收到4000点。','若P1送出2000点，P2先收到2000点。']
    if f=='second_mover_rule':
        return ['P1选择LEFT后，P2仍可自主选择WEST或SOUTH。','P1选择LEFT后固定执行SOUTH，P2不能选择WEST。'],['P1仍可自主选择LEFT或RIGHT。','P1只能选择LEFT。']
    if f=='source_of_initial_transfer':
        s=a['payoffs']['parameters']['original_sent']
        return [f'最初的{s}点由P1自主决定送出。',f'最初的{s}点由程序强制从P1转出，P1没有选择权。'],[f'P2已收到{2*s}点。',f'P2已收到{s}点。']
    if f=='unequal_allocation_T_and_P':
        x,y=leaf(a,'EAST'),leaf(c,'EAST')
        return [f'若P1选RIGHT且P2选EAST，P1得到{x["P1"]}点，P2得到{x["P2"]}点。',f'若P1选RIGHT且P2选EAST，P1得到{y["P1"]}点，P2得到{y["P2"]}点。'],['若P1选RIGHT且P2选EAST，两人的收益总和是100点。','若P1选RIGHT且P2选EAST，两人的收益总和是200点。']
    raise ValueError(f)

probe_quartets=[];probe_checks=[]
for p in pairs:
    a,c=cases[p['case0_id']],cases[p['case1_id']]
    change,stable=facts(p,a,c)
    for typ,options in [('changed_fact',change),('stable_fact',stable)]:
        index={}
        for v in ['base']+[v for v in b.VARIANTS if b.applicable(a,c,v)]:
            swap=int(hashlib.sha256((p['pair_id']+'|'+v+'|'+typ).encode()).hexdigest()[:8],16)%2
            order=[1,0] if swap else [0,1]
            for side,case in enumerate([a,c]):
                body=b.render(case,v)[1]['content'].split('【本次决策】')[0]
                opts=[options[i] for i in order]
                if v=='V5_ids':opts=[b.renumber(x) for x in opts]
                question='【独立事实检查】\n下面哪一项符合本题当前条件？只判断事实，不作社会行为选择。\nA：'+opts[0]+'\nB：'+opts[1]+'\n只输出JSON对象，仅包含answer字段，值为A或B。'
                messages=[{'role':'system','content':'请依据题目材料回答事实问题，只输出要求的JSON对象。'},{'role':'user','content':body+question}]
                pid='f_'+digest(messages)[:20]
                correct_option=side if typ=='changed_fact' else 0
                expected='AB'[order.index(correct_option)]
                item=dict(prompt_id=pid,messages=messages,output_schema=obj_schema('answer',{'type':'string','enum':['A','B']}),
                          task_type=typ,expected_answer=expected,pair_id=p['pair_id'],condition=side,variant_id=v,
                          task=p['task'],factor=p['intervention']['factor'],semantic_answer=correct_option,
                          option_semantic_order=order,body_case_id=case['case_id'])
                if pid in inputs:assert inputs[pid]['expected_answer']==expected
                inputs[pid]=item;index[v,side]=pid
        for v in [v for v in b.VARIANTS if b.applicable(a,c,v)]:
            probe_quartets.append(dict(quartet_id=p['pair_id']+'__'+typ+'__'+v,pair_id=p['pair_id'],task=p['task'],
                                      factor=p['intervention']['factor'],probe_type=typ,variant_id=v,
                                      A=index['base',0],B=index[v,0],C=index['base',1],D=index[v,1]))
        probe_checks.append(dict(pair_id=p['pair_id'],probe_type=typ,condition_descriptions=options,source='actual effective case rules/payoffs; no human response labels'))

# Role checks: global candidate set includes wrong IDs. Never force the focal ID.
ids=list(b.IDMAP)+list(b.IDMAP.values())
for old in read(D/'controls/role_comprehension.jsonl'):
    item=copy.deepcopy(old)
    item['messages'][1]['content']=item['messages'][1]['content'].split('【事实理解检查】')[0]+'【事实理解检查】\n你在本题的参与者编号是什么？不作行为选择。只输出JSON对象，仅含answer字段，值为题目中的实际编号，不要输出“编号”占位词。'
    item['prompt_id']='role_'+digest(item['messages'])[:20]
    item['output_schema']=obj_schema('answer',{'type':'string','enum':ids});item['task_type']='role'
    inputs[item['prompt_id']]=item

# Positive controls with explicitly stated objectives; no relabeling free choices.
goal_quartets=[];goal_audit=[]
policies={'own':'只最大化你本人在本题最终得到的点数；不要考虑其他目标。',
          'total':'只最大化本题所有参与者最终得到的点数总和；不要考虑其他目标。',
          'equality':'只最小化本题所有参与者最终点数中最高值与最低值的差；不要考虑其他目标。'}
def utilities(case,policy):
    space=case['action_space'];actions=space.get('values')
    if actions is None:return None
    pays=case['payoffs'].get('by_action',case['payoffs']);result={}
    for action in actions:
        row=pays.get(action,pays.get(str(action)))
        if isinstance(row,dict):
            row=dict(row)
            if 'self' in row:row.setdefault(case['focal_id'],row['self'])
        if not isinstance(row,dict) or not all(k in row and isinstance(row[k],(int,float)) for k in case['entities']):return None
        values=[row[k] for k in case['entities']]
        result[action]=row[case['focal_id']] if policy=='own' else sum(values) if policy=='total' else -(max(values)-min(values))
    best=max(result.values());return result,[k for k,v in result.items() if v==best]
for p in pairs:
    a,c=cases[p['case0_id']],cases[p['case1_id']]
    for policy,objective in policies.items():
        ua,uc=utilities(a,policy),utilities(c,policy)
        if ua is None or uc is None:continue
        switch=len(ua[1])==len(uc[1])==1 and ua[1]!=uc[1]
        goal_audit.append(dict(pair_id=p['pair_id'],policy=policy,condition0=ua,condition1=uc,unique_optimum_switch=switch))
        index={}
        for v in ['base']+[v for v in b.VARIANTS if b.applicable(a,c,v)]:
            for side,case in enumerate([a,c]):
                messages=b.render(case,v)
                messages[1]['content']=messages[1]['content'].replace('【本次决策】','【本次明确目标】\n'+objective+'\n\n【本次决策】')
                pid='goal_'+digest(messages)[:20]
                if pid in inputs:assert inputs[pid]['expected_actions']==[ua,uc][side][1]
                inputs[pid]=dict(prompt_id=pid,messages=messages,output_schema=action_schema(case['action_space']),
                                 task_type='goal',pair_id=p['pair_id'],variant_id=v,condition=side,policy=policy,
                                 expected_actions=[ua,uc][side][1],unique_optimum_switch=switch,task=p['task'],action_space=case['action_space'])
                index[v,side]=pid
        for v in [v for v in b.VARIANTS if b.applicable(a,c,v)]:
            goal_quartets.append(dict(quartet_id=p['pair_id']+'__goal_'+policy+'__'+v,pair_id=p['pair_id'],task=p['task'],policy=policy,
                                     variant_id=v,unique_optimum_switch=switch,A=index['base',0],B=index[v,0],C=index['base',1],D=index[v,1]))

mirror={}
for item in list(inputs.values()):
    if item['task_type'] not in ('changed_fact','stable_fact'):continue
    other=copy.deepcopy(item)
    text=other['messages'][1]['content']
    prefix,tail=text.rsplit('\nA：',1)
    opta,tail=tail.split('\nB：',1);optb,suffix=tail.split('\n只输出JSON',1)
    other['messages'][1]['content']=prefix+'\nA：'+optb+'\nB：'+opta+'\n只输出JSON'+suffix
    other['prompt_id']='f_'+digest(other['messages'])[:20]
    other['expected_answer']='B' if item['expected_answer']=='A' else 'A'
    other['option_semantic_order']=list(reversed(item['option_semantic_order']))
    if other['prompt_id'] in inputs:assert inputs[other['prompt_id']]['expected_answer']==other['expected_answer']
    inputs[other['prompt_id']]=other;mirror[item['prompt_id']]=other['prompt_id']
original_probe_quartets=list(probe_quartets)
for q in original_probe_quartets:
    q['option_order_version']='original'
    mirrored=dict(q,quartet_id=q['quartet_id']+'__mirror',option_order_version='mirror')
    mirrored.update({tag:mirror[q[tag]] for tag in 'ABCD'})
    probe_quartets.append(mirrored)
write(R/'inputs.jsonl',sorted(inputs.values(),key=lambda x:x['prompt_id']))
write(R/'fact_quartets.jsonl',probe_quartets);write(R/'fact_checks.jsonl',probe_checks)
write(R/'goal_quartets.jsonl',goal_quartets);write(R/'goal_audit.jsonl',goal_audit)
repeat_ids=[]
for key,group in __import__('itertools').groupby(sorted(ps,key=lambda x:json.dumps(x['action_space'],sort_keys=True)),key=lambda x:json.dumps(x['action_space'],sort_keys=True)):
    group=list(group); repeat_ids.extend(p['prompt_id'] for p in sorted(group,key=lambda x:digest(x['prompt_id']))[:max(1,round(len(group)*.1))])
plans=[]
for m in models:
    for p in inputs.values():
        reps=256 if p['task_type']=='behavior' else 8 if p['task_type'] in ('changed_fact','stable_fact') else 16 if p['task_type']=='goal' else 32
        for rep in range(reps):
            batch='main' if p['task_type']=='behavior' else p['task_type']
            trial=digest(['v2',m,p['prompt_id'],rep,batch])[:24]
            plans.append(dict(model=m,prompt_id=p['prompt_id'],replicate=rep,batch=batch,trial_id=trial))
    for pid in repeat_ids:
        for rep in range(256):
            plans.append(dict(model=m,prompt_id=pid,replicate=rep,batch='repeat',trial_id=digest(['v2',m,pid,rep,'repeat'])[:24]))
random.Random(20261024).shuffle(plans)
write(R/'main_plan.jsonl',plans)
manifest=dict(version='0.2.0',models=models,inputs_by_type=dict(collections.Counter(p['task_type'] for p in inputs.values())),
              plan_rows=len(plans),rows_per_model=len(plans)//8,rows_by_batch_per_model={k:v//8 for k,v in collections.Counter(p['batch'] for p in plans).items()},
              behavior_prompts=1240,behavior_quartets=701,condition_pairs=92,behavior_reps=256,repeat_prompts=len(repeat_ids),
              fact_reps_per_order=8,fact_options_counterbalanced=True,goal_reps=16,role_reps=32,goal_switch_pairs=dict(collections.Counter(x['policy'] for x in goal_audit if x['unique_optimum_switch'])),
              input_sha256=hashlib.sha256((R/'inputs.jsonl').read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((R/'main_plan.jsonl').read_bytes()).hexdigest(),
              decoder_constraint_scope='format and candidate action menu, never expected answers',all_sources_in_one_main_analysis=True,
              human_behavior_alignment=False,review='program checks and AI-assisted source/semantic review; no claim of independent human certification')
(R/'frozen_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
