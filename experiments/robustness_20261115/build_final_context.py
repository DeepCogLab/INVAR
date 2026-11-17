"""Build writing context only after all canonical matrices have passed acceptance."""
import csv,json,shutil,collections
from datetime import datetime,timezone,timedelta
from pathlib import Path
from campaign_common import R,load,save,sha

S=R/'supplement_overnight';SG=timezone(timedelta(hours=8))

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def p(value):return '未定义' if value in ['',None] else f'{100*float(value):.2f}%'
def x(value):return '未定义' if value in ['',None] else f'{float(value):.4f}'

def main():
    main=load(R/'results/MAIN_MATRIX_ACCEPTANCE.json');supp=load(S/'results/MATRIX_INTEGRITY_ACCEPTANCE.json')
    assert main['status']=='passed_all_eight_main_checkpoints' and main['generated_rows']==94080
    assert supp['status']=='passed_all_six_supplement_jobs' and supp['accepted_rows']==21888 and not supp['missing']
    recovery=load(R/'gpt_date_restore_20261116/latest_check.json')
    assert recovery['queue']['status']=='finished' and not recovery['issues']
    assert all(j['status']=='generated_and_scored' for j in recovery['queue']['jobs'])
    rows=read(S/'results/summary.csv');primary=read(R/'results/summary_corrected.csv')
    assert len(primary)==32 and len(rows)==32
    out=R/'results';matrix=read(S/'results/completion_matrix.csv')
    for r in matrix:assert r['status']=='complete_and_independently_checked'
    language=[r for r in rows if r['experiment']=='instruction_language']
    budget=[r for r in rows if r['experiment']=='output_budget']
    table(out/'paper_supplement_language.csv',language);table(out/'paper_supplement_budget.csv',budget)
    table(out/'paper_supplement_completion.csv',matrix)
    table(out/'paper_supplement_language_interactions.csv',read(S/'results/instruction_language_interactions.csv'))
    table(out/'paper_supplement_budget_contrasts.csv',read(S/'results/output_budget_contrasts.csv'))
    failure_counts=collections.Counter();mirror=Path('artifacts/robustness_20261115/remote')
    for row in matrix:
        root='gpt_date_restore_20261116' if row['result_source']=='engineering_date_restoration' else 'supplement_overnight'
        for a in read(mirror/root/'runs'/row['job']/'attempt_scores.csv'):
            valid=a['valid']=='True';correct=a['correct']=='True'
            if row['job'].startswith('length_goals_') and valid:assert correct,'Budget goal valid-but-wrong action must be explicitly reflected in report'
            category='valid_correct' if valid and a['task_type']=='goal' and correct else 'valid_wrong' if valid and a['task_type']=='goal' else 'valid_behavior' if valid else a['error']
            failure_counts[row['job'],a['cell'],a['stage'],a['task_type'],category]+=1
    table(out/'paper_supplement_delivery_error_counts.csv',[dict(job=k[0],cell=k[1],stage=k[2],task_type=k[3],category=k[4],count=v) for k,v in sorted(failure_counts.items())])
    lines=['# 最终新增实验发现与写作边界','',f'更新：{datetime.now(SG):%Y-%m-%d %H:%M:%S}，新加坡时间。',
        '', '## 完成范围与结果入口','',
        '预先冻结主矩阵8/8检查点、94,080次；事后单独冻结补充6/6任务、21,888次，全部生成并通过原始文件SHA、身份、计划/种子、去重试次、完整最优集合和独立算法核验。合计115,968条新增科学回答，但两个阶段不得混写成一个预先计划的确认实验。补充GPT两任务的5,664条已包含在21,888内，不能再加一次。原有实验、sources和各版原始输出保持冻结。',
        '', '主结果统一引用 [summary_corrected.csv](results/summary_corrected.csv) 及 [主矩阵验收](results/MAIN_MATRIX_ACCEPTANCE.json)。补充结果引用 [summary.csv](supplement_overnight/results/summary.csv)、[完整性验收](supplement_overnight/results/MATRIX_INTEGRITY_ACCEPTANCE.json) 和 [来源矩阵](supplement_overnight/results/completion_matrix.csv)。可写作表已集中到results/paper_supplement_*.csv；旧v1/v2只作追溯，不能引用其中受解析缺陷影响的能力数字。',
        '', '主原始及身份/adapter/token/日志镜像在 `artifacts/robustness_20261115/remote`；主统一v3评分在其同级 `corrected_v3`。每个任务的COMPLETE、SCORING_COMPLETE和独立验收均链接同一raw SHA。副本与本项目artifacts入口可追溯，大权重不重复拷进源码目录。',
        '', '## 1. 最可靠的主张是联合诊断，不是“思考后问题消失”','',
        '原生推理对目标执行的巨大收益在Qwen3-8B及SocialR1-8B两语获得新增支持；但都属于Qwen3家族，不能称已在五个家族复现同一种原生开关效果。GPT-OSS比较low/high，二者本身均推理；Llama、GLM、Mistral与SOTOPIA是提示干预，Base又使用completion格式示例，须分开报告。',
        '', '| 检查点 | 语言 | 核心Q：off→on | 参数四题全对：off→on | 行为交付：off→on |','|---|---|---:|---:|---:|']
    for model in ['qwen3_8b','socialr1_8b']:
        for lang in ['zh','en']:
            a=next(r for r in primary if r['model']==model and r['language']==lang and r['arm']=='native_off')
            b=next(r for r in primary if r['model']==model and r['language']==lang and r['arm']=='native_on')
            lines.append(f"| {model} | {lang} | {p(a['Q'])}→{p(b['Q'])} | {p(a['parameter_joint'])}→{p(b['parameter_joint'])} | {p(a['behavior_delivery'])}→{p(b['behavior_delivery'])} |")
    lines+=['', 'Q以全尝试为分母、接受全部合法最优行动；核心13条件对来源不均衡，表用对等权，来源/任务等权另列。参数20对构成24个互斥最优四元组，每语分别检验四题全对；不能把四题率与平均单题Q混用。',
        '', 'Qwen英文条件E从0.05313降至0.00537，SocialR1英文却从0.09912升至0.14110；SocialR1行为交付也明显恶化。Qwen主英文on的E缺失上界0.13623，低条件E不能消除未交付结果。自由行为没有真人标准答案，不能仅因贡献/返还集中就断言“不看题”。',
        '', '“会答规则不等于按目标执行”仍有强实例：SocialR1中文off的变化/不变事实正确率99.22%/97.46%，核心Q仅64.10%；Qwen中文off两类事实96.09%/95.12%，核心Q71.63%。但不能把它说成模型不可修复的能力上限。',
        '', '旧28配置中最低E几个模型能力较差的观察仍保留其原范围；本轮低E且高Q的Qwen on构成对“越稳定越没用”普遍化说法的反例。应写：**只测表述稳定性不能区分有效稳定、能力不足和交付缺失；必须联合报告事实、目标执行与交付。**',
        '', '## 2. 新的关键素材：系统指令语言会改变提示推理是否执行','',
        '主实验中题目与系统指令语言一起变化；补充A固定用户题目，仅交叉系统指令语言，检验同一题目语言下的[推理−直接]系统英文 − [推理−直接]系统中文。GLM和Llama在看见其主结果后被选中，因此这是有选择的事后探索，不能宣传为无选择的普遍跨家族效应。',
        '', '| 模型 | 题目语言 | 中文系统：直接→推理Q | 英文系统：直接→推理Q | 交互百分点 |','|---|---|---:|---:|---:|']
    for model in ['glm4_9b','llama31_8b']:
        for lang in ['zh','en']:
            d={(r['instruction_language'],r['arm']):float(r['core_Q_pair_equal']) for r in language if r['model']==model and r['language']==lang}
            a,b,c,e=[d[k] for k in [('zh','direct'),('zh','brief_reasoning'),('en','direct'),('en','brief_reasoning')]]
            lines.append(f'| {model} | {lang} | {p(a)}→{p(b)} | {p(c)}→{p(e)} | {100*((e-c)-(b-a)):+.2f} |')
    lines+=['', '两个模型的中文题也在英文系统推理请求下明显改善，不能把主实验中较弱的中文收益全部归为中文题目理解差。GLM有英文系统下输出中文计算的固定SHA实例；这些是可见依从/输出证据，不证明内部机制。',
        '', '复杂参数四题仍体现大差异：英文系统+提示推理时，GLM中文/英文四题率58.33%/70.83%，Llama仅6.25%/10.42%。因此“英文指令即可修复”也不成立。来源/任务等权交互见paper_supplement_language_interactions.csv；核心题和参数题分别报告。',
        '', '## 3. 输出预算主要修复长尾交付，不能保证正确或更稳定','',
        '补充B固定Qwen native_on、GPT-OSS effort_high，在共同context12288内比较4096/8192输出上限；每臂为独立新种子，不回填主实验失败、不假设相同采样轨迹。目标每输入2次，行为每输入4次。',
        '', '| 模型 | 题目语言 | 上限 | 行为交付率 | 截断率 | 平均输出token | 条件E | E缺失界 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in budget:
        if r['job'].startswith('length_behavior_'):
            lines.append(f"| {r['model']} | {r['language']} | {r['max_tokens']} | {p(r['core_behavior_delivery'])} | {p(r['core_behavior_truncated'])} | {float(r['core_behavior_mean_completion_tokens']):.1f} | {x(r['behavior_pair_E_conditional'])} | [{x(r['behavior_pair_E_low'])}, {x(r['behavior_pair_E_high'])}] |")
    lines+=['', 'Qwen英文行为交付93.12%→99.87%，中文99.07%→99.87%。英文4096的完整面板条件E因至少一个输入没有有效样本而未定义，不能补零或只删缺失题后声称E下降。8192的E约0.013–0.015，同题拆半W1约0.012–0.019，处在同量级；有限n4抽样不足以精确断言真实提示差异接近零。',
        '', 'GPT-OSS high行为交付中文98.41%→100%、英文97.88%→100%，两语截断均降至零；但条件E中文0.02350→0.02839、英文0.00684→0.00781，并没有同步降低。其同题拆半噪声也不可忽略，不能把这些小幅E点差写成确定的表述敏感性恶化。更长预算在这两个模型中支持交付改善，不能支持稳定性单调改善。',
        '', '两模型目标题在4096时已近天花板：Qwen核心Q中文99.52%→99.68%、英文99.68%→98.56%，参数四题两语两cap均100%。GPT核心Q中文均100%、英文99.68%→100%，但中文参数四题100%→97.92%，参数单题Q100%→99.38%、有效交付192/192→191/192，并非更长预算处处更好。',
        '', '原始失败检查表明：GPT中文参数8192的唯一未交付将整数1500写成JSON字符串"1500"；Qwen英文核心8192有7次类型失败，4096为1次截断及1次类型失败。在补充B两个目标job的5,266条schema有效输出中，行动全部属于完整最优集合；此范围不包括主实验或补充A。截断减至零不等于schema交付或全尝试Q单调改善。这是输出类型/交付边界，不能写成合法目标推理能力下降；也不能通过宽松转整数救分。各cap独立抽样，这些差异不识别输出长度降低能力的机制。',
        '', '行为E/G不定义正确的人类行为；补充只对交付长尾和抽样/预算敏感性作诊断。SocialR1未纳入长度对照，因此不能声称其截断已经被证明可修复。主实验context8192与补充12288不同，cap因果对比只在补充两臂内进行。',
        '', '## 4. 专门化与跨家族结果必须保留反例','',
        'SocialR1相对Qwen3-8B的核心Q四格均低，参数四题也弱；部分事实指标却更高。SOTOPIA相对共同Mistral底座的中文核心Q高14.42个百分点（直接）和11.06个百分点（提示），但四格事实检查及参数单题Q未同步改善，参数四题均0。旧“专门模型没有优势”应收窄为：**没有跨诊断维度一致的优势，存在检查点与任务层面的正反差。**',
        '', 'SOTOPIA与Mistral底座权重、模板、tokenizer及输入token逐SHA匹配；LoRA实际加载和请求ID由冻结worker断言，raw未单独存lora_id字段。它仍是共同单次JSON接口下的适配器，不能代表原论文完整多轮社会代理。原训练commit未完整恢复，训练语言与中英测试的匹配也需限定，不能据此识别专门训练的因果效果。详见matched_specialization_contrasts.csv及MISTRAL_MATCHED_PAIR_NOTE.md。',
        '', 'GPT-OSS low本身会推理，主核心Q已99.84%–100%，high没有普遍核心收益；参数四题则从79.17%/95.83%升至100%/100%。Llama/GLM英文提示核心收益与Mistral/SOTOPIA反向或微小结果并列，不跨接口合成一个“思考效应”。Base使用固定格式示例的completion接口，另列诊断，不能直接作为同接口聊天排名。',
        '', '## 5. 聚合与成本是结论的一部分','',
        '主Llama中文提示的对等权核心Q59.13%→57.85%，来源等权59.46%→60.59%，方向反转；Mistral中文也有权重方向差异。不能挑权重写单一可靠收益。行为面板每来源8对，故对等权与来源等权相同是构造结果。',
        '', 'Qwen主核心目标平均输出中文11.93→553.11 tokens、英文11.87→640.05；GPT-OSS low→high中文67.03→489.56、英文69.48→556.16。输出token包含思考/通道内容，按全部尝试统计；不能等同硬件无关的延迟或不同tokenizer间严格可比的计算量。运行时间、交付、截断应与正确率并报。',
        '', '## 6. 对旧结论的最终处理与论文取舍','',
        '| 旧表述 | 新增证据后的写法 |','|---|---|',
        '| 开启思考后问题基本消失 | 目标执行在Qwen原生开关下显著改善；自由行为、交付和其他接口不统一 |',
        '| 最稳定的模型最没用 | 保留旧28配置的观察；跨数据的一般主张改为稳定性不足以判断有效性 |',
        '| 会答规则不等于会用规则 | 得到新增支持，但需写明推理配置，不能忽略原生on高Q |',
        '| 社会专门模型无优势 | 改为维度间无一致优势，承认SOTOPIA核心Q正差及部分事实反差 |',
        '| 中文/英文差别说明语言能力差 | 主有指令语言混杂；补充交叉揭示请求语言交互，机制仍未识别 |',
        '| 延长输出即可修复 | 交付改善具有任务/模型条件，正确率不保证单调，SocialR1未检验 |',
        '| 一致性训练可普遍修复 | 本轮没有新训练，仍按旧多来源/种子负面与异质结果报告 |',
        '', '建议短文优先：联合可靠性评测与事实/目标/交付错配；原生推理正面结果及跨接口反例；系统指令语言对照作为最有区分度的新增材料；长度交付边界作为精简支持。旧训练压缩成有限范围的异质结果，完整证据放可审计材料，勿为增加素材而平均铺陈所有实验。',
        '', '## 7. 不可省略的局限','',
        '只有四个既有来源，没有新的独立人类研究、真人行为对照或人类语义认证；英文由AI翻译及AI/程序检查。主仅取32对行为/事实、13对核心目标及20对参数切片，两种表述变体，不能冒充旧92对全部改写的英文覆盖。核心来源不均衡且参数四元组共享输入；试次数不等于独立情境数。',
        '', '只覆盖本地可运行的8检查点/5家族，社会模型与底座匹配范围有限，没有闭源商业模型；不能由模型发布地域推断文化机制。行为主n8、补充n4，W1有限抽样偏差、同题拆半噪声及缺失界必须保留；500次共享prompt重采样是敏感性范围，不是经校准置信区间。自由选择没有规范正确行动，稳定不是“接近真人”。',
        '', '补充在观察主结果后选择模型/问题，预先冻结的是执行计划而不是此前未知的研究假设；不得把探索结果伪装为独立确认。没有通过模型/预算/来源选择来删除反例或按效果重试。',
        '', '## 8. 工程偏离与评分修订完整披露','',
        '主生成未重跑。CPU评分v3修正Harmony final格式元数据及空白行重复计JSON起点，两版历史、原raw和更正manifest均保留；全部94,080行与1,376汇总值由不同解析/运输/权重算法核验。拒绝重复JSON键、不从未完成思考或analysis通道救答案；全尝试分母不变。详见SCORING_ERRATUM_TERMINAL_V3.md。',
        '', '补充GPT模板含自动日期：午夜前预检为Nov15、运行到Nov16时全render map断言停止，发生在模型加载、identity和任何回答之前。原两个failed_no_retry记录与空目录/日志保留。独立工程分支只将模板日期恢复为原Nov15，两个不同CPU实现证明全部2076个输入key文本及token map精确等于原预检；33个原绑定文件字节不变，固定计划/种子/模型/输出cap不变，随后每项运行一次，无科学回答重抽。',
        '', '这是明确披露的零交付前工程协议偏离，不能写成原补充无偏离运行。原scientific freeze d4723b…52b45，恢复freeze 1ea33f…da59d；每完整任务ENGINEERING_RECOVERY_LINK将新raw连至原失败日志SHA、双freeze和prior_answer_count=0。见gpt_date_restore_20261116/RECOVERY_PROTOCOL.md、BOOTSTRAP_CPU_ACCEPTANCE.json和来源矩阵。硬截止不延长，仅使用授权GPU1，GPU0既有服务不操作。',
        '', '## 9. 可复用文件与复算路径','',
        '- 主科学范围：PROTOCOL.md、freeze.json；主总表results/summary_corrected.csv、控制/成本表paper_controls_delivery_cost.csv、任务策略分表goal_task_policy_breakdown.csv、专门化匹配表matched_specialization_contrasts.csv。',
        '- 补充科学范围：supplement_overnight/PROTOCOL.md、freeze.json；结果summary.csv、交互及预算差表、MATRIX_INTEGRITY_ACCEPTANCE.json；写作副本results/paper_supplement_*.csv。',
        '- 独立原始验收：主independent_audits_v3与accept_main_matrix.py；补充每job INDEPENDENT_ACCEPTANCE.json、accept_supplement_matrix.py；同步latest_sync.json与D盘snapshots逐文件SHA。',
        '- 准确的执行状态与停止验收：results/FINAL_CAMPAIGN_ACCEPTANCE.json；写作主张须同时引用此报告和对应CSV/验收，不只引用截图或聊天总结。']
    (R/'FINAL_FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sources=[R/'FINAL_FINDINGS.md',R/'results/summary_corrected.csv',R/'results/MAIN_MATRIX_ACCEPTANCE.json',S/'results/summary.csv',S/'results/MATRIX_INTEGRITY_ACCEPTANCE.json',S/'results/ACCEPTANCE_INDEX.json',R/'gpt_date_restore_20261116/recovery_freeze.json',R/'results/FINAL_CAMPAIGN_ACCEPTANCE.json']+sorted(out.glob('paper_supplement_*.csv'))
    save(out/'FINAL_WRITING_PROVENANCE.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),files={str(p.relative_to(R)):sha(p) for p in sources},
        main_rows=94080,supplement_rows=21888,total_new_rows=115968,main_and_posthoc_kept_separate=True,engineering_restoration_not_added_twice=True))
    print(json.dumps(dict(final_findings=str(R/'FINAL_FINDINGS.md'),main_rows=94080,supplement_rows=21888)))

if __name__=='__main__':main()
