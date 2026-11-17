"""Complete accepted mechanism evidence and four-page-paper addendum."""
import csv
import math
from common import *
MIRROR=Path('artifacts/mechanism_20261113_v2')
RESULT=R.parents[1]/'results/20261113_训练机制补实验'
LABELS={'train':'训练池内','heldout_source':'来源留出','heldout_variant':'呈现留出'}


def csv_rows(name):
    with (RESULT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def value(row,key):return float(row[key]) if row.get(key) not in ('',None) else None
def mean(values):return math.fsum(values)/len(values) if values and all(v is not None for v in values) else None
def fmt(x):return '—' if x is None else f'{x:+.5f}'


def main():
    acc=load(MIRROR/'local_mirror_acceptance.json');assert acc['all_jobs_complete'] and acc['independent_scalar_audit']=='passed_all_mirrored_jobs'
    provenance=load(RESULT/'mechanism_table_provenance.json');assert provenance['status']=='complete'
    contrasts=csv_rows('accepted_JS_minus_SFT_all_checkpoints.csv');points=csv_rows('accepted_points_all_checkpoints.csv')
    signs=csv_rows('accepted_descriptive_sign_counts.csv')
    train=[c for c in contrasts if c['stratum']=='train']
    reverse=[c for c in contrasts if value(c,'delta_proxy_W1') < -1e-12 and value(c,'delta_decoder_W1_conditional') > 1e-12]
    reverse_cross_zero=sum(value(c,'delta_decoder_W1_shared_prompt_bootstrap_empirical_lower')<=0<=value(c,'delta_decoder_W1_shared_prompt_bootstrap_empirical_upper') for c in reverse)
    end_goal={}
    for job in MODELS:
        end_goal[job]={}
        for state in ('anchor_sft_150','anchor_js_150'):
            selected=[p for p in points if p['model']==job and p['state']==state and value(p,'Q_all_attempts') is not None]
            weight=sum(int(p['goal_prompts']) for p in selected)
            end_goal[job][state]=math.fsum(value(p,'Q_all_attempts')*int(p['goal_prompts']) for p in selected)/weight
    summary=[]
    for job in MODELS:
        for step in (50,100,150):
            for stratum in LABELS:
                selected=[r for r in contrasts if r['model']==job and int(r['step'])==step and r['stratum']==stratum]
                assert len(selected)==4
                summary.append(dict(model=job,step=step,stratum=stratum,
                    delta_proxy_W1=mean([value(r,'delta_proxy_W1') for r in selected]),
                    delta_proxy_JS=mean([value(r,'delta_proxy_JS') for r in selected]),
                    delta_decoder_W1=mean([value(r,'delta_decoder_W1_conditional') for r in selected])))
    counts={}
    for job in MODELS:
        bridge=jl(MIRROR/'runs'/job/'analysis/bridge.jsonl')
        goals=jl(MIRROR/'runs'/job/'analysis/goals.jsonl')
        counts[job]=dict(failures=sum(x['failures'] for x in bridge),attempts=sum(x['attempts'] for x in bridge),
            behavior_failures=sum(x['failures'] for x in bridge if x['task_type']=='behavior'),
            goal_failures=sum(x['failures'] for x in bridge if x['task_type']=='goal'),
            goal_attempts=sum(x['attempts'] for x in goals),
            goals_correct=round(math.fsum(x['Q']*x['attempts'] for x in goals)))
    lines=['# 训练机制补实验：完成总结与结论边界','',
        '两模型的309,120条科学回答、16条工程先导和5,754个完整候选分布均已完成。原始逐SHA镜像、本地完整评分与另写标量实现通过；三个实际服务恢复并核对argv/GPU/cwd/模型端点。没有新增训练，也未选择最佳检查点。','',
        '本轮回答两个问题：固定题上的代理一致性变化是否迁移；同题、同菜单的实际生成是否跟随代理变化。使用预定旧种子20261026、两模型四来源、SFT/JS各50/100/150步，全部48检查点保留。','',
        '## 主要发现','',
        f"固定训练池的{len(train)}组模型×来源×检查点比较中，代理W1有{sum(value(c,'delta_proxy_W1') < -1e-12 for c in train)}组降低，真实生成W1有{sum(value(c,'delta_decoder_W1_conditional') < -1e-12 for c in train)}组降低。应承认所测JS配方在这一预定小菜单诊断上的正面改善；这里的24组由检查点而非三个旧训练种子构成，与旧24组种子匹配统计不是同一分母。",'',
        f"全部72项分层比较中，有{len(reverse)}项出现代理W1下降而生成W1上升的点估计；其中{reverse_cross_zero}项的500次共享提示经验范围跨0。因此这些点差异不支持声称已检出稳定的反向生成效应；经验范围也不是校准置信区间，不能据此证明两接口等价。",'',
        '150步端点的四面板等权实际W1：Qwen3在三层均降低；Qwen2.5在训练池/呈现留出降低，来源留出约+0.00052。Qwen2.5来源留出层的平均代理JS在三个检查点均降低，而平均代理W1均略升，说明训练的JS与W1在代理层面也不保证同向。来源留出的均匀改善仍缺乏支持，但不应把这个小幅点差称为普遍失败。','',
        '所选目标面板150步端点：Qwen3的Q为'+f"{end_goal['qwen3_8b']['anchor_sft_150']:.4%}→{end_goal['qwen3_8b']['anchor_js_150']:.4%}；Qwen2.5为{end_goal['qwen25_14b_instruct']['anchor_sft_150']:.4%}→{end_goal['qwen25_14b_instruct']['anchor_js_150']:.4%}。每臂88个面板-提示引用、每引用16次，共1,408次尝试；按引用等权，不能混写成原全部1,410目标题的Q。",'',
        '## 全部检查点的四留出面板等权点差值','',
        '各单元是四个来源面板的JS−SFT差值等权平均，每个面板每层16条固定小菜单行为边。负值表示该指标降低。此处只汇总点估计，不组合面板经验范围来制造整体置信区间。','',
        '| 模型 | 步数 | 分层 | Δ代理W1 | Δ代理JS | Δ实际生成W1 |','|---|---:|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['model']} | {r['step']} | {LABELS[r['stratum']]} | {fmt(r['delta_proxy_W1'])} | {fmt(r['delta_proxy_JS'])} | {fmt(r['delta_decoder_W1'])} |")
    lines+=['','## 保留每个面板的异质性','',
        '每模型每层四面板×三个检查点共12项。以下计数是固定面板上的描述，不把共享来源/检查点当独立样本；完整数值及500次共享提示经验范围在accepted_JS_minus_SFT_all_checkpoints.csv。','',
        '| 模型 | 分层 | 比较项 | 代理W1降低 | 实际W1降低 | 代理降低且实际升高 |','|---|---|---:|---:|---:|---:|']
    for s in signs:lines.append('| '+' | '.join([s['model'],LABELS[s['stratum']],s['comparisons'],s['proxy_W1_decreases'],s['decoder_W1_decreases'],s['proxy_W1_down_decoder_W1_up']])+' |')
    lines+=['','## 完整性与目标控制','']
    for job,c in counts.items():
        lines.append(f"- {job}：154,560条科学尝试；行为失败{c['behavior_failures']}，目标失败{c['goal_failures']}。所选目标面板的全部{c['goal_attempts']:,}次尝试中，目标正确{c['goals_correct']:,}次。该合计跨越不同适配器与检查点，仅用于完整性控制，不作为方法效果分数。")
    lines+=['','代理正确集合概率、全尝试Q和失败数均按面板/状态/分层保留在accepted_points_all_checkpoints.csv。PGG来源留出没有目标题，该层Q不填0、不虚构真值。',
        '', '行为每提示64次、目标16次，温度0.7/top-p0.95/n=1/maxnew128/context4096，Qwen3关闭思考。目标正确集合与模型输入分离；所有非法、类型、截断失败保留，不从内部推理拾取答案。',
        '', '## 可以与不可以支持的结论','',
        '这组结果直接检验预定小菜单、固定题上的代理与实际生成关系。训练池、来源留出、呈现留出分开；同时比较同一单位的W1和原训练的JS，避免把不同指标量纲直接称为解码缺口。全部50/100/150步都报告，不因最终结果选中间最好一步。',
        '', '生成W1/JS是可解析子集的conditional量，目标Q是全部尝试。缺失质量W1界与经验抽样范围分开；500次bootstrap和16/32/64次下采样只给经验范围，不是校准置信区间。未定义抽样不补零。代理计算也有BF16及后端数值条件，不作无限精度断言。',
        '', '即使代理与生成异向，也只能说明所测代理/解码接口不能互相替代；完整序列归一化、逐token约束、温度、top-p和后端均不同，不能把单一因素认作已识别原因。若二者同向降低，须承认改善，不追预设负面。',
        '', '本轮为已观察来源上的一个旧种子、小菜单机制诊断。呈现留出按等价边，原表达端可能已在训练；不是全提示未见、新来源泛化或新增训练算法。此处边均值不替代原92条件对主权重E，不计算未闭合G/R；自由社会行为没有据低熵判错的规范真值。',
        '', '## 工程修订与保存','',
        '初版独立对齐校验的attention_mask遗漏在任何科学回答交付前暴露，原版本、日志和复现探针完整保留。v2固定评分助手不变，仅使校验的mask与投影位置条件相同。其后只读交付审计重复计算词表大小过慢，另存等价缓存审计器；只中止旧审计子进程，不重跑生成。原controller异常状态保留，最终后处理以postprocessing_recovery记录为准。',
        '', '科学冻结SHA：3fa77d5c406c43cd12d01742903f9b6a79dad8c47cd6d5bea3cd25924fca23a5。完整原始、评分、模型/检查点身份及逐SHA清单位于artifacts/mechanism_20261113_v2；最终报告包与完整接受记录同目录入口列明。']
    (RESULT/'完成总结与结论边界.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    compact=['# 四页正文增补：固定代理与检查点机制证据','',
        '保留上一轮四页正文的两条问题：联合可靠性与同曝光训练迁移。完整92条件对原生思考确认与全部1,410目标题仍是能力解释的主要证据；本轮补充训练机制，不能用关闭思考结果覆盖已验收的正面推理结果。','',
        '正文建议仅增加下面六行150步端点表，50/100/150全部轨迹、每面板差值、目标控制和敏感性放附录/结果包。150为原训练终点，保留该端点不是按成绩选择最佳checkpoint。','',
        f"建议正文表述：固定训练池{len(train)}项比较中，JS使代理W1全部降低、实际W1有{sum(value(c,'delta_decoder_W1_conditional') < -1e-12 for c in train)}项降低；跨全部72项分层比较的{len(reverse)}个反向点估计，其经验范围均跨0，不能以此声称稳定的代理/解码反向效应。结果支持所测小菜单上的一致性改善，并保留来源、指标及接口的边界。",'',
        '| 模型 | 等价边分层 | JS−SFT代理W1 | JS−SFT实际生成W1 |','|---|---|---:|---:|']
    for r in summary:
        if r['step']==150:compact.append(f"| {r['model']} | {LABELS[r['stratum']]} | {fmt(r['delta_proxy_W1'])} | {fmt(r['delta_decoder_W1'])} |")
    compact+=['','四个回溯留出面板等权，每面板每层16条小菜单边、每提示64次真实生成；只有一个预定旧种子。这里的归一化W1边均值不是主基准条件对E，不能与旧总表直接拼接。',
        '', '正文应并列说明：在线代理受优化、固定验证代理、实际生成分布是不同证据层。新实验只定位整体代理/解码接口及迁移边界，未识别某一解码组件的单独因果效应。相反方向的面板必须与同向改善面板一起报告。',
        '', '旧48训练的两模型×四来源×三种子全匹配结果仍是主要方法结论；本轮48个检查点来自其中预定一组旧种子，不冒充48次新独立训练。原生思考确认继续报告目标近满分、规则互斥分层100%，并保留14B来源/任务等权E上升的聚合边界。',
        '', '全部检查点CSV及原始、独立标量审计、完整性/恢复证明由mechanism_table_provenance.json与mechanism_completion_acceptance.json指向；经验重采样范围非校准CI，目标Q含全部失败。']
    (RESULT/'四页正文增补与机制证据表.md').write_text('\n'.join(compact)+'\n',encoding='utf-8')
    save(RESULT/'summary_values.json',dict(checkpoint_stratum_means=summary,completeness_counts=counts,end150_goal_Q=end_goal,
        reverse_point_estimates=len(reverse),reverse_empirical_ranges_cross_zero=reverse_cross_zero,
        accepted_provenance_sha256=sha(RESULT/'mechanism_table_provenance.json'),writer_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
