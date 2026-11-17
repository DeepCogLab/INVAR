"""Build compact, traceable evidence tables without claiming unfinished results."""
import csv
import hashlib
import json
from pathlib import Path
from analyze_existing import OUT


ROOT=Path(__file__).resolve().parents[2]


def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))


def main():
    paths=[OUT/'new_model_main_summary.csv',OUT/'new_native_summary.csv',OUT/'native_delivery_and_token_use.csv',ROOT/'results/20261029_全部实验总结/三种子完整汇总.csv']
    main_rows,native,delivery,training=map(read,paths)
    joint_path=OUT/'new_joint_reasoning_summary.csv'
    joint=read(joint_path) if joint_path.exists() else []
    closed=(OUT/'supplement_completion_acceptance.json').exists()
    if joint_path.exists():paths.append(joint_path)
    slices_path=OUT/'joint_reasoning_all_slices.csv'
    if slices_path.exists():paths.append(slices_path)
    lines=['# 四页正文：实验取舍与证据表','',
        ('这是全部补实验验收后的正文取舍建议，不是已排版的完整论文。固定科学预算1,091,072条及80条工程先导已完成，原始回答双端复算、镜像和三次服务恢复已验收。' if closed else '这是当前可复核证据的排布建议，不是已完成论文。尚未验收的长预算配置及自由行为思考对照，不预写其结果或摘要结论。主表数字只来自已独立验收记录。'),'',
        '## 建议只保留两条主问题','',
        '1. 等价呈现稳定性与事实/目标执行能否相互代替？以配对设计、声明接口下的主表，以及同样提示的原生推理对照回答。原生诊断用于校正能力解释，不只挑关闭思考的失败。',
        '2. 在训练曝光相同的条件下，直接优化表达一致性能否稳定迁移？使用全部两模型、四来源、三种子，保留来源权重和失败方向。', '',
        '四页内优先保留：一幅四元组示意、一张按家族/规模选择代表的联合指标表、一张同曝光训练表，以及一组最有信息的接口对照。完整28配置、收益表目标分解、全部来源/种子和失败明细放可审计结果包。收益表是特权规则展开诊断，若空间紧张只用一句定量例子，不新增第三条方法主线。', '',
        '## 已验收新增主评测（供并入模型表）','',
        '| 配置 | E↓ | G↓ | 变化事实U | 未变事实U | 目标Q |','|---|---:|---:|---:|---:|---:|']
    for r in main_rows:
        if r['subset']=='all':lines.append(f"| {r['job']} | {float(r['E']):.4f} | {float(r['G']):.4f} | {float(r['changed_fact_U']):.1%} | {float(r['stable_fact_U']):.1%} | {float(r['goal_Q']):.1%} |")
    lines+=['', '每配置421,504次回答，92条件对；目标仅36适用对、五任务。条件对等权；Qwen3-14B与30B量化MoE不能作为纯规模效应。原有26配置从既有完整清单按家族/匹配底座选择，不能按本轮成绩挑代表。不同解码/预算诊断不混入这张主表。', '',
        '## 同曝光训练主表：全部三种子平均','',
        '| 模型 | 方法 | E↓ | G↓ | Q |','|---|---|---:|---:|---:|']
    labels={'baseline':'原模型','anchor_sft':'SFT','anchor_js':'SFT+JS'}
    for model in ['qwen3_8b','qwen25_14b_instruct']:
        for method in labels:
            r=next(r for r in training if r['model']==model and r['method']==method and r['aggregation']=='pair_weighted')
            lines.append(f"| {model} | {labels[method]} | {float(r['behavior:E']):.4f} | {float(r['behavior:G']):.4f} | {float(r['goal:accuracy']):.1%} |")
    lines+=['', '每个模型×种子行由四个来源留出适配器拼合，不是单一适配器的全92对得分。原模型只采样一份，不冒充三个模型重复；V4参照与旧基线采样不同，不能交叉相减当训练效果。', '',
        '| JS相对同曝光SFT | 条件对等权ΔE | 四来源等权ΔE | 条件对等权ΔG | 四来源等权ΔG |','|---|---:|---:|---:|---:|']
    for model in ['qwen3_8b','qwen25_14b_instruct']:
        delta={}
        for agg in ['pair_weighted','source_macro']:
            a=next(r for r in training if r['model']==model and r['aggregation']==agg and r['method']=='anchor_sft')
            b=next(r for r in training if r['model']==model and r['aggregation']==agg and r['method']=='anchor_js')
            for metric in ['E','G']:delta[agg,metric]=float(b['behavior:'+metric])-float(a['behavior:'+metric])
        lines.append(f"| {model} | {delta['pair_weighted','E']:+.5f} | {delta['source_macro','E']:+.5f} | {delta['pair_weighted','G']:+.5f} | {delta['source_macro','G']:+.5f} |")
    lines+=['', '权重改变后的方向变化是收益分布不均的证据，不选有利权重替代主结果。四来源不是来源总体随机样本，三个训练种子也不提供新来源保证；仅凭微小平均差不宣称显著或非劣。', '',
        '## 原生思考诊断：同160题，每题每模式8次','',
        '| 配置 | 关闭思考Q | 开启思考Q | 截断 | 非法行动 | 其他接口失败 | 思考平均输出token |','|---|---:|---:|---:|---:|---:|---:|']
    for job in sorted({r['job'] for r in native if r['budget']=='1408'}):
        a=next(r for r in native if r['job']==job and r['budget']=='1408' and r['arm']=='native_no_thinking')
        b=next(r for r in native if r['job']==job and r['budget']=='1408' and r['arm']=='native_thinking')
        t=next(r for r in delivery if r['job']==job and r['stage']=='native' and r['arm']=='native_thinking')
        illegal=int(t.get('menu') or 0);other=int(b['n'])-int(b['valid'])-int(b['truncated'])-illegal
        lines.append(f"| {job} | {float(a['Q_all_attempts']):.1%} | {float(b['Q_all_attempts']):.1%} | {b['truncated']}/1280 | {illegal}/1280 | {other}/1280 | {float(t['mean_output_tokens']):.1f} |")
    lines+=['', '均为全部尝试分母。仅通过类型及菜单检查的子集全部正确，不等于所有完整JSON都正确：Social-R1-8B有62条选择不在七档菜单中的1200（11个提示、同一分配情境的表达/目标重复），属于真实行动约束遗漏，不能叫普通格式问题，也不能当62个独立情境。Social-R1-4B的23条缺少思考闭合标签、Qwen3-8B的4条把整数写成字符串，分别保留为接口失败。', '',
        'Social-R1在本固定短预算样本比同规模底座更常交付合法正确答案，且平均输出更短；不能据此证明其内部推理更准确或整体社会模拟更真实。token统计包含截断，14B/30B API按唯一请求用量去重，其他按每条token ID计数；不当作延迟或算力成本。', '']
    long_rows=[r for r in native if r['budget']=='4096']
    if long_rows:
        lines+=['## 长预算思考：已验收配置','', '| 配置 | 全尝试Q | 截断 | 非法行动 | 其他失败 | 原短预算完整文本为新输出前缀 |', '|---|---:|---:|---:|---:|---:|']
        for r in long_rows:
            t=next(t for t in delivery if t['job']==r['job'] and t['stage']=='native_long')
            illegal=int(t.get('menu') or 0)
            lines.append(f"| {r['job']} | {float(r['Q_all_attempts']):.2%} | {r['truncated']}/{r['n']} | {illegal}/{r['n']} | {int(r['n'])-int(r['correct'])-int(r['truncated'])-illegal}/{r['n']} | {r.get('matching_prefix','待核验')}/{r['n']} |")
        lines+=['', '4B长预算配置的Qwen3/Social-R1点估计顺序与短预算不同。不能外推统一专门模型优势，也不将Social-R1的8条少成功归因为增加预算有害。两款已完成4B运行都没有原截断文本被原样延续的记录；上下文和调度同时变化，故不能写“把相同失败推理延长至4096就成功了”，也不能仅凭种子相同作轨迹级因果解释。', '']
        if len(long_rows)==6:
            lines+=['六款长预算均已验收。14B/30B在固定160题达99.92%/99.45%，说明原关闭思考表现不能被外推为能力上限。Social-R1-8B仍有72次菜单外行动（70次1200、2次2400；13提示、同一分配情境的目标/表达重复），另外12截断、1格式失败；这支持保留“合法行动约束”作为独立检查，不是证明所有模型都不懂目标。', '']
    selected=[r for r in joint if r['weighting']=='pair_equal']
    if selected:
        def num(x):return f'{float(x):.4f}' if x not in [None,''] else '—'
        lines+=['## 行为思考开关：已验收配置，固定15条件对','', '| 配置 | 模式 | 有效率 | 条件E | 全尝试E范围 | 条件G | 全尝试G范围 |','|---|---|---:|---:|---|---:|---|']
        for r in selected:
            lines.append(f"| {r['job']} | {r['arm']} | {float(r['parse_rate']):.2%} | {num(r['E_conditional'])} | [{num(r['E_low'])}, {num(r['E_high'])}] | {num(r['G_conditional'])} | [{num(r['G_low'])}, {num(r['G_high'])}] |")
        lines+=['', '来源/任务等权和R见CSV；R无统一优劣。范围针对缺失合法输出的可能影响，不能称抽样CI；条件点估计不等于全尝试指标。仅显示已完成模型，不以先完成的模型替代整个固定两模型实验。', '']
        lines+=['14B预设条件对等权下E/G同时降低，不能继续笼统写“开启思考不能改善稳定性”。但四来源等权E由0.01286变为0.01588；完整来源切片显示PIQ572下降、HPVBG和IGVOFO上升。须并列保留这些方向，不能只选主均值或只选退化来源。这里的点估计和缺失范围尚不构成有限采样下真实分布改善的显著性证明；目标诊断与行为诊断也属于分别冻结的子集。', '']
        if any(r['job']=='qwen3_30b_a3b_gptq' for r in selected):
            lines+=['30B的条件对等权E/G由0.26191/0.21341降至0.01859/0.01782，来源与任务等权也同向下降；31条未合法交付中29条截断，计入缺失质量后下降方向仍保持。这是必须保留的正面直接对照，不能只呈现关闭思考下的失败。它支持声明接口和预算的重要性，不构成92对全量、新来源或人类行为预测准确性的保证。', '']
    lines+=[('## 完整证据怎样调整摘要' if closed else '## 剩余验收怎样影响摘要'),'',
        '- 六模型固定4,096-token目标扩展已全部完成并复核：保留原160题全部尝试，残余截断、非法行动及类型/标签问题分别列出。不作同轨迹延长或纯预算因果解释。',
        '- 15条件对的自由行为思考对照：直接检验开启思考是否也改善E/G。条件指标及完整尝试缺失质量范围分别报告，并列描述性采样噪声参照。不能将目标提高自动推为行为稳定性提高。',
        ('- 本轮两模型主权重的行为E/G均下降，目标诊断也改善，应明确承认所测设置中的有效路径。14B来源等权E例外；不把两种诊断的不同子集合并成全量保证。E/G不独立（G≤2E），R无统一优劣，自由行为没有合作率真值。' if closed else '- 若两者联合改善，应明确承认所测设置存在有效路径；若指标方向或缺失范围不足以判定，结论写未定。E/G不独立（G≤2E），R无统一优劣，自由行为没有合作率真值。'), '',
        '## 摘要现在可以保留与必须收窄的内容','',
        '可保留：联合评估必要；主表、目标分解和同曝光迁移表揭示不同指标/子集的得失；不能从一个汇总量推断全面可靠。必须收窄：不能把关闭思考下的目标失败写成模型普遍缺乏目标能力，不能称所有专门模型不如底座，不能把一款量化MoE的表现归因于参数规模，也不能把原收益表诊断包装成通用新算法。', '',
        ('直接行为对照现已完成：两模型在主权重下E/G点估计均下降，30B在三种权重下同向，14B的来源等权E例外。摘要应承认所测设置中的有效改善，同时保留同曝光训练的迁移异质性和联合检查的必要性；不得写“开启思考仍无法改善稳定性”。完整数字及验收见[当前结果](当前补实验结果.md)。' if closed else '当前不提前写“开启思考仍不能修复呈现稳定性”。此句必须由尚在运行的直接行为对照决定。最新进度与数字见[当前结果](当前补实验结果.md)。'),'']
    (OUT/'四页正文实验取舍与证据表.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'paper_table_provenance.json').write_text(json.dumps(dict(status='derived_from_accepted_tables',inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2),encoding='utf-8')
    print('Wrote compact paper evidence tables with source hashes')


if __name__=='__main__':main()
