"""Create final compact paper tables only after all four raw-answer stages pass locally."""
import csv
from pathlib import Path
from common import R, load, save, sha

ROOT = R.parents[1]
OUT = ROOT/'results/20261112_确认补实验'
MIRROR = Path('artifacts/confirmation_20261112')


def main():
    cfg = load(R/'freeze.json')
    receipt = load(R/'local_mirror_acceptance.json')
    assert receipt['scientific_rows'] == cfg['scientific_rows']
    assert len(receipt['accepted']) == 4
    provenance = {}
    audits = {}
    for item in receipt['accepted']:
        job, stage = item['job'], item['stage']
        path = MIRROR/'runs'/job/stage/'SCORING_COMPLETE.json'
        data = load(path)
        assert data['raw_sha256'] == item['raw_sha256']
        local = load(MIRROR/'local_scalar_audits'/(job+'_'+stage+'.json'))
        assert data['raw_sha256'] == local['raw_sha256']
        stats = OUT/'native_statistics'/job/stage/'COMPLETE.json'
        assert load(stats)['source_raw_sha256'] == item['raw_sha256']
        audits[job,stage] = data
        provenance[str(path)] = sha(path)
        provenance[str(stats)] = sha(stats)
    labels = {'qwen3_14b':'Qwen3-14B', 'qwen3_30b_a3b_gptq':'Qwen3-30B-A3B-GPTQ-Int4'}
    arms = {'native_no_thinking':'关闭', 'native_thinking':'开启'}
    lines = ['# 四页正文：最终实验取舍与紧凑表', '',
             '本文件整合两轮补实验及旧同曝光训练证据。第二轮280,128条科学回答均已从完整原始独立复算，四项原生采样分析已完成；先导40条单列。旧48次训练、52项生成与第一轮补测保留原样。', '',
             '## 建议保留两条问题', '',
             '第一条：在声明的推理接口与预算下，等价呈现稳定性、事实表现和明确目标执行是否共同可靠？正文用联合指标基准表，并用完整92条件对、全部1,410目标题和可求解规则切换对照校正能力解释。', '',
             '第二条：相同训练曝光下，增加等价呈现一致性损失能否稳定迁移？正文用两模型×四来源×三种子的全部匹配实验，保留权重敏感性。结论限定所测配方，不外推一致性正则类别。', '',
             '建议篇幅：约0.7页引言与相关工作、1.0页设计与指标、1.6页结果、0.7页讨论与结论，具体仍取决于最终模板和参考文献占页规则。这里没有核实会务版式要求，不将这个分配当官方规定。', '',
             '## 原生思考完整严格格式对照', '',
             '| 模型 | 思考 | 条件E↓ | 条件G↓ | 全尝试Q↑ | 行为合法率 |',
             '|---|---|---:|---:|---:|---:|']
    for job in cfg['models']:
        for summary in audits[job,'canonical']['summaries']:
            b=summary['behavior']['pair_equal'];g=summary['goal']['pair_equal'];c=summary['counts']['behavior']
            lines.append(f"| {labels[job]} | {arms[summary['arm']]} | {b['E_conditional']:.4f} | {b['G_conditional']:.4f} | {g['Q']:.2%} | {c['valid']/c['n']:.2%} |")
    lines += ['', '每模型132,384次尝试：429唯一行为提示×每模式128次，加全部1,410目标题×每模式8次。原表达端分别与两种严格格式变体比较，共184行为四元组，覆盖92条件对；目标有840四元组、36适用条件对、五任务。E/G是合法输出条件分数；Q以全部尝试为分母。均以条件对等权为主。两模式同为4,096新token、8,192上下文、温度0.7、top-p0.95。', '',
              '本表不与旧主基准或15条件对子集拼接。30B为量化MoE，不能作纯规模效应解释；目标Q近满分不证明自由行为预测真实。完整来源/任务等权、缺失范围和采样分析见已验收结果表。', '',
              '## 参数化规则切换：全尝试Q', '',
              '| 模型 | 思考 | 全部 | 最优集合互斥 | 相交但改变 | 相同 |',
              '|---|---|---:|---:|---:|---:|']
    for job in cfg['models']:
        for summary in audits[job,'parameter']['summaries']:
            g=summary['goal'];t=g['by_transition']
            lines.append(f"| {labels[job]} | {arms[summary['arm']]} | {g['pair_equal']['Q']:.2%} | {t['disjoint_optimum_sets']['Q']:.2%} | {t['changed_but_overlapping']['Q']:.2%} | {t['same_optimum_set']['Q']:.2%} |")
    lines += ['', '五种可求解机制、40参数条件对、80情境、480目标提示，每模式每提示8次。120个条件对×目标四元组中24互斥、12相交、84相同；互斥分层覆盖20参数条件对。所有参数在推理前冻结，保留并列最优与合法菜单；这些新参数不是新经验来源，原INVAR输入数仍为8,666。此表正文空间不足时可缩成互斥分层一句，完整表放结果包。', '',
              '## 同曝光训练：三种子均值', '',
              '| 模型 | 方法 | E↓ | G↓ | Q↑ |', '|---|---|---:|---:|---:|']
    training_path=ROOT/'results/20261029_全部实验总结/三种子完整汇总.csv'
    old_provenance=load(ROOT/'results/20261111_短文补实验/paper_table_provenance.json')['inputs']
    key=str(training_path.relative_to(ROOT))
    assert sha(training_path)==old_provenance[key]
    provenance[str(training_path)]=sha(training_path)
    with training_path.open(encoding='utf-8-sig',newline='') as stream:
        training=list(csv.DictReader(stream))
    for model in ['qwen3_8b','qwen25_14b_instruct']:
        for method,label in [('baseline','原模型'),('anchor_sft','SFT'),('anchor_js','SFT+JS')]:
            row=next(x for x in training if x['model']==model and x['method']==method and x['aggregation']=='pair_weighted')
            lines.append(f"| {model} | {label} | {float(row['behavior:E']):.4f} | {float(row['behavior:G']):.4f} | {float(row['goal:accuracy']):.2%} |")
    lines += ['', '每个模型×种子的全量分数由四个留出来源适配器拼合；不是单个适配器的全92条件对结果。原模型只有一份共享采样，不当作三个独立基线。SFT和SFT+JS曝光、训练步数及适配器配方匹配；这些来源已被观察，轮流留出是回溯迁移。', '',
              '| JS−SFT | 条件对等权ΔE | 来源等权ΔE | 条件对等权ΔG | 来源等权ΔG |',
              '|---|---:|---:|---:|---:|']
    for model in ['qwen3_8b','qwen25_14b_instruct']:
        delta={}
        for weighting in ['pair_weighted','source_macro']:
            a=next(x for x in training if x['model']==model and x['method']=='anchor_sft' and x['aggregation']==weighting)
            b=next(x for x in training if x['model']==model and x['method']=='anchor_js' and x['aggregation']==weighting)
            for metric in ['E','G']:
                delta[weighting,metric]=float(b['behavior:'+metric])-float(a['behavior:'+metric])
        lines.append(f"| {model} | {delta['pair_weighted','E']:+.5f} | {delta['source_macro','E']:+.5f} | {delta['pair_weighted','G']:+.5f} | {delta['source_macro','G']:+.5f} |")
    lines += ['', '正文可用这两行权重差值代替长篇种子列表，但全部24组匹配统计与90项摘要保留在statistics_v2。经验重采样非校准置信区间；Q点态界依赖有界独立抽样，不作多重比较或新来源保证。训练日志显示JS在线代理下降，只能说明代理受到优化，不能解释为真实生成分布或固定留出验证代理已统一改善。', '',
              '## 其余结果怎样安放', '',
              '联合指标主基准仍需保留代表配置和事实U，用于回答事实表现是否足以代表目标执行。代表模型按家族与匹配底座选择，完整28配置保留，不按成绩挑选；第一轮主表与原生严格格式表分开。', '',
              '六模型短/长预算、Social-R1匹配底座、收益表目标分解及15条件对原生对照均作为解释与审计材料。收益表是特权规则展开诊断，不是通用端到端算法。Social-R1在一种预算的优势不推为全族优势。', '',
              '失败须按截断、类型/标签、菜单非法与合法行动下目标错误拆分。所有尝试保留，不从思考内部捡答案；事实正确不证明完整理解。R没有统一优劣，G≤2E，独立双调用乘积不是连续轨迹成功率。', '',
              '摘要应基于完整原生确认承认所测推理配置的正面结果，同时保留模型/任务/来源与训练配方的边界；不能追求预设负面，也不能把固定题上的正面结果外推到所有社会模拟。', '']
    (OUT/'四页正文最终实验取舍与紧凑表.md').write_text('\n'.join(lines),encoding='utf-8')
    save(OUT/'final_paper_table_provenance.json',dict(status='derived_from_all_four_locally_accepted_stages',
         inputs=provenance,writer_sha256=sha(Path(__file__))))
    print('Wrote final compact paper evidence tables')


if __name__ == '__main__':
    main()
