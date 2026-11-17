# 确认补实验完成总结与摘要边界

2026-11-13，新加坡时间。本文件以完整原始回答的本地独立复算和预定统计为依据。第二轮固定280,128条科学回答、40条工程先导已全部完成；63份不可变文件逐SHA验收，四个科学阶段的交付身份与标量复算通过，四项原生采样分析完成。三个原服务均已按本轮私有快照恢复，并核对实际argv、GPU、工作目录及模型端点。最终收尾证明为 `confirmation_completion_acceptance.json`。

两轮补实验合计1,371,200条科学回答及120条先导，各批次分别冻结、分别验收，不拼成新的抽样分母。旧48次训练、52项生成及全部失败保留原样；本轮没有新增P1模型或训练。

## 当前最重要的结果

**原生推理配置提供了明确的正面路径。** 完整严格格式确认覆盖92条件对、429行为提示与全部1,410目标题。以下为条件对等权；E/G是合法输出条件分数，Q包含所有尝试。

| 模型 | 关闭→开启条件E | 关闭→开启条件G | 关闭→开启全尝试Q |
|---|---:|---:|---:|
| Qwen3-14B | 0.02332→0.00725 | 0.04309→0.00890 | 78.02%→99.99% |
| Qwen3-30B-A3B-GPTQ-Int4 | 0.21742→0.01010 | 0.27934→0.01081 | 74.74%→99.86% |

主权重下计入缺失行动的质量范围，E/G下降方向仍保持。14B的ΔE范围为[−0.016475,−0.014901]、ΔG为[−0.034603,−0.032645]；30B分别为[−0.209512,−0.184008]、[−0.271857,−0.242984]。这些范围不包含采样误差。

预定500次共享提示全尝试重采样中，14B主ΔE经验范围为[−0.017553,−0.014136]、主ΔG为[−0.036621,−0.029777]；30B分别为[−0.212491,−0.201205]、[−0.273876,−0.258624]。经验范围不是校准置信区间，固定提示上的重复采样不提供新来源保证；与上述缺失质量范围分开解释。

**稳定性改善依赖汇总权重，14B不能写成全面改善。** 14B来源等权E为0.01672→0.02025，任务等权E为0.02124→0.02385。来源等权ΔE经验范围[0.000835,0.008088]，任务等权ΔE范围[−0.001924,0.008560]；后者跨零，不能据点估计称真实分布退化。14B来源/任务等权G点估计下降，但对应经验范围也跨零。30B三种权重的E/G均下降，经验范围也同向。所有预定权重完整列出，不挑有利汇总方式。

**可求解规则对照支持“能够响应实质变化”，需要收窄能力解释。** 40个新参数条件对的全尝试Q：14B为72.11%→99.40%，30B为69.74%→99.66%。最优集合互斥的24个条件对×目标四元组引用覆盖20个参数条件对，14B/30B的该分层Q分别由47.66%/40.00%升至100%，四侧最差正确率的加权均值也均为100%。这些是在推理前冻结的参数诊断，不是新经验来源或24个独立研究；最优集合相交和保持两类均保留。

**失败类型仍须准确区分。** 原始目标题上，14B开启思考11,280次有3次截断、30B有16次截断，其余合法最终行动均正确。参数化诊断中，14B开启思考3,840次有22次截断和1次合法但目标错误；30B有13次截断，其余合法目标行动均正确。14B那1次最终选择1800，而指定极差目标的最优集合是[0,900,2700]，属于真实目标执行错误，不能归为解析问题；也不能用单次错误描述普遍无能。原始目标检查与参数目标检查不可合并成“合法回答全对”。

行为上，14B开启思考54,912次中有38次截断、1次数值类型错误；30B有857次截断、1次非法菜单。所有失败计入交付分母，不从思考内部捡答案，不把无效行动当0。逐提示单一经验输出数14B为381→349/429，30B为308→272/429；这些是描述统计，低熵无统一好坏。R变化也没有统一优劣方向。

**同曝光训练仍呈现来源和聚合上的异质性。** 两模型×四来源×三种子的全部24组JS−SFT比较已完成共享提示统计。Qwen3-8B的主权重ΔE为+0.00022，来源等权为−0.00597；Qwen2.5-14B分别为−0.00395和+0.00129。不能从微小平均差推导通用优劣。24组匹配比较中，JS臂最后25步的平均在线代理均低于SFT臂，这说明在线代理受到优化，不证明固定留出验证代理或实际解码分布统一改善。没有执行P1代理/解码对齐、系数消融或新训练，论文结论限定所测配方。

## 四页论文怎么收敛

建议正文只保留两条问题：联合可靠性如何随推理配置变化，以及同曝光一致性训练能否稳定迁移。正文最多两张主表：联合基准表，以及分面组织的原生对照/训练表；四元组设计用一个紧凑示意。参数诊断可缩为一两句定量结果，其完整表、目标分层、来源/任务/种子和错误类型放可审计结果包。完整取舍与数表见 `四页正文最终实验取舍与紧凑表.md`。

主结论应写成：INVAR用联合检查识别不同配置的表现与边界；充分使用所测原生推理接口可同时改善目标执行和主权重呈现稳定性，但稳定性汇总仍可随来源权重改变，同曝光一致性训练的迁移也不统一。论文不能继续依靠“模型普遍无法执行目标”维持负面主线。

事实高分与目标执行之间的原有差异应限定对应运行配置；事实正确不证明完整理解。G≤2E，不当作完全独立能力；R无规范优劣，自由行为没有统一合作率真值。30B为量化MoE，不作纯规模推断。独立两调用目标成功概率不是连续智能体轨迹成功率。固定题的误差分析和新参数诊断都不构成新来源或真实人类行为预测的保证。

## 可替换摘要的英文版本

**Stable to Form, Responsive to Substance: Evaluating Reliability in LLM-Based Simulation**

Reliable LLM-based social simulation requires equivalent descriptions to preserve decision distributions and substantive changes in rules or objectives to elicit correct responses. We introduce INVAR, a Chinese-language benchmark with 8,666 unique inputs built around paired scenarios across six social-decision task families. INVAR jointly evaluates presentation stability, factual performance, and explicit goal execution. Across general-purpose and socially specialized model configurations, these measures provide different evidence of reliability. Full-coverage comparisons of two native-reasoning configurations further show that inference settings materially change the assessment: with reasoning enabled, all-attempt goal accuracy exceeds 99.8% under pair weighting, while pair-weighted presentation discrepancy decreases substantially. However, one model's stability improvement reverses under source weighting. Both reasoning configurations also answer all tested disjoint-optimum rule contrasts correctly, qualifying interpretations of failures under other inference configurations. In matched-exposure fine-tuning across four held-out sources and three training seeds, adding equivalent-presentation consistency regularization to rule-supervised fine-tuning yields uneven gains. Aggregate improvements can conceal source- or objective-specific losses. These results support evaluating simulation reliability jointly, with explicit inference settings and aggregation choices, rather than inferring reliability from a single score or extrapolating configuration-specific failures to a model's capabilities.

这版摘要对应已验收结果，仍供作者取舍；没有将经验范围写成总体显著性，没有将参数确认写成新来源，也没有为保留原摘要的负面方向删去正面结果。

## 产物与停止

完整原始回答、输入、计划、模型身份和评分在 `artifacts/confirmation_20261112`；逐文件清单为 `mirror_index.json`。本地独立评分在 `local_scalar_audits/` 和 `local_verification_workspace/`，原始镜像文件不覆盖。

中途传输在最后分块遇到一次SSH退出255；已核对保留44,040,192字节与远端前缀哈希一致，并从同一原始归档续传剩余部分，完整SHA通过。故障与恢复记录为 `transport_interruption_20261113.json`、`transport_recovery_20261113.json`，没有重新生成回答或更改科学设置。

科学计算到此结束。最终报告包在D盘 `final_report_packages/`，其路径、文件数与SHA由 `final_report_package.json` 记录。全部完成检查和报告包验收通过后暂停本自动化，不再自动扩展模型、训练或采样预算。
