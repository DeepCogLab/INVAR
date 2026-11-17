# 新增实验的写作结果入口

优先阅读 [CURRENT_RESULTS_CORRECTED.md](CURRENT_RESULTS_CORRECTED.md) 与 [summary_corrected.csv](summary_corrected.csv)，它们只汇入已通过独立原始回答复算的检查点。

当前统一采用v3：修正空白行重复计JSON起点，并包含GPT-OSS的Harmony v2元数据修正。原表 `summary.csv`、`CURRENT_RESULTS.md` 是保留的冻结代码输出，**其中GPT-OSS、Llama、GLM、Mistral及Base受影响的数值不应引用**；全部八项已按同一v3规则完整验收。Qwen3、SocialR1和SOTOPIA解析结果未变，仍统一从更正表引用。原因和版本链见 [v3评分更正](../SCORING_ERRATUM_TERMINAL_V3.md)、[Harmony更正](../SCORING_ERRATUM_HARMONY_V2.md) 及 `SCORING_VERSION_SELECTION.json`。不能把解析器错误造成的差异解释成模型能力或交付失败。

主八项已经完整验收，见 [主发现记录](MAIN_FINDINGS.md)、`MAIN_MATRIX_ACCEPTANCE.json` 和 `main_completion_matrix.csv`；32单元的控制题/交付/成本表见 `paper_controls_delivery_cost.csv`，生成后任务/目标分项见 `goal_task_policy_breakdown.csv`。首个Qwen检查点的旧 [分项记录](FIRST_MODEL_NOTE.md) 未受影响。

夜间追加实验另行冻结、已完整验收，见 `../supplement_overnight/PROTOCOL.md` 及该目录的结果。最终跨阶段结论以 `../FINAL_FINDINGS.md` 为准；主矩阵完成不代表补充已完成。

最终入口：[FINAL_FINDINGS](../FINAL_FINDINGS.md)、FINAL_CAMPAIGN_ACCEPTANCE.json、FINAL_WRITING_PROVENANCE.json。

补充写作表：paper_supplement_language.csv、paper_supplement_language_interactions.csv、paper_supplement_budget.csv、paper_supplement_budget_contrasts.csv、paper_supplement_completion.csv。表中空白表示该诊断不适用或无定义，不能填0；阶段与干预分别解释。原补充目录summary.csv和各job独立收据是这些写作副本的依据。
