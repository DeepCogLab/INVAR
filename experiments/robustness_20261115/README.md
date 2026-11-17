# 双语与跨模型稳健性实验：最终入口

先读[FINAL_FINDINGS.md](FINAL_FINDINGS.md)，再查[写作表和评分权威入口](results/README.md)。主94,080与事后补充21,888各自完整验收，合计115,968，不重复计GPT工程恢复。

| 目录/文件 | 用途 |
|---|---|
| PROTOCOL.md、freeze.json、frozen/ | 不改的主科学方案与输入 |
| results/ | 主统一v3表、写作副本、矩阵及最终验收 |
| supplement_overnight/ | 单独事后协议/冻结/六任务结果；interim文件仅历史快照 |
| gpt_date_restore_20261116/ | GPT零答案前日期恢复的工程偏离、独立证明与新收据 |
| SCORING_ERRATUM_*、independent_audits_v3/ | CPU评分修订与独立原始验收链 |
| monitor_snapshots/、latest_sync.json | 运行/同步审计；旧PID不代表当前运行 |
| accept_main_matrix.py、accept_supplement_matrix.py、accept_final_campaign.py | 只读验收入口，不启动GPU |
| build_final_context.py、finalize_context_index.py | 从已验收表生成和整理写作上下文 |

完整raw/token/identity/adapter/log在artifacts/robustness_20261115/remote，统一主v3评分在corrected_v3。sources只读，旧实验冻结。原v1/v2和所有失败保留，不能用旧错误表写能力结论。内部研究包未自动公开发布。
