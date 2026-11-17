# 只读交付审计性能修订

2026-11-13 19:25（新加坡时间）定位：冻结`audit_delivery.py`在每个输出token上调用`len(tokenizer)`，本环境该调用会重复计算含新增token的词表大小。Qwen3实测100次用时5.92464秒，均为151669；因此全交付审计会异常缓慢。

科学冻结保持SHA `3fa77d5c406c43cd12d01742903f9b6a79dad8c47cd6d5bea3cd25924fca23a5`，原审计代码不改。新增`audit_delivery_cached.py`只将同一不可变tokenizer的词表大小缓存一次，保留逐token类型及相同上下界检查；不改评分、回答、阈值、预算或题目。

新增`repair_postprocessing.py` PID1693525。它按模型等待原队列完成全部生成并进入交付审计，核对真实审计子进程argv后仅向该子进程发送SIGTERM。原控制器保留failed记录并执行原串行恢复；确认其退出且按该轮快照恢复实际服务后，运行新增缓存审计器和原冻结score.py。另一模型尚在生成时保持其计算不中断。没有重新生成或续拼任何科学回答。

后处理权威状态为`postprocessing_recovery_JOB.json`，原`status_JOB.json`异常不覆盖。新增`verify_restoration_recovery.py`在后处理完整评分通过后验证原控制器已退出及实际argv/GPU/cwd/模型端点，另存同名公开恢复验收记录。严禁重复启动原campaign或repair进程。

新增代码SHA与性能证据在`audit_performance_addendum.json`，各模型中止的准确PID、原审计日志SHA和生成/前向COMPLETE原SHA在`audit_performance_incident_JOB.json`。代码、身份审计、恢复和原始数据全部纳入D盘逐SHA镜像。完整原始本地复算和另写标量核验仍必须通过后才报告科学结论。
