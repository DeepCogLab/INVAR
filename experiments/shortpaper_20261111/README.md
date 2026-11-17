# WWW short paper 定向补实验

2026-11-11冻结。研究问题、证据审查、实验优先级与四页取舍见[补实验规划与证据审查](../../results/20261111_短文补实验/补实验规划与证据审查.md)。

[当前补实验结果](../../results/20261111_短文补实验/当前补实验结果.md)：固定1,091,072条科学回答及80条先导全部完成；18个科学阶段双端独立复算、扩展身份/计划/权重延续、联合行为噪声参照均已验收，D盘310份不可变文件逐SHA通过。GPU0两次和GPU1一次服务借用均已按原配置恢复健康，全部科学队列退出。见[最终验收](../../results/20261111_短文补实验/supplement_completion_acceptance.json)与[完成总结](../../results/20261111_短文补实验/补实验完成总结与摘要边界.md)。自动检查 `invar-2` 在最终整理后暂停，旧 `invar` 保持暂停，不再启动已完成队列。

## 新实验

- Qwen3-14B与Qwen3-30B-A3B-GPTQ-Int4：各421,504条原协议全量回答。
- 两模型原文/完整行动收益表：各90,240条目标诊断。
- 同一冻结160个目标题，开/关思考、每题每臂8次：14B/30B及Social-R1-4B/8B、Qwen3-4B/8B，共15,360条原生诊断。
- 合计1,038,848条评测/诊断，另80条工程先导。

以上为初始冻结预算。因14B/30B原生思考仍有114/1280、78/1280短预算截断，已另行冻结 `native_long_freeze.json`：六模型、全部原160提示、每题8次、思考4096新token，共7,680条；不重抽失败题。新总预算1,046,528条评测/诊断，另80条先导；旧协议保持原样。

主矩阵完成后，为检验目标改善能否同时带来行为稳定性改善，新增固定15条件对的开关思考对照：四来源六任务、两严格格式、87唯一提示，14B/30B每提示每模式128次，共44,544条。活动协议 `joint_reasoning_freeze_v2.json`，运行器 `offline_joint_reasoning_v2.py`，队列 `joint_reasoning_campaign_v2.py`。v1启动前的空参考测试暴露评分断言边界，原版保持不动且未推理，修正另记 `joint_reasoning_prelaunch_revision.json`；科学设置不变。总科学预算至此1,091,072条，另80条先导，不继续扩张。

主协议为 `freeze.json`、`runtime_code.json`、`frozen/`；Social-R1匹配诊断另有 `reasoning_pair_freeze.json`。输入生成文件不包含正确标签；规范收益表不提供最优行动。所有生成输出保留，格式/截断不按答案重抽。

远端：`/workspace/project_social_simulation/invar_shortpaper_runs/20261111`。原始回答保存在远端；小型记录同步至 `remote_records/`。完整镜像请使用D盘，勿耗尽C盘。

## 运行和检查

```powershell
python experiments/shortpaper_20261111/remote_control.py status
python tools/sync_shortpaper.py
```

两条主队列并行复用现有服务，阶段为pilot→goal_diagnostic→native→main，分别独立评分。后续原生匹配队列等待主队列完成，仅临时借用已核验的GPU0服务并尝试恢复；检查 `reasoning_pair_status.json` 和 `service_restoration.json`。

长预算现使用 `native_long_campaign_serial_v2.py` 包装冻结的原 `native_long_campaign.py`；检查 `native_long_status.json` 和 `native_long_control/service_restoration.json`。原两次借用快照分开保存。它记录短/长输出的前缀一致性，不能假定上下文/调度变化后轨迹相同。旧等待队列已停止，不重复启动。

19:06验收：六款长预算已全部完成，串行恢复后的18241/18242按原argv、GPU、工作目录及模型端点再次确认健康，包装队列已退出，见 `native_long_restore_acceptance.json`。勿再启动任何原生短/长预算队列。

16:12续作：原匹配队列完成后并行恢复两个服务，vLLM共享GPU内存剖析相互干扰，一项恢复失败。保留原日志及冻结脚本，新增 `service_restore_serial_v2.py` 和 `native_long_campaign_serial_v2.py`，按原argv/GPU/工作目录逐项启动并等待健康；只替换仍在等待且未生成任何长预算回答的队列。两原服务健康已核验，长预算已开始运行；后续恢复也使用串行包装。原始科学协议、输入、采样、评分和回答不改，见 `native_long_serial_fix_v2.json`、`native_long_waiter_replaced_v2.json` 及恢复事件日志。勿启动旧等待队列。

`joint_reasoning_campaign_v2.py` 在GPU1并行运行，仅借用已核验的18240服务。检查 `joint_reasoning_status.json`、`joint_reasoning_control/`日志及独立恢复记录。评分报告可解析子集条件指标与全部尝试的缺失质量上下界；后者不是抽样置信区间。不得以条件分数冒充全部尝试结果，也不把目标提升推为E/G提升。该队列结束仍须核对原18240服务按实际快照恢复健康。

行为阶段主评分完成后，分别运行 `analyze_joint_noise.py --job qwen3_14b` 与 `--job qwen3_30b_a3b_gptq`，另存199次池化置换的条件E噪声参照。它只描述有限采样的参照大小，不是p值、无偏校正或缺失选择保证，也不修改主评分。两个队列不会自动执行这一附加分析，续作验收时运行一次并镜像。

完成阶段完整镜像与本地原始回答复算：`python tools/mirror_shortpaper_completed.py`。目标 `artifacts/shortpaper_20261111`，验收为本目录 `local_mirror_acceptance.json`；先确认没有同一同步进程在运行。大文件scp曾断流，该工具使用可续传的分块SSH和逐原字节SHA；旧部分传输保留。本地复算放单独工作区与 `local_scalar_audits/`，远端原始审计记录保持原字节。

不要重新上传不同字节覆盖冻结协议或运行器。运行故障应保存原始错误、定位原因并另建有版本的修复记录；不删题、不按分数换样本、不用部分结果作为最终排名。

## 本次新增的无GPU分析

- `analyze_existing.py`：26配置清单、配对事实/目标、互斥目标区分、噪声参照。
- `audit_structure.py`：2,804行为引用再渲染核验、纯格式子集、条件效应方向敏感性。
- `transfer_slices.py`：576行V4分层与192组JS−SFT匹配比较。
- `score_api.py --stage old_native`：保守解析完整思考段后的最终答案，另存旧原生诊断修正。
- `verify_goal_diagnostic.py --job 模型ID`：在远端按原始回答独立验算收益表两臂及逐政策Q；新14B阶段完成后同样需要运行。
- `audit_new_runs.py --job 模型ID --stage main或native`：独立核对计划、身份、种子、消息哈希、原生最终答案与主指标。
- `summarize_supplement.py`、`write_current_status.py`：只汇总已独立验收的新阶段，保留旧状态文本快照。
- `analyze_native_delivery.py`：从D盘已验收原始回答区分截断、非法行动、类型/标签及合法行动下的目标错误，逐条保留失败目录；API n=8输出token按请求去重。
- `analyze_joint_slices.py`：从D盘已验收四元组评分独立核对条件对优先的全部来源/任务聚合，并保留所有切片及缺失质量范围；不把切片或范围当抽样显著性结果。
- `audit_extension_identity.py --job 模型ID --stage native_long或joint_reasoning`：完成后在远端新增身份审计，独立比对冻结协议、全部计划/种子/消息、已验收短预算权重及模型配置、token预算和前缀标志。完整镜像工具自动在D盘另存 `local_identity_audits/`，不覆盖远端原字节。
- `write_paper_tables.py`：生成[四页正文取舍与证据表](../../results/20261111_短文补实验/四页正文实验取舍与证据表.md)，含输入CSV的哈希来源；未完成研究不填结果。
- `accept_supplement.py`：核对冻结18个科学阶段的总数、逐阶段原始SHA及本地审计、完整镜像、两项联合噪声复算和三次恢复记录，并实时检查三个原服务的argv、工作目录、GPU及模型端点。最终验收只新增一次，不覆盖既有记录。

旧的基线、方法轮次、原始回答、失败记录和 `sources/` 均保留。研究结论以完整结果和明确的适用范围为准。
