# 整理后的写作数表

这些表只做已验收数字的提取、连接或四面板算术平均，不重新抽样，不改原分数。生成器：[build_writing_context.py](../../tools/build_writing_context.py)。标准库即可运行：在项目根目录执行`python tools/build_writing_context.py`，只更新本目录。

| 文件 | 行数 | 单位及用途 |
|---|---:|---|
| [all_model_configurations.csv](all_model_configurations.csv) | 28 | 八基线+18社会扩展+两新增主配置；原结构化接口、92对/701四元组、每配置421,504条；含E/G/R、两类事实U、Q及原始SHA |
| [native_confirmation.csv](native_confirmation.csv) | 12 | 两模型×思考开关×三权重；完整canonical确认；E/G/R为合法输出条件量，Q为全部尝试 |
| [parameter_confirmation.csv](parameter_confirmation.csv) | 12 | 同两模型×开关×三权重的40参数条件对；不是原8,666输入的扩充主表 |
| [training_summary.csv](training_summary.csv) | 12 | 两模型×baseline/SFT/JS×条件对/来源等权；三种子均值。每种子分数由四个来源留出适配器拼合 |
| [mechanism_checkpoint_means.csv](mechanism_checkpoint_means.csv) | 18 | 两模型×三步数×三层；各四来源面板等权的JS−SFT点差；原经验范围仍在完整72项表 |

`provenance.json`逐一列出86个读取文件的SHA与全部输出SHA。28主配置不与原生确认配置拼为40款模型；原训练三种子24组比较不与机制三检查点24组比较相加。CSV数值保留原精度，论文展示时才舍入。

2026-11-16新增独立表不覆盖旧表：见[主统一v3及补充写作入口](../../experiments/robustness_20261115/results/README.md)。主32格、补充语言16格及预算16格分表；来源SHA见该目录FINAL_WRITING_PROVENANCE.json。
