# 激活与校准：固定超参数的二候选代理筛选

这是方法筛选，**不是修订版 V2 的生成复跑**。只取合法输出菜单恰好两项的 behavior、changed_fact、stable_fact、goal；角色与大整数菜单不进入。每个候选按原生聊天模板后的完整紧凑 JSON 序列求条件对数似然，累加全部 JSON token，不做长度归一、不加 EOS，然后在两候选间 T=1 softmax。没有温度0.7、top-p、xgrammar，也没有生成试次。输出中的 `sampled_trials=0` 是明确说明，绝不能改写为采样频次。

## 方法

运行版修订 v2：事实问题的 A/B 选项顺序是具有行为意义的答案绑定信息。训练校准、等价差分、条件保护差分三处均排除两端 `option_semantic_order` 不同的事实边，防止把答案字母翻转学成应删除的格式方向；只用训练分组元数据，不用正确答案。各折输出去重后的排除数量。推理仍只读 messages 和合法菜单。v1 旧结果保留，不能与 v2 混用缓存或参数。

- `baseline`：同一 HF 接口、同一层位挂钩但不修改激活的匹配基线。
- `shared_format_calibration`：训练组 AB/CD 等价边的 log 概率差估计模板偏移，按完全相同菜单和 task_type 分组，最小二乘拟合样式图上的共享偏移。推理的样式识别**只读 messages**：三个模块顺序、表格/清单/自然段、匿名编号字母前缀；不读 variant_id、pair_id、case_id、condition 或正确答案。未知样式/菜单零偏移并报告覆盖。固定减去 0.5 倍偏移；c0/c1 不分别归一化。
- `paired_svd`：训练 AB/CD 等价差分的未中心化随机 SVD，秩最多4；在一个 token、一个中层块输出处以强度0.5抑制。中心为训练表示均值。
- `condition_protected_svd`：先从训练 AC/BD 条件差分提取最多8维保护空间，再把 AB/CD 差分投到该空间正交补，之后做秩最多4的 SVD；同样强度0.5。这只保护有限训练方向，不保证全部社会机制被保留。

所有特征位于同一 prompt 两个完整候选 token 序列的最长公共前缀最后 token，也就是两候选开始分歧之前。该位置可以包括 JSON 字段前缀，但不包含任何候选特有答案 token。对因果模型，未来候选不能影响此表示；实现会检查两个候选得到的表示是否一致。中层固定 Qwen3-8B 第18块（36层）、Qwen2.5-14B 第24块（48层），1-based；不做层位、秩、强度搜索，不读留出分数选参。

## 来源隔离与范围

默认构建4来源留一验证；先在**全部**四元组引用上合并共享 prompt 的条件对，再排除触及留出来源的整个闭包组。测试只使用该来源的输入；拟合只用其余来源且不共享输入的训练提示。输入候选由公开 output_schema 构造；`expected_answer/expected_actions` 仅在最后评分时使用，不进入 renderer、特征提取、校准或投影。

`--screening-only --split-manifest ...` 支持主实验选定的小规模清单，读取顶层 `selected_prompt_ids`，或回退到 `selected_pairs/selected_pair_ids`。只缩小支持菜单的输入范围；仍独立构造相同来源LOO闭包。无此参数时全量支持输入。`activation_split_manifest.json` 明确存储实际训练/测试 IDs、闭包组及上游清单哈希。

原全部基线已被研究者观察，这些新方法在现有题目上的迁移比较仍是回溯实验，不能称为未见确认性检验。

## 输出

`activation_proxy/MODEL/` 存储 identity、覆盖、split 清单、baseline 特征缓存、4折参数、逐提示两候选概率及逐四元组代理分数。`sequence_log_scores` 是模型完整候选序列分数；校准后的决策分数另列 `method_log_scores`。逐提示的 `expected_accuracy_probability` 是正确候选的代理概率质量，另报 argmax 是否正确，不是实测采样正确率。

行为代理 E/R/G 按原固定尺度定义；事实/目标报 U/Q 代理概率与 argmax accuracy。所有汇总先条件对内对变体等权，再条件对等权。仅能支持完整二项菜单四元组；缺失菜单覆盖数量单列，不与全量 V2 的总体分数直接作差。确定性代理不构造采样区间、不报告虚构重抽样。

## 运行

CPU 验证和输入覆盖审计（不导入 torch、不加载模型）：

```text
python -m unittest discover -s experiments/methods_v1 -p test_activation_probe.py -v
python experiments/methods_v1/activation_probe.py --model qwen3_8b --audit-only
```

服务器在项目根目录运行（每条命令单独占一块 GPU，由调度者执行）：

```text
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python experiments/methods_v1/activation_probe.py --model qwen3_8b --device cuda:0 --batch-size 8
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python experiments/methods_v1/activation_probe.py --model qwen25_14b_instruct --device cuda:0 --batch-size 4
```

有主筛选清单时加 `--screening-only --split-manifest experiments/methods_v1/split_manifest.json`。本地/远端目录不同可用 `--base`、`--weights`、`--output` 显式指定。权重默认沿用基线的 `INVAR_dataset/run_vllm_batch.py` 中 MODEL_PATHS；聊天模板直接导入同文件的 prompt_text，Qwen3禁用thinking。不会读取原模型回答或用其准确率调参。

资源估计：BF16 8B 权重约16GB，14B约28GB，加激活、CUDA及输出头临时区；8/4候选序列批量建议分别预留至少约24GB/40GB显存，实际取决于提示长度和后端。为了避免整段输入的 vocabulary logits 爆内存，实现只对续写位置投影 lm_head。无梯度、不训练模型参数。默认浮点精度与任意 vLLM 内核不保证逐位一致，故所有方法只能与本脚本 baseline 比较。

基线每64提示落一个可恢复特征分片；完整缓存保存后重跑复用。投影结果按折、方法完成后缓存。运行身份包含代码、模型配置、输入和划分哈希；不同身份禁止混写。修改代码后请用新的 output 目录。

参考方法与来源核验见 `research/methods_20261025/activation_calibration_review.md`。这些是已有校准/子空间方法的明确适配，尚无本项目实测改进结果。
