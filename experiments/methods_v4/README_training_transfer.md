# 四来源、三训练种子、两模型迁移训练

本阶段只新增训练实现，不修改V1/V2冻结文件、生成评估或调度。所有来源与已有结果已被查看，因此称为**回溯迁移实验**，不称为新确认集。

## 复用与变化

复用V2的`heldout_source`、`seed`、来源引用闭包、AB/CD等价边、条件连通分量、完整候选似然、后缀投影及曝光schedule。D0监督loss与自由JS公式保持原样。新增内容仅为：

- 去除不需要的teacher/KL依赖，两个arm都不会创建或读取教师缓存。
- 支持Qwen3-8B及Qwen2.5-14B-Instruct的原生模板和q/v LoRA。
- 四来源留一、三个固定训练seed的独立运行身份与输出锁。

精确protocol：`INVAR_retrospective_source_model_seed_transfer_20261026_v1`。

## 冻结设置

模型ID为`qwen3_8b`、`qwen25_14b_instruct`；留出来源为HPVBG、IGVOFO、PGG、PIQ572；训练seed为20261026、20261028、20261029。两个arm是`anchor_sft`和`anchor_js`，后者自由JS权重1，前者0。

两臂均固定rank8、alpha16、q/v投影、LoRA dropout0、学习率1e-4、weight decay0.01、梯度裁剪1、梯度累积8、BF16、梯度检查点、150优化器步。每步8个监督边与自由边单元。`anchor_sft`也计算自由分布，并通过零系数保持相同的曝光和计算图；不伪造自由行为正确答案。

只有完整2/7/11项公开菜单进入自由学习。大菜单不抽样、不重标、不缩成小菜单，在原始评估中保留作迁移。V4规则表格、V6关系表述不作为训练边；监督正确集合不因自由菜单预算而截断。监督D0仍是正确完整字符串集合概率的负对数除以正确后缀平均长度，本阶段不修改其已说明的长度/格式权重影响。

训练seed与推理seed分别配置。本脚本只控制训练初始化和样本曝光，不能把推理seed自动改成训练seed。

## 来源池审计

| 留出来源 | 监督唯一输入 | 自由唯一输入 | 自由条件分量 |
|---|---:|---:|---:|
| HPVBG | 5,010 | 774 | 115 |
| IGVOFO | 5,192 | 842 | 127 |
| PGG | 4,450 | 562 | 87 |
| PIQ572 | 2,394 | 432 | 64 |

同一留出来源下，两个模型、两个arm和三个seed共用完全相同的来源池；同一seed的曝光schedule也相同。改变seed仅改变曝光顺序/抽取与初始化，不改变来源池。不同来源训练池大小不同，不能将差异解释为同一难度的独立任务。

`source_pool_sha256`标识与模型/seed无关的准确来源池；`exposure_schedule_sha256`标识固定抽样计划；`split_content_sha256`同时绑定protocol、模型和训练seed。保存六个准备JSONL及其hash，以及候选/条件拆分审计，供评估端独立核验。

## 命令

本地纯CPU准备：

```text
python experiments/methods_v4/train_transfer.py --prepare-only --model-id qwen3_8b --heldout-source HPVBG --seed 20261026 --arm anchor_sft --output experiments/methods_v4/cpu_audit/qwen3_HPVBG_seed26
```

服务器脚本放在新stage目录；显式指定真实基线和冻结helper位置。以下是单个训练任务示例，GPU由root统一调度。本次实现没有执行GPU训练。

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/vllm/bin/python train_transfer.py \
  --baseline /workspace/project_social_simulation/invar_reliability_runs/20261024_v2 \
  --legacy-helper-dir /workspace/project_social_simulation/invar_methods_runs/20261025 \
  --v2-helper-dir /workspace/project_social_simulation/invar_methods_runs/20261026 \
  --model-id qwen25_14b_instruct --model /workspace/models/Qwen2.5-14B-Instruct \
  --heldout-source PGG --seed 20261028 --arm anchor_js \
  --output learning/qwen25_14b_instruct/PGG/seed20261028/anchor_js
```

`INVAR_METHODS_V1`/`INVAR_METHODS_V2`可以替代两个helper参数。`--model-id`是评估身份，`--model`是已有权重路径，不能混为一个参数。Qwen3模板设置`enable_thinking=False`；Qwen2.5使用原生模板，不插入Qwen3思考块。

CPU实际分词审计在上述命令添加`--prepare-only --tokenizer-audit`并清空`CUDA_VISIBLE_DEVICES`即可；只加载分词器，不加载模型权重。默认4096，超长报错，不截断。实际训练仅投影答案长度+1个LM-head位置，和V2完整概率口径一致。

## 输出与锁

默认输出：`learning/<model_id>/<heldout_source>/seed<training_seed>/<arm>/`；也可指定准确叶目录。

每个叶目录独占`RUN.lock`；成功准备/训练后删除锁，失败保留锁和带时间戳的错误记录。不能自动覆盖另一个模型/来源/seed的池，也不能并发写同一叶目录。已开始训练不自动resume；checkpoint不含优化器和RNG状态。

配置保存模型、模板、基础配置hash、权重文件metadata、helper/runner hash、来源池/曝光/划分hash、teacher_required=False及teacher_manifest_sha256=None。权重metadata不等于全权重字节校验，不作过度声明。

每25步写适配器checkpoint与身份，逐步写进度；最终保存adapter文件hash、实际曝光计数、训练配置hash和完成记录。`training_complete`仅表示训练完整，不能代替生成评估或宣布可靠性提高。

## 已完成CPU检查

```text
python -m unittest discover -s experiments/methods_v4/tests -p test_train_transfer_cpu.py -v
```

测试核验四来源闭包、无自由硬标签、菜单完整、V4/V6和AC排除、跨模型/arm同曝光、不同seed同池、原生模板参数及不截断。

在服务器现有环境以CPU小张量测试通过：JS数值及梯度与冻结V2一致；D0保持且反向有限；SFT自由JS零系数产生零梯度。另用随机微型Qwen2模型验证其原生`logits_to_keep`与完整投影的损失一致、反向有限。均未调用CUDA或载入真实模型权重。
