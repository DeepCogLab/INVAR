# LoRA 学习先导运行与验证

此轮为回溯开发先导。HPVBG 来源不进入权重训练，但V2结果已被查看；不能称为从未接触的确认集。当前脚本仅实施SFT与SFT加一致性代理，不表示完整候选方法已经奏效。

## CPU准备

在项目根目录运行，不需要加载任何模型或使用GPU：

```text
python experiments/methods_v1/train_lora.py --prepare-only --output experiments/methods_v1/learning_pilot/cpu_audit
```

如果服务器已有Qwen3权重，先额外做实际分词器的纯CPU审计。下列为服务器Linux示例；模型路径应指向已有权重。

```bash
CUDA_VISIBLE_DEVICES="" python experiments/methods_v1/train_lora.py \
  --prepare-only --tokenizer-audit --model /workspace/models/Qwen3-8B \
  --output experiments/methods_v1/learning_pilot/cpu_tokenizer_audit
```

`tokenization_audit.json`报告全部合法候选的最长完整输入、最长提示、最长答案及最大LM-head投影位置数。默认预算4096；任何超长都会报错，不截断。该命令仅加载分词器，不加载模型权重、不调用CUDA。默认不加`--tokenizer-audit`的准备命令连torch也不导入。

CPU数据审计为5,010个唯一监督输入、4,382条等价边；其中PIQ572 3,288、PGG 1,232、IGVOFO 490。所有引用来源必须一致且不为HPVBG；V4_rules_table和V6_relation排除。仅训练事实与明示目标，不标自由行为。两种mode使用相同任务平衡抽样与相同种子；一致性只计算AB/CD，并按语义映射反转过的A/B选项。

## 单GPU独立训练

先用`--max-steps 2`和独立输出目录做显存/梯度先导，再执行完整150步。以下两条应顺序运行，或明确分配不同空闲GPU；不要共用输出目录。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/methods_v1/train_lora.py \
  --mode sft --model /workspace/models/Qwen3-8B \
  --output experiments/methods_v1/learning_pilot/HPVBG_sft_seed20261025

CUDA_VISIBLE_DEVICES=0 python experiments/methods_v1/train_lora.py \
  --mode sft_consistency --model /workspace/models/Qwen3-8B \
  --output experiments/methods_v1/learning_pilot/HPVBG_sft_consistency_seed20261025
```

固定rank8、alpha16、q/v投影、LoRA dropout0、学习率1e-4、批量1、梯度累积8、BF16及梯度检查点。批量1指每次候选前向；一个监督单元仍包含两个等价输入，并分别处理完整合法候选。无需datasets、trl或bitsandbytes，不下载额外权重。

每个候选仅请求`logits_to_keep=answer_len+1`，随后丢弃最后一项并与答案后缀对齐。因此大词表LM-head不再投影完整提示。每个候选的上下文计算及训练图仍然存在，候选数增加仍会增加开销；实际峰值以先导日志为准。模型若不原生支持该参数就拒绝训练，不能静默回退到全提示logits。

每25个优化器步和最后一步写`adapter_checkpoint_XXX`，每步原子更新`progress.json`；最终另存`adapter/`。checkpoint是可检视、可推理的适配器，**没有保存优化器和RNG状态，不能精确续训**。脚本拒绝自动覆盖已开始目录；中断后可以评估已有checkpoint，重新训练则需新目录并从0步计数，不能把旧步数伪装成已恢复。

## vLLM适配器核验

在正式大规模测试前，先检验PEFT导出的适配器能被服务器现有vLLM载入。使用相同基础权重、分词器、原生模板和V2结构化输出策略；开启LoRA支持，`max_lora_rank=8`，通过`LoRARequest`显式指定适配器目录。需要同时记录适配器权重哈希、基础模型配置哈希、模板、vLLM版本和推理参数。不要把检查点目录存在当成适配器已生效。

建议依次检查：

1. 在启用LoRA的同一推理引擎中，分别运行无适配器基线与指定适配器，确认请求确实绑定相应适配器，未默默回退到基础模型。
2. 将同一适配器用Transformers+PEFT与vLLM分别加载，在少量固定输入比较候选对数似然/输出分布及合法JSON。两套实现可能有浮点或解码差异，不要求随机采样文本逐字相同。
3. 最好增加零增量适配器检查，以识别仅开启LoRA引擎造成的数值差异。不能用有训练适配器输出偶然相同作为“未生效”的唯一证据，也不能只凭输出不同确认实现正确。
4. 在同接口、同种子计划下重新采样SFT与SFT一致性方法；正式报告E/R/G/U/Q，而不是把训练代理JS当生成行为E。

完整合法答案序列似然的归一化分布不等于逐token语法约束、temperature和top_p组成的生成分布；代理改善必须经过实际生成验证。目标并列最优使用正确集合概率质量，不任意挑一个行动作唯一答案。

这轮禁止依据HPVBG结果挑checkpoint、超参或某个幸运种子；150步固定终点是主先导对比，25步检查点用于运行检查与后续独立开发。若看过测试结果后继续调参，该结果只能作为开发证据，仍需新的冻结确认集。
