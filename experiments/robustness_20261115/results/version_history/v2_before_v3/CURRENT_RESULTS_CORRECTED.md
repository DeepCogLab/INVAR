# 新增实验：经独立验收的更正结果

**写作引用请使用 summary_corrected.csv。GPT-OSS 原v1表存在final通道格式元数据误判，已由版本化CPU复算更正；原始回答、冻结协议及原表均保留。**

当前独立验收：2/8检查点；其余仍在运行或等待验收。

| 检查点 | 语言 | 设置 | 全尝试Q（对等权） | 参数四题全对 | 条件E | 行为交付率 |
|---|---|---|---:|---:|---:|---:|
| qwen3_8b | zh | native_off | 71.63% | 0.00% | 0.03887 | 99.14% |
| qwen3_8b | en | native_off | 73.56% | 0.00% | 0.05312 | 99.93% |
| qwen3_8b | zh | native_on | 99.20% | 100.00% | 0.00745 | 98.68% |
| qwen3_8b | en | native_on | 99.36% | 100.00% | 0.00537 | 93.25% |
| gpt_oss_20b | zh | effort_low | 100.00% | 79.17% | 0.07839 | 100.00% |
| gpt_oss_20b | en | effort_low | 99.84% | 95.83% | 0.06230 | 100.00% |
| gpt_oss_20b | zh | effort_high | 99.84% | 100.00% | 0.01787 | 96.96% |
| gpt_oss_20b | en | effort_high | 99.52% | 100.00% | 0.01821 | 98.41% |

Qwen原生开关、GPT-OSS low/high推理强度、其他模型的提示推理分别分析。基模单列。英文没有独立人类语义认证；E每题8抽样，只是低预算探索性诊断，须结合缺失界和抽样敏感性。

GPT-OSS更正接受final通道附带json/JSON格式元数据，schema/type/menu/重复字段/截断规则不变；不读取analysis中的答案。其header组合按通道与格式元数据独立的语法解释处理，不声称官方逐字示例认证了该组合。

详见 ../SCORING_ERRATUM_HARMONY_V2.md 和 SCORING_VERSION_SELECTION.json；其余模型使用已独立核验的原v1结果。
