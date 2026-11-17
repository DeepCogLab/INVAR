# 新增稳健性实验：当前结果

已评分 8/8 个检查点。此文件由已验收的结果生成，未完成项不补零。

范围：中英双语；行为与事实为四来源各8对；目标检查为13对（来源不均衡）；参数检查为20对互斥最优切片。英文经AI翻译及结构核验，未获人工语义认证。

| 模型 | 语言 | 设置 | E（条件有效） | Q（全部尝试） | 参数四题全对 | 行为交付率 |
|---|---|---|---:|---:|---:|---:|
| qwen3_8b | zh | native_off | 0.0389 | 0.7163 | 0.0000 | 0.9914 |
| qwen3_8b | en | native_off | 0.0531 | 0.7356 | 0.0000 | 0.9993 |
| qwen3_8b | zh | native_on | 0.0075 | 0.9920 | 1.0000 | 0.9868 |
| qwen3_8b | en | native_on | 0.0054 | 0.9936 | 1.0000 | 0.9325 |
| gpt_oss_20b | zh | effort_low | 0.0796 | 0.9647 | 0.7083 | 0.9709 |
| gpt_oss_20b | en | effort_low | 0.0664 | 0.8093 | 0.3542 | 0.8889 |
| gpt_oss_20b | zh | effort_high | 0.0179 | 0.9984 | 1.0000 | 0.9696 |
| gpt_oss_20b | en | effort_high | 0.0182 | 0.9952 | 1.0000 | 0.9841 |
| llama31_8b | zh | direct | 0.1236 | 0.5913 | 0.0000 | 0.9980 |
| llama31_8b | en | direct | 未定义 | 0.6362 | 0.0000 | 0.9881 |
| llama31_8b | zh | brief_reasoning | 0.1081 | 0.5785 | 0.0208 | 0.9980 |
| llama31_8b | en | brief_reasoning | 0.1928 | 0.6426 | 0.0208 | 0.7057 |
| glm4_9b | zh | direct | 未定义 | 0.6651 | 0.0000 | 0.9722 |
| glm4_9b | en | direct | 0.0716 | 0.6571 | 0.0000 | 0.9954 |
| glm4_9b | zh | brief_reasoning | 未定义 | 0.6811 | 0.0208 | 0.9537 |
| glm4_9b | en | brief_reasoning | 0.1172 | 0.9696 | 0.6875 | 0.9967 |
| mistral7b_v01 | zh | direct | 0.1688 | 0.4952 | 0.0000 | 0.9835 |
| mistral7b_v01 | en | direct | 0.1410 | 0.5801 | 0.0208 | 0.9914 |
| mistral7b_v01 | zh | brief_reasoning | 未定义 | 0.4968 | 0.0000 | 0.9411 |
| mistral7b_v01 | en | brief_reasoning | 0.1252 | 0.5401 | 0.0000 | 0.9815 |
| sotopia_mistral7b | zh | direct | 0.1094 | 0.6394 | 0.0000 | 0.9980 |
| sotopia_mistral7b | en | direct | 0.0716 | 0.5913 | 0.0000 | 1.0000 |
| sotopia_mistral7b | zh | brief_reasoning | 0.1418 | 0.6074 | 0.0000 | 0.9901 |
| sotopia_mistral7b | en | brief_reasoning | 0.0774 | 0.5978 | 0.0000 | 1.0000 |
| socialr1_8b | zh | native_off | 未定义 | 0.6410 | 0.0000 | 0.9769 |
| socialr1_8b | en | native_off | 0.0991 | 0.6587 | 0.0000 | 0.9934 |
| socialr1_8b | zh | native_on | 0.1076 | 0.9712 | 0.7500 | 0.8267 |
| socialr1_8b | en | native_on | 0.1411 | 0.9567 | 0.7708 | 0.8816 |
| qwen3_8b_base | zh | direct | 0.1251 | 0.6490 | 0.0208 | 0.9960 |
| qwen3_8b_base | en | direct | 0.1279 | 0.5865 | 0.0208 | 0.9967 |
| qwen3_8b_base | zh | brief_reasoning | 0.1414 | 0.6843 | 0.0000 | 0.9180 |
| qwen3_8b_base | en | brief_reasoning | 0.1694 | 0.6490 | 0.0000 | 0.9200 |

解读要求：原生开关、推理强度、提示推理三类对照分别分析；基模单列。E较小不等于目标执行正确，也不证明模型关注或忽略了材料。先检查交付率、缺失界和同题抽样波动，再解释稳定性。来源加权的方向差异只能在明确口径下描述。

详细数字见 summary.csv、各模型 SCORING_COMPLETE.json 及 MONTE_CARLO_SENSITIVITY.json。后者为固定题库的抽样敏感性范围，不是对新来源的置信区间。
