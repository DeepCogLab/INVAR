# 新增实验：经独立验收的更正结果

**写作使用 summary_corrected.csv：全部采用统一v3评分。v3修复空白行重复计JSON起点，并包含GPT-OSS的v2通道元数据修正。原始回答、冻结代码及旧分数保留。**

当前独立验收：8/8检查点；原主矩阵全部验收，夜间补充单独报告。

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
| llama31_8b | zh | direct | 59.13% | 0.00% | 0.12358 | 99.80% |
| llama31_8b | en | direct | 63.62% | 0.00% | 未定义 | 98.81% |
| llama31_8b | zh | brief_reasoning | 57.85% | 2.08% | 0.10809 | 99.80% |
| llama31_8b | en | brief_reasoning | 85.26% | 14.58% | 0.16083 | 94.71% |
| glm4_9b | zh | direct | 66.51% | 0.00% | 未定义 | 97.22% |
| glm4_9b | en | direct | 65.71% | 0.00% | 0.07163 | 99.54% |
| glm4_9b | zh | brief_reasoning | 68.11% | 2.08% | 未定义 | 95.37% |
| glm4_9b | en | brief_reasoning | 96.96% | 68.75% | 0.11757 | 99.74% |
| mistral7b_v01 | zh | direct | 49.52% | 0.00% | 0.16884 | 98.35% |
| mistral7b_v01 | en | direct | 58.01% | 2.08% | 0.14101 | 99.14% |
| mistral7b_v01 | zh | brief_reasoning | 49.68% | 0.00% | 0.14202 | 94.18% |
| mistral7b_v01 | en | brief_reasoning | 54.01% | 0.00% | 0.12525 | 98.15% |
| sotopia_mistral7b | zh | direct | 63.94% | 0.00% | 0.10941 | 99.80% |
| sotopia_mistral7b | en | direct | 59.13% | 0.00% | 0.07161 | 100.00% |
| sotopia_mistral7b | zh | brief_reasoning | 60.74% | 0.00% | 0.14180 | 99.01% |
| sotopia_mistral7b | en | brief_reasoning | 59.78% | 0.00% | 0.07736 | 100.00% |
| socialr1_8b | zh | native_off | 64.10% | 0.00% | 未定义 | 97.69% |
| socialr1_8b | en | native_off | 65.87% | 0.00% | 0.09912 | 99.34% |
| socialr1_8b | zh | native_on | 97.12% | 75.00% | 0.10758 | 82.67% |
| socialr1_8b | en | native_on | 95.67% | 77.08% | 0.14110 | 88.16% |
| qwen3_8b_base | zh | direct | 64.90% | 2.08% | 0.12479 | 99.74% |
| qwen3_8b_base | en | direct | 58.65% | 2.08% | 0.12786 | 99.67% |
| qwen3_8b_base | zh | brief_reasoning | 71.47% | 0.00% | 0.14143 | 92.66% |
| qwen3_8b_base | en | brief_reasoning | 66.67% | 2.08% | 0.16698 | 93.92% |

原生开关、原生推理强度、提示推理分开解释；基模单列。英文没有独立人类语义认证。E每题8次，只作探索诊断，须结合缺失界与抽样敏感性。

详见 ../SCORING_ERRATUM_TERMINAL_V3.md、../SCORING_ERRATUM_HARMONY_V2.md 与 SCORING_VERSION_SELECTION.json。部分题全部未交付时，完整面板条件E保持未定义，不能填零。
