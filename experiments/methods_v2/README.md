# 保守模块归一化：独立第二轮开发实验

本轮仅检验两个无需模型改述的输入处理基线，使用第一轮相同冻结清单：1,378份输入，每模型每方法18,752次响应。三款模型为Qwen3-8B、Qwen2.5-14B-Instruct、GLM-4-9B-0414。方法包括：

- `baseline`：原始messages逐字保留。
- `canonical_modules`：仅重新排列已识别完整模块，固定为个体设定、实验规则、当前情况、本次明确目标；不改模块原文。
- `canonical_profile`：模块顺序同上；另外仅把严格识别的“条目/内容”两列个体设定表转为列表。表头词和全部单元格文字保留；只移除已识别Markdown装饰与连续行号。

预处理直接调用冻结的`../methods_v1/conservative_normalizer.py`，只接收当前messages，不访问case、pair、variant、答案和收益真值。未知、重复或无法安全分段的标题，及不支持的表结构，一律保留原始输入并记录fallback。问题、原系统指令、规则收益表、明示目标与未知事实不得删改。该方法对当前显式模块模板保守适用，不声称已具有任意自然语言的一般语义规范化能力。

CPU审计已完成：全8,666份输入的元数据哨兵和原文片段保留检查通过；模块方法改变1,994份，模块加个体表转换改变2,991份，其中997份转换个体表，均无回退。冻结开发清单内，模块方法改变312/1,378份，另加个体表转换后改变468份，其中156份转换表格；baseline逐字保留全部1,378份。未知标题等人工边界测试仍验证必须回退，不因当前语料未触发回退而删除该保护。

这是第二轮独立运行协议`INVAR_conservative_normalization_screen_v2_20261026`，**base_seed=20261026**。基线重新采样，不能拿第一轮base_seed=20261025的结果代替匹配基线。三方法的行为32次、事实每种选项顺序8次、目标16次、角色16次；temperature=0.7、top_p=0.95、JSON xgrammar、max_tokens=128，与第一轮相同。方法不共享随机种子；比较配对在题目层面，不把同一replicate编号当同一决策。

运行前CPU检查：

```text
python experiments/methods_v2/run_normalized_methods.py --self-test
python experiments/methods_v2/run_normalized_methods.py --audit-corpus --baseline experiments/baseline_v2 --manifest experiments/methods_v1/split_manifest.json --out experiments/methods_v2/corpus_audit
```

GPU运行示例（由根进程审查并排队，本文件交付时尚未启动GPU）：

```text
python experiments/methods_v2/run_normalized_methods.py --baseline experiments/baseline_v2 --manifest experiments/methods_v1/split_manifest.json --model qwen3_8b --methods baseline,canonical_modules,canonical_profile --out experiments/methods_v2/outputs
```

每个模型单个vLLM实例依次运行三种方法。默认batch为192，14B为96，可明确指定`--batch-size`。新config包含driver、normalizer、V1 helper、原始模型模板renderer的代码哈希、模型权重身份、库版本、输入及清单哈希、解码参数和新种子政策。helper函数通过明确路径导入，未改写其任何全局常量；新种子由本driver独立计算。

本地默认从相邻`methods_v1`加载helper。远端第二轮目录为`/workspace/project_social_simulation/invar_methods_runs/20261026`，第一轮目录名为`20261025`，运行前显式设置环境变量：

```text
export INVAR_METHODS_V1=/workspace/project_social_simulation/invar_methods_runs/20261025
```

随后运行第二轮driver。`config.json`、运行identity及CPU审计identity会记录解析后的helper目录，以及具体normalizer/helper文件的绝对路径与SHA-256；更换目录或代码不能混入既有配置。

输出`<out>/<model>/`：

| 文件 | 内容 |
|---|---|
| `normalizer_input.jsonl` | 当前原messages及公开合法菜单，供独立核对；没有正确答案元数据 |
| `<method>.normalizer_changes.jsonl` | 变换后的原文、完整保留/删除片段审计、metadata哨兵检查 |
| `normalizer_summary.json` | 变换、无变化、回退、角色表转换覆盖率及文件哈希 |
| `config.json`、`identity.json` | 明确的第二轮运行身份与模板/权重/代码哈希 |
| `<method>.jsonl` | 与第一轮评分格式兼容的逐响应、输入输出token、fallback、种子、内容哈希 |
| `<method>.complete.json` | 独立完整性、原始响应哈希与token成本 |
| `COMPLETE.json`、`progress.json` | 所请求方法的完成状态及进度 |

续跑只恢复同配置未写入的试次；已有响应必须满足新seed、输入hash、schema及预算。非法输出先原样保存，再停止，不重抽到答对。已完成调用可复核并重建完成标记，无需加载GPU。

评分需要第二轮独立评分目录及明确支持新方法名的评分入口；第一轮冻结评分程序的白名单尚不包含两个新名称，不能把其结果伪装成原方法或改写第一轮scores。主评价仍共同看E/G、事实更新/保持U、目标Q、熵、收益与权限信息保留；并用平方距离有限采样辅助指标检查表观改善。归一化后原/变体输入可能逐字相同，此时零差异是预处理系统的性质，不能宣称底层模型获得了普遍语义不变性。
