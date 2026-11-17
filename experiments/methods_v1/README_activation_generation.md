# 激活方法：真实 HF 受约束生成筛选

`run_activation_generation.py` 是独立生成实验，不改变已冻结 `activation_probe.py`，不把其候选似然归一化概率当成生成分布。默认只运行 Qwen3 的 HPVBG 来源留出折：256提示 × 32真实采样 × 3方法，共24,576条。`--all-folds` 才扩展到四来源；它仍只覆盖代理筛选中菜单恰好二选一的同一子集。

三方法为匹配 HF baseline、paired_svd、condition_protected_svd。参数直接加载 proxy 对应折的训练IDs、均值和投影基，不按生成结果重新拟合或选参。训练ID哈希、参数文件哈希、拟合审计哈希、proxy身份、当前输入、模型配置及生成代码身份全部保存并核对。校准方法未进入本轮，因为代理的整项分数偏移与逐token生成偏置不能未经定义就互换。

只接受已修正“事实选项语义顺序不匹配训练边”的 proxy v2 身份和参数；旧v1部分运行参数拒绝加载，不能静默继续。

## 生成与干预定义

两个合法完整JSON候选只从题目 output_schema 得到，与正确答案无关。使用与 proxy 同一 native chat template；tokenize 后取完整候选最长公共前缀，其最后token为固定特征位。共同 JSON 前缀以输入 token IDs 固定提供，**response_text 明确是固定共同JSON前缀加真实生成suffix**；这两个部分分别存储。没有候选特有行动作为固定输入。

真实生成使用 HF `generate`、精确候选 suffix trie、temperature=0.7、top_p=0.95、HF top_k=0、repetition_penalty=1、max_new_tokens=128、KV cache。只允许完整合法JSON后EOS；早停EOS或非法token会失败。EOS后的HF padding允许存在，但不能混入其他token。语法不读取 expected_answer/expected_actions。后者只在所有响应保存后的评分中使用。

一次chunk内所有提示完全相同，仅重复采样4或8次。首次prefill固定前缀末token在模型中层块输出处修改一次；后续每个decode调用都只有1个token，hook验证它们不被修改。中层、强度沿用 proxy：18/24块、0.5；模型和精度与当前匹配 baseline一致。每chunk记录prefill次数、decode次数、修改位置数、最大L2修改范数。

base_seed固定20261025。按(model, fold, prompt_id, chunk_replicate_start)哈希生成chunk_seed，排除method，因此三方法使用相同初始随机数流。每个chunk内部由HF真实抽样，记录同一个chunk_seed及各行chunk_position；**不宣称HF已对每行设置独立Generator，也不宣称方法间每个后续随机数消费位置都完全相同**。批量大小写入身份，改变批量须新建输出目录。

生成分布受prefix trie逐步重归一化及top-p影响，与完整JSON原始条件似然softmax不同。它也不宣称与vLLM/xgrammar协议逐位一致。因此改善只与本实验HF baseline比较，V2结果仅作另一个接口的背景。

## 文件与检查

每折每方法普通JSONL保存真实response_text、token、trial_id、pid、replicate、seed、finish_reason、source、fold及身份哈希。没有“复制代理概率抽样”的步骤。重跑只跳过身份一致的已保存试次；部分chunk已写入时重新生成同seed chunk，并仅补未保存行。非法/截断响应写入后使验收失败，不替换为新正确回答。

保存完整采样响应后按实际候选计数评分，得到E/R/G及U/Q经验正确率。每提示32次，仅报告开发筛选点估计，不构造虚假大样本置信区间或显著性。只对完整支持菜单的四元组报告行为结果，覆盖缺失单列。

CPU模拟验证：

```text
python -m unittest discover -s experiments/methods_v1 -p test_activation_generation.py -v
```

GPU由调度者启动；这里没有自动启动：

```text
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python experiments/methods_v1/run_activation_generation.py --model qwen3_8b --fold HPVBG --proxy-dir experiments/methods_v1/activation_proxy_screen/qwen3_8b --output experiments/methods_v1/activation_generation_screen/qwen3_8b/HPVBG --batch-size 4 --device cuda:0
```

参数文件齐全后可以先加 `--audit-only`，只做CPU来源、参数、候选范围审查。`--all-folds` 与 `--fold` 互斥。首次有希望后，应另冻结更大范围和更多采样的确认运行，不能直接把本轮32次开发结果当最终方法收益。
