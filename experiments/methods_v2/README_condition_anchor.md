# 条件软锚定四臂学习先导

状态：CPU准备与小张量测试完成，尚未由本实现启动GPU教师缓存或训练。第一轮出现监督正确率提高但自由行为E变大的现象，因此本轮直接约束自由行为，同时检查是否抹平真实条件信息。此前V2及第一轮结果已经查看，本轮仍是回溯方法开发，不是新确认集。

## 四臂与固定设置

| arm | 原D0监督损失 | 自由行为JS | 每条件教师KL |
|---|---|---|---|
| anchor_sft | 有 | 0 | 0 |
| anchor_js | 有 | 1 | 0 |
| anchor_kl | 有 | 0 | 1 |
| anchor_js_kl | 有 | 1 | 1 |

四臂不加入第一轮的事实/目标JS，以保证组件可归因。D0保持原口径：正确完整答案集合概率的负对数，除以正确答案后缀平均token长度。并列最优全部进入正确集合；不在本轮静默更换监督loss。

初始模型Qwen3-8B，rank8、alpha16、q/v投影、LoRA dropout0、学习率1e-4、梯度累积8、150个优化器步、BF16、梯度检查点，seed=20261026。每步8个监督边+自由行为边单元。四臂预先生成完全相同的曝光schedule；连anchor_sft也计算自由行为分布与两项诊断，只将它们的系数设0，保留零梯度路径。所有臂从相同基模与适配器初始化开始，不承接不同的最佳checkpoint。

## 来源、变体与菜单预算

HPVBG不进入训练或教师。来源由全部题目引用闭包核验，训练只用PIQ572、PGG、IGVOFO；V4_rules_table/V6_relation不作为训练边。自由行为只连AB与CD，连通分量构成独立真实条件；如果任何分量同时包含某个干预的A和C，会立即报错。

首轮公开菜单预算固定为`max_free_candidates=16`，完整枚举其中全部候选，不做候选抽样。实际进入训练的菜单大小为2、7、11。2,001—12,001项的大菜单不进入自由行为教师/JS/KL训练，但完整保留在生成评估中作为大菜单迁移；这按公开菜单大小事前决定，不按模型分数选题。

CPU准备得到：

- 监督输入5,010个、等价边4,382条，沿用第一轮的事实/目标标签。
- 自由行为输入774个、等价边659条、独立条件分量115个。
- 自由输入来源：PIQ572 438、PGG 308、IGVOFO 28。
- 条件分量来源：PIQ572 67、PGG 44、IGVOFO 4。
- 监督合法菜单最大7项，并列正确集合最大7项；不为了自由菜单上限删掉监督正确集合。

`data_audit.json`逐来源/任务/菜单列明纳入与排除数。`condition_split_audit.json`、`candidate_audit.json`、六个准备JSONL保存来源引用、候选、边、条件及曝光清单。大菜单排除边数与独立输入数是不同计量，不能混用。

## CPU准备与实际分词审计

本地项目根目录：

```text
python experiments/methods_v2/train_condition_anchor.py --prepare-only --output experiments/methods_v2/cpu_audit
```

远端新阶段在`/workspace/project_social_simulation/invar_methods_runs/20261026`；冻结V1 helper在相邻`20261025`目录；基线仍在`/workspace/project_social_simulation/invar_reliability_runs/20261024_v2`。通过`INVAR_METHODS_V1`或`--legacy-helper-dir`显式指定helper，并保存其代码hash。远端基线必须显式传入，不能假定本地相对布局。

```bash
export INVAR_METHODS_V1=/workspace/project_social_simulation/invar_methods_runs/20261025
CUDA_VISIBLE_DEVICES="" python train_condition_anchor.py \
  --baseline /workspace/project_social_simulation/invar_reliability_runs/20261024_v2 \
  --prepare-only --tokenizer-audit --model /workspace/models/Qwen3-8B \
  --output cpu_tokenizer_audit
```

已在服务器现有分词器上以CPU运行，全部14,690个训练合法候选通过：最长提示591 token、最长完整序列598、最长答案后缀10、最大LM-head投影位置11；未加载模型权重，未调用CUDA。预算4096，不截断。

## 教师缓存：明确单独执行的GPU阶段

下面命令需要GPU，只应在root审查及排队后运行一次。本交付没有执行它。

```bash
CUDA_VISIBLE_DEVICES=0 python train_condition_anchor.py \
  --baseline /workspace/project_social_simulation/invar_reliability_runs/20261024_v2 \
  --model /workspace/models/Qwen3-8B --teacher-only \
  --teacher-cache teacher_cache --output teacher_preparation
```

教师为无适配器、冻结、eval、no_grad的原基模，HF计算全部合法JSON+结束标记的完整序列似然，再在菜单内归一化。**没有序列长度归一化，也不使用vLLM256次频率代替同接口教师。** 每个条件对各唯一呈现的概率做算术平均，形成该条件自己的软分布；不把不同真实条件拉近，不指定任何自由行为正确行动。

缓存构建时有排他`BUILD.lock`。只有两个分布文件完整写入并计算字节hash后，才原子发布最后的`teacher_manifest.json`；训练检查manifest、所有文件hash、来源/输入/candidate对齐、完整似然softmax和逐条件算术均值。缺少完整manifest或锁未释放都会拒绝读取。失败保留锁和错误，需人工排查，不自动把半成品当完成。

四臂共享同一已发布cache；它保存基模路径、配置hash、权重文件metadata、模板、版本、代码、拆分及候选身份。权重文件metadata不是全权重byte hash，不作过度声明。教师不是人类行为真值，也可能继承原模型错误。

## 四臂独立训练

先构建同一个150步配置的教师，随后在两张空闲GPU上各自串行安排两个arm。每个arm必须使用独立输出目录；示例只列一个：

```bash
CUDA_VISIBLE_DEVICES=0 python train_condition_anchor.py \
  --baseline /workspace/project_social_simulation/invar_reliability_runs/20261024_v2 \
  --model /workspace/models/Qwen3-8B --arm anchor_js_kl \
  --teacher-cache teacher_cache --output learning/anchor_js_kl_seed20261026
```

训练不会自动创建缺失教师。每25步与终点保存adapter checkpoint，逐步保存进度；最终另存adapter、曝光计数和训练完成hash。checkpoint没有优化器/RNG状态，不能精确自动续训；中断后保留失败，新目录从0开始，不能伪造已恢复步数。

后缀投影使用`logits_to_keep=answer_len+1`，避免全提示大词表logits。候选仍有各自上下文计算图；最坏一个单元为两份7正确答案监督输入加两份11候选自由输入，最多36个候选图。没有声称通过不正确的prefix-KV复用消除了这些图。真实峰值须在运行先导中观察，若OOM则保留失败、修订实现版本；不能临时删候选改变科学设置。

教师cache身份含完整150步拆分/曝光身份；不能随意换成`--max-steps 2`后继续声称同一缓存配置。运行检查应明确记录所用独立配置，或监视正式固定配置的早期步骤；不利用HPVBG分数挑checkpoint或改系数。

## CPU验证与后续评估

`--self-test`仅用CPU小张量验证JS/KL有限反向、teacher停止梯度、相同分布零损失、D0并列正确集合梯度、AB/CD两条件隔离及AC桥接拒绝；已在远端现有torch环境、`CUDA_VISIBLE_DEVICES=""`下通过。

```bash
CUDA_VISIBLE_DEVICES="" python train_condition_anchor.py --self-test
```

生成评估应由独立V2 runner使用同一个原始输入和合法schema加载对应adapter，重新测E/G/R/U/Q；HF代理下降不等于vLLM行为E下降。分别报告已见来源、HPVBG留出、V4/V6及大菜单迁移，并检查固定输出/均匀化、条件差异、U/Q退化与教师错误继承。四臂固定终点全部保留，不能只报告最优一臂。
