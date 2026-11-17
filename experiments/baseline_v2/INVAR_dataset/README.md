# INVAR 数据包：多模型测试入口

版本：0.1.0，构建日期：2026-10-22。**已完成实际数据构建、来源复核、程序校验和评估流程测试，可以开始模型先导测试。尚未运行真实模型，也未冒称已完成人类研究者双人审核。**

本数据检验两个问题：相同社会情境的不同呈现是否改变模型行为；社会条件变化带来的行为差异，是否会因这些呈现变化而改变。数据不预设模型一定失败，不提供虚构的“人类正确选择”。

## 1. 先看哪些文件

| 文件 | 用途 |
|---|---|
| `data/prompts.jsonl` | **发送给模型的输入**；每行一份去重提示，只发送其中的`messages` |
| `data/quartets.jsonl` | A/B/C/D四个提示的对应关系、变体类型、所属比较 |
| `data/pairs.jsonl` | 为什么比较这两个条件、来源、干预性质和改编记录 |
| `data/cases.jsonl` | 原始事实、行动空间、收益表及来源条件，供审核，不整体发送给模型 |
| `samples/全部基准题.md` | 按条件对浏览所有基准题，适合先阅读 |
| `samples/完整四元组示例.jsonl` | 跨来源、任务和角色的完整四元组文本 |
| `DATA_CARD.md` | 实际规模、适用结论与限制 |
| `audit/实施调整与验收.md` | 构建时发现并修正的问题，以及已完成的检查 |
| `audit/*_audit.md` | 三组来源的逐项规则复核 |

`expected_direction`为空表示不提供方向真值，不表示“预期不变”。`human_direction_scoring_enabled=false`禁止将这些题直接算成人类行为准确率。

## 2. 实际数据规模

- 92组条件对，184条条件记录。
- 701个四元组，展开后有2804个条件引用。
- **去重后1240份唯一模型输入**；这是数据规模，不是1240个独立社会情境。
- 54组是原研究已有条件之间的比较，38组包含研究者新增干预。
- 6类任务，6类变体家族、8种具体操作；另有36条独立角色理解检查，不计入1240份主输入。

| 分析层 | 条件对 | 四元组 | 唯一提示 |
|---|---:|---:|---:|
| `source_conditions`：原研究已有条件比较 | 54 | 415 | 836 |
| `researcher_extension`：新增制度或行动来源 | 38 | 286 | 608 |
| 全部合并去重 | 92 | 701 | 1240 |

两层共享204份提示，不能将836与608直接相加作为总调用量。“原研究已有条件”仍包含中文、固定角色或单节点适配，不等于原实验逐字复现。

建议先运行`source_conditions`，随后单列扩展分析。独裁者任务目前只在扩展层；不能声称原条件核心层完整覆盖六类任务。

## 3. 如何直接接入多个模型

Python脚本只需标准库，建议Python 3.10及以上。先进入本目录。以下名称只是占位符，运行前换成实际模型版本；脚本不会访问模型API或产生服务费用。

先生成小规模技术先导计划：

```powershell
python make_run_plan.py --models actual-model-version --tier source_conditions --smoke --out runs/pilot_plan.jsonl
```

先导默认每题2次，覆盖各来源/任务/角色的代表条件对。当前示例对应136个唯一提示，加独立重复控制后，每模型304次计划调用。**2次只检查流程，不能作为稳定性证据。**

正式计划示例：

```powershell
python make_run_plan.py --models model-A-snapshot model-B-snapshot --tier source_conditions --repeats 32 --out runs/main_plan.jsonl
```

若决定同时测试两层，将`--tier`改为`all`。每个模型的主调用量为1240×32＝39680次，另加约10%提示的独立第二批重复；准确预算见自动生成的`.manifest.json`。不同模型各自复用同一提示清单。

对计划中的每一行：

1. 根据`prompt_id`找到`prompts.jsonl`中的记录。
2. **只将`messages`原样传给模型**；不发送收益校验表、来源说明、分析方向、四元组ID或其他条件。
3. 每次独立对话，不共享历史。按照计划顺序交错调用，不把四元组放进同一聊天。
4. 固定并记录每个模型的实际版本、温度、推理设置、最大输出长度及运行日期。模型不支持某设置时如实记录。
5. 保存原始返回内容，按下列结构每行一条。`replicate`和`batch`从计划复制，服务失败也留一行。

```json
{
  "model": "actual-model-version",
  "prompt_id": "从计划复制真实ID",
  "replicate": 0,
  "batch": "main",
  "status": "ok",
  "response_text": "模型原始返回文本",
  "generation_config": {
    "snapshot": "实际版本",
    "temperature": 0.7,
    "reasoning": "实际设置",
    "max_output_tokens": 256
  }
}
```

`main`是主样本，`repeat`是同提示的独立第二批。相同模型、提示、重复编号和批次不得重复写入。模型配置改变时使用新的模型键，不混在一个模型名下。响应格式中的参数只是填写示例，不代表所有模型都支持或应使用这些值。

原始失败、拒答或超额输出不得由人工替模型补答案。若做格式重试，保留原始输出，将重试放在独立文件中；主评估默认使用第一次响应，避免混淆重试后行为。

## 4. 如何评估保存的输出

```powershell
python evaluate.py runs/responses.jsonl --tier source_conditions --out reports/source_conditions
python evaluate.py runs/responses.jsonl --tier researcher_extension --out reports/extensions
```

脚本会生成：

- `quartet_metrics.csv`：等价行为差异E、两种呈现的干预效应Δ、效应差异G、每侧有效回答数和失败率。
- `stratified_summary.csv`：按模型、来源、任务、角色、干预类别和变体分别汇总。
- `output_validity.csv`：全部选定提示的合法输出情况，包含未返回记录。
- `same_prompt_repeats.csv`：相同提示独立两批的行为距离。
- `evaluation_report.json`：覆盖、运行配置及解释限制。

金额行动使用固定尺度Wasserstein-1距离；二元任务等价于选择概率差。脚本严格保留行动类型，例如整数3与字符串"3"不同，带解释或代码围栏的输出不会悄悄清洗为合法答案。

`null_E_mean`是合并样本置换得到的有限采样参考，**不是已经扣除误差的分数，也不是显著性检验结论**。评估脚本提供描述性结果与完整性检查；论文推断应在看正式结果前确定实际意义阈值、依赖结构和多重比较处理。共享提示须只重采样一次再回填各四元组。

不把Δ越大当成越好；不把零响应直接叫失敏；不把G包装成独立第三种能力。在本数据的标量Wasserstein定义下有G≤2E。

## 5. 理解对照与原提示重复

原提示重复已进入运行计划。`controls/role_comprehension.jsonl`另提供36条角色绑定检查，覆盖基准、自然段和编号变化，单独调用、单独保存。只发送其`messages`，绝不发送`expected_answer`，也不将这些问答接在主决策前。

```powershell
python score_controls.py runs/control_responses.jsonl --out reports/role_control.csv
```

控制结果格式同主响应文件，但返回对象应为`{"answer":"P2"}`等实际编号。控制仅检查角色理解，**不是完整收益理解测试**；论文如需将错误归因为特定社会能力，应再对预先固定的子集补充收益、权限理解检查，并使用一般格式对照，不能仅凭这36题排除通用语言因素。

## 6. 重建与验证

验证当前数据和评分程序：

```powershell
python -m unittest discover -s tests -v
```

从包内事实记录重新生成所有提示，不需要联网或参与者记录：

```powershell
python build_dataset.py --from-canonical
```

此路径已验证为数据文件逐字节一致。它验证模板渲染的可重复性，不替代对原文的审核。

从原始研究档案重新提取并构建：

```powershell
python build_dataset.py
```

后一条需要本项目原有的`../research/evidence/`档案，包括PIQ参数表、Econ问卷文本、PGG指令、El Zein条件存在核验所用CSV。**为避免重新分发参与者记录，压缩包不包含这些个人回答表。** 原刺激文件及许可元数据见`provenance/source_materials/`，获取链接和哈希见`sources.jsonl`及各来源审核记录。

## 7. 正式运行前的研究者验收

程序和AI复核已完成，尚不能写成“两名人类专家审核通过”。研究者应先阅读每类任务的基准题、来源映射和关键改编，确认研究边界；按`audit/equivalence_checks.jsonl`保留真实的人工作业记录。所有V6关系改写及角色卡句式转换尤其需要核对语用暗示。

可先做技术先导；正式发表前应完成真实审核并锁定数据版本。若审核发现规则错误，可以修复、重新生成并升级版本；不能根据哪些题更容易得到模型失败来选题或调整阈值。
