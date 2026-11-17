# 保守输入归一化对照

这是针对当前已知模板的强预处理对照，用来检验“把整块信息放回固定顺序”是否已经能解释部分收益。它不使用模型，不访问canonical题目、收益字典或答案，不声称方法创新或对外部格式泛化。本次仅实现与CPU审计，尚未跑模型效果实验，也没有修改冻结runner。

## 做什么

输入只取当前可见的`messages`，复用既有`split_material_question`保留问题边界。明确模块按以下顺序排列：

1. 个体设定。
2. 实验规则。
3. 当前情况。
4. 本次明确目标（若有）。

模块整体搬移，保留每个模块全部原文、空白、数字、参与者编号、权限、条件分支和收益规则。系统消息及问题部分保持原文。不会将“个体设定”变成额外Persona，也不会补充自利、公平或群体目标。

未知或重复heading、未归类的前置正文、缺少核心模块、没有安全换行边界等情况，都回退整条原消息，记录原因。不会删除未知段落后继续处理。规则表格始终原样保留。

可选的第二种处理只转换严格识别的个体设定二列表格：表头恰为`条目 | 内容`，第一列是从1开始连续的纯数字行号，第二列非空，没有额外列、嵌套管道符或中间空行。无法确认格式则整条回退。它只去除该表格的Markdown管道、分隔行、单元格边缘填充和行号，将各内容单元格转为清单。

为严格满足“不删除原文字词”，**表头词“条目”“内容”也保留为一行“条目 内容”**，不会把它们当作可以任意去掉的说明。新增字符只有连接表头词的空格与清单前缀`- `。实验规则中的任何表格均不转换。

## 接口

```python
from conservative_normalizer import normalize_messages, normalize_row

result = normalize_messages(messages, convert_persona_table=False)
normalized_messages = result["messages"]
audit = result["transformation_audit"]

# 第二种对照：再转换经严格确认的个体设定表格。
result_with_table = normalize_messages(messages, convert_persona_table=True)

# 便利接口只读取row["messages"]，不读取其他字段。
same_result = normalize_row(input_row, convert_persona_table=False)
```

返回的`messages`仅含role/content。`prompt_id`、`case_id`、`pair_id`、`variant_id`、`expected_answer`和`expected_actions`不会加入模型输入。参与者编号若原本在正文中，则原样保留；这和泄漏元数据ID是不同的事情。

`transformation_audit`记录输入/输出消息hash、原/目标模块次序、每个原始块和实际保留块的精确hash、整组保留块hash、问题hash、是否逐字不变、表格转换及被删除的原始行片段位置/hash/字符数。未知格式回退时，保存原消息hash和原因。审计不声称词面保留即可证明全部语用等价；效果仍需真实模型对照。

## CPU自测与全量审计

从项目根目录运行：

```text
python experiments/methods_v1/conservative_normalizer.py --self-test
python experiments/methods_v1/conservative_normalizer.py --audit-corpus experiments/baseline_v2 --out experiments/methods_v1/normalizer_audit
```

`--audit-corpus`也接受直接指向`inputs.jsonl`。一次统计“只重排”和“重排+严格个体表格转换”两种模式，输出：

- `corpus_summary.json`：按任务类型统计实际改变、已是目标格式、回退及表格转换覆盖。
- `transformation_audit.jsonl`：每个输入、每种模式的逐条保留证明与转换审计。

每行额外做元数据哨兵检查：改变所有题目ID及答案字段，归一化结果必须完全不变，且哨兵字符串不能进入输出messages。

2026-10-25对V2全部8,666个输入审计通过，输入字节hash为`79208ffb70a3ffe975a85df5ca5fd8072471de1bfd64b66f614dd459403f2a28`。

| 模式 | 实际改变 | 已是目标格式 | 回退 | 表格转换 |
|---|---:|---:|---:|---:|
| 只重排 | 1,994 | 6,672 | 0 | 0 |
| 重排+严格个体表格转换 | 2,991 | 5,675 | 0 | 997 |

其中自由行为1,240个输入：只重排改变286个；加入个体表格转换后改变429个。全体零回退仅说明当前已知模板均能被这套严格规则解析，不意味着未知外部文本也有相同覆盖。

自测覆盖：明确目标保留、收益表格与全部规则逐字保留、未知heading回退、重复heading回退、未知表格回退、严格表格转换与表头词保留、不修改原输入、幂等性、元数据隔离及精确保留块hash。

## 后续如何做强对照

如安排生成实验，使用与提示方法相同的模型、采样、预算和原合法JSON菜单；仅把当前messages替换为归一化结果。回退题必须保留并使用原消息，不删题、不重抽到成功。保存每个输入的transformation_audit和覆盖，分别报告真正改变题与全体题，但不能只选择变好题。

“只重排”与“重排+表格转换”应独立命名比较，不把二者效果混作一个机制。它只能处理已明确识别的模块，不统一匿名实体、不重新计算收益、不从原题库恢复规范文本。若简单对照已解释主要收益，论文应据实采用更简单的解释。
