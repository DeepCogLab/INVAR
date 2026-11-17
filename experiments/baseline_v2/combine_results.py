"""Render audited INVAR V2 reports; never runs models or changes frozen files.

Run this file in either the local or remote campaign folder. Only eight passed,
matching completion audits and no FAILED marker permit a final-result title.
Until then, only a clearly labelled interim report is written. Standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

MODEL_NAMES = {
    "qwen3_8b": "Qwen3-8B", "qwen25_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen25_7b_base": "Qwen2.5-7B Base（普通续写诊断）", "qwen25_14b_instruct": "Qwen2.5-14B-Instruct",
    "glm4_9b": "GLM-4-9B-0414", "llama31_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral7b_instruct_v03": "Mistral-7B-Instruct-v0.3", "deepseek_llm7b_chat": "DeepSeek-LLM-7B-Chat",
}
TASK_NAMES = {
    "dictator": "独裁者分配", "ultimatum": "最后通牒", "trust": "信任与返还",
    "sequential_prisoners_dilemma": "序贯合作", "public_goods": "公共物品",
    "third_party_punishment": "第三方惩罚",
}
VARIANT_NAMES = {
    "V1_paragraph": "个体设定改自然段", "V1_table": "个体设定改表格",
    "V2_rules_first": "规则提前", "V2_situation_first": "当前情况提前",
    "V3_situation_list": "情境改清单", "V4_rules_table": "规则改表格",
    "V5_ids": "匿名编号替换", "V6_relation": "关系对应表述",
}
POLICY_NAMES = {"own": "最大化本人点数", "total": "最大化全体总点数", "equality": "最小化最高与最低点数差"}


def read_json(path):
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, text):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {key}")
    return value


def cell(row, metric, percentage=False, empirical=False):
    if row is None:
        return "待评分"
    scale = 100 if percentage else 1
    fmt = (lambda x: f"{x * scale:.1f}%") if percentage else (lambda x: f"{x:.4f}")
    point = fmt(number(row, metric))
    low, high = metric + "_mc_low", metric + "_mc_high"
    if metric == "accuracy" and not empirical and "accuracy_bound95_low" in row:
        low, high = "accuracy_bound95_low", "accuracy_bound95_high"
    if low in row and high in row:
        point += f" [{fmt(number(row, low))}, {fmt(number(row, high))}]"
    return point


def table(lines, headers, rows):
    lines += ["", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |")
    if not rows:
        lines.append("| " + " | ".join(["尚无通过验收的结果"] + ["—"] * (len(headers) - 1)) + " |")


def index_rows(rows):
    result = {}
    for row in rows:
        key = row["group"], row["value"]
        if key in result:
            raise ValueError(f"duplicate summary group {key}")
        result[key] = row
    return result


def inspect(root, verify_raw=True):
    """No partial or malformed score table is promoted into a final report."""
    manifest = read_json(root / "frozen_manifest.json")
    models = manifest["models"]
    global_errors = []
    if not verify_raw:
        global_errors.append("本次跳过原始文件哈希复核，仅允许阶段报告")
    if len(models) != 8 or len(set(models)) != 8:
        global_errors.append("冻结模型列表不是八个不同模型")
    failures = sorted(str(p.relative_to(root)) for p in root.rglob("*FAILED*") if p.is_file())
    if failures:
        global_errors.append("存在 FAILED 标记：" + ", ".join(failures))
    verified_files = {}
    for filename, field in (("inputs.jsonl", "input_sha256"), ("main_plan.jsonl", "plan_sha256")):
        path = root / filename
        if path.exists():
            verified_files[filename] = sha256(path) == manifest[field]
            if not verified_files[filename]:
                global_errors.append(filename + " 与冻结哈希不一致")
        else:
            global_errors.append(filename + " 缺失，不能核验冻结输入或计划")
    reports = {}
    records = []
    for model in models:
        folder = root / "reports" / model
        errors = []
        audit = {}
        loaded = {}
        path = folder / "completion_audit.json"
        if not path.exists():
            errors.append("评分验收尚未生成")
        else:
            try:
                audit = read_json(path)
                for ok, reason in (
                    (audit.get("model") == model, "模型身份不匹配"),
                    (audit.get("status") == "passed", "评分验收未通过"),
                    (audit.get("input_sha256") == manifest["input_sha256"], "验收输入哈希不匹配"),
                    (audit.get("expected") == manifest["rows_per_model"], "验收计划数不匹配"),
                    (audit.get("response_rows") == manifest["rows_per_model"], "响应数量不匹配"),
                    (audit.get("counts") == manifest["rows_by_batch_per_model"], "批次数量不匹配"),
                    (audit.get("interface_failures") == 0, "接口存在失败或未记载"),
                    (audit.get("truncated") == 0, "存在截断或未记载"),
                    (len(audit.get("configuration_sha256", [])) == 1, "配置不唯一"),
                    (len(audit.get("raw_sha256", "")) == 64, "原始文件哈希缺失"),
                ):
                    if not ok:
                        errors.append(reason)
                raw = root / "outputs" / "main" / (model + ".jsonl")
                if verify_raw and not raw.exists():
                    errors.append("原始响应文件尚未归档，不能完成哈希复核")
                if verify_raw and raw.exists() and sha256(raw) != audit.get("raw_sha256"):
                    errors.append("现存原始响应与评分验收哈希不匹配")
                identity_path = raw.with_suffix('.identity.json')
                if not identity_path.exists():
                    errors.append("模型身份与实际渲染输入记录缺失")
                else:
                    identity = read_json(identity_path)
                    if identity.get('config_sha256') not in audit.get('configuration_sha256', []):
                        errors.append("身份记录与验收配置不一致")
                    if identity.get('config', {}).get('inputs_sha256') != manifest['input_sha256']:
                        errors.append("身份记录与冻结输入不一致")
                    if not identity.get('resolved_weights') or len(identity.get('weight_config_sha256', '')) != 64 or len(identity.get('rendered_sha256', '')) != 64:
                        errors.append("权重或渲染输入证据不完整")
                for name in ("behavior_summary", "control_summary", "role_checks", "same_prompt_repeat"):
                    loaded[name] = read_csv(folder / (name + ".csv"))
                    if not loaded[name]:
                        errors.append(name + " 为空")
                bi = index_rows(loaded["behavior_summary"])
                ci = index_rows(loaded["control_summary"])
                all_row = bi["all", "all"]
                if int(all_row["n_pairs"]) != manifest["condition_pairs"] or int(all_row["n_quartets"]) != manifest["behavior_quartets"]:
                    errors.append("主分析没有覆盖全部冻结条件对与四元组")
                for row in loaded["behavior_summary"]:
                    for metric in ("E", "R", "G"):
                        for key in (metric, metric + "_mc_low", metric + "_mc_high"):
                            if number(row, key) < -1e-9:
                                errors.append("行为指标出现负数")
                        if number(row, metric + "_mc_low") > number(row, metric + "_mc_high"):
                            errors.append("行为区间上下界颠倒")
                for kind in ("changed_fact", "stable_fact", "goal"):
                    if ("type", kind) not in ci:
                        errors.append(kind + " 汇总缺失")
                for policy, count in manifest["goal_switch_pairs"].items():
                    if int(ci["goal_switch", policy]["n_pairs"]) != count:
                        errors.append(policy + " 最优行动切换子集大小不匹配")
                for row in loaded["control_summary"]:
                    for key in ("accuracy_bound95_low", "accuracy_bound95_high"):
                        if not 0 <= number(row, key) <= 1:
                            errors.append("正确率Hoeffding界不在[0,1]")
                    for metric in ("accuracy", "worst_side", "dual_original", "dual_variant", "min_original", "min_variant"):
                        for key in (metric, metric + "_mc_low", metric + "_mc_high"):
                            if not -1e-9 <= number(row, key) <= 1 + 1e-9:
                                errors.append("控制正确率不在[0,1]")
                for row in loaded["role_checks"]:
                    if not 0 <= int(row["correct"]) <= int(row["n"]):
                        errors.append("角色计数不合法")
                if sum(int(r["n"]) for r in loaded["role_checks"]) != manifest["rows_by_batch_per_model"]["role"]:
                    errors.append("角色响应数不匹配")
                if len(loaded['role_checks']) != manifest['inputs_by_type']['role']:
                    errors.append('角色唯一输入数不匹配')
                if len(loaded['same_prompt_repeat']) != manifest['repeat_prompts']:
                    errors.append('同提示重复控制输入数不匹配')
                loaded["bi"], loaded["ci"] = bi, ci
                loaded["role_summary"] = read_json(folder / "role_summary.json")
                for key in ("accuracy", "accuracy_bound95_low", "accuracy_bound95_high"):
                    if not 0 <= number(loaded["role_summary"], key) <= 1:
                        errors.append("角色正确率Hoeffding界不在[0,1]")
                macro_path = folder / "task_macro.json"
                loaded["macro"] = read_json(macro_path) if macro_path.exists() else {}
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append("评分文件未就绪或无法解析：" + str(exc))
        if not errors:
            reports[model] = loaded
        records.append({"model": model, "accepted": not errors, "errors": errors, "audit": audit})
    complete = len(reports) == 8 and not global_errors
    return manifest, reports, records, global_errors, verified_files, complete


def behavior_detail(lines, reports, models, group, title, labels=None):
    lines += ["", "## " + title]
    keys = sorted({value for data in reports.values() for g, value in data["bi"] if g == group})
    rows = []
    for value in keys:
        for model in models:
            row = reports.get(model, {}).get("bi", {}).get((group, value))
            if row is None:
                continue
            label = (labels or {}).get(value, value)
            rows.append([label, MODEL_NAMES.get(model, model), row["n_pairs"], row["n_quartets"], cell(row, "E"), cell(row, "R"), cell(row, "G")])
    table(lines, ["分组", "模型", "条件对", "四元组", "E↓", "R（无优劣方向）", "G↓"], rows)


def control_detail(lines, reports, models, group, title, labels=None):
    lines += ["", "## " + title]
    keys = sorted({value for data in reports.values() for g, value in data["ci"] if g == group})
    rows = []
    for value in keys:
        for model in models:
            row = reports.get(model, {}).get("ci", {}).get((group, value))
            if row is None:
                continue
            label = (labels or {}).get(value, value)
            rows.append([label, MODEL_NAMES.get(model, model), row["n_pairs"], cell(row, "accuracy", True), cell(row, "worst_side", True)])
    table(lines, ["分组", "模型", "条件对", "平均正确率↑", "四侧最小正确率的汇总↑"], rows)


def render(root, manifest, reports, records, global_errors, complete):
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8")
    models = manifest["models"]
    lines = ["# INVAR 修订版" + ("最终结果" if complete else "阶段结果"), "", "生成时间：" + now + "。"]
    if complete:
        lines += ["", f"八模型评分完成验收通过，合计{manifest['plan_rows']:,}条冻结正式响应，每模型{manifest['rows_per_model']:,}条；先导单独保存。没有 FAILED 标记。此处的“最终”指冻结运行的完整性验收，不能解释为全部科研解释均已得到证明。"]
    else:
        lines += ["", f"**尚未完成最终验收：目前{len(reports)}/8个模型的完整评分文件通过检查。下表只包含已通过本地检查的评分，不能用于完整八模型比较或替代最终结果。**"]
    lines += ["", "## 完成状态"]
    table(lines, ["模型", "可核对响应/计划", "验收", "原因"], [
        [MODEL_NAMES.get(r["model"], r["model"]), f"{r['audit'].get('response_rows', '待验收')}/{manifest['rows_per_model']:,}", "通过" if r["accepted"] else "未通过/待完成", "；".join(r["errors"]) or "无"] for r in records
    ])
    if global_errors:
        lines += ["", "全局验收未通过：" + "；".join(global_errors) + "。"]
    lines += ["", "## 如何解读数值", "",
        "主分析合并全部92组条件对：先在条件对内对适用变体等权，再对条件对等权。数据来自四项原研究，覆盖六类任务；独裁者仅有1组条件对。来源与新增条件的记录保留用于追溯，来源敏感性见后文。不同细分组的题目构成不同，不能直接推断哪种机制更难。",
        "",
        "E=[W1(A,B)+W1(C,D)]/2，检验等价呈现下的行为分布差异；R=[W1(A,C)+W1(B,D)]/2，描述社会条件改变后的行为响应；G=|(mean(C)−mean(A))−(mean(D)−mean(B))|，描述条件效应是否依赖呈现。E/G越小越稳定；R没有越大越好的含义，自由社会选择保持不变也可能合理。G≤2E，三项不是独立能力的总分。",
        "",
        "行为E/R/G及最差侧、双调用等非线性统计量的方括号为1000次经验自助抽样的2.5%—97.5%分位范围。每个唯一提示每轮仅重采样一次，各四元组复用同一份抽样，保留共享输入依赖。这是固定题目、当前经验分布下的采样区间，不是新情境或社会研究总体区间。E/R/G含绝对值，零点附近有非光滑和边界问题，区间下界大于0不能充当显著性检验。观察全对/全错时经验自助区间可能退化，不代表真实正确概率已确定为100%/0%。",
        "",
        "事实、目标和角色平均正确率的主要方括号采用双侧95% Hoeffding保守界；把共享提示的最终汇总系数合并后计算。该界在采样输出相互独立、数据与汇总权重固定的条件下对单个预先给定汇总有效，即使样本全对仍不退化；不是所有细分组同时95%保证，也不能推广到新情境。正确率的经验自助区间仅在附录列示。",
        "",
        "事实正确率检验独立问题的规则真值；“四侧最小”是每个四元组先取A/B/C/D四个估计正确率的最小值，再按条件对汇总，并非全数据最小值或实测联合正确率。事实选项两个顺序均测试并等权汇总。角色正确率只检查编号绑定。明示目标执行另行独立调用，不把自由社会行为重标为目标答案。",
        "",
        "本轮评估统一结构化解码接口下的模型行为。合法JSON和行动菜单受到接口约束，格式无失败是技术验收，不能算作模型理解能力。事实和目标题的错误候选仍可输出。Base使用普通续写，需单列解释。当前审核为程序与AI辅助审核，不宣称独立真人专家认证，也不作人类行为对齐或已排除所有干扰的声明。",
        "", "## 主合并行为结果"]
    rows = []
    for model in models:
        row = reports.get(model, {}).get("bi", {}).get(("all", "all"))
        rows.append([MODEL_NAMES.get(model, model), row["n_pairs"] if row else "—", cell(row, "E"), cell(row, "R"), cell(row, "G")])
    table(lines, ["模型", "条件对", "E↓", "R（无优劣方向）", "G↓"], rows)
    lines += ["", "## 社会事实更新与保持"]
    table(lines, ["模型", "应变事实正确率↑", "应变四侧最小↑", "未变事实正确率↑", "未变四侧最小↑"], [
        [MODEL_NAMES.get(m, m), cell(reports.get(m, {}).get("ci", {}).get(("type", "changed_fact")), "accuracy", True), cell(reports.get(m, {}).get("ci", {}).get(("type", "changed_fact")), "worst_side", True), cell(reports.get(m, {}).get("ci", {}).get(("type", "stable_fact")), "accuracy", True), cell(reports.get(m, {}).get("ci", {}).get(("type", "stable_fact")), "worst_side", True)] for m in models
    ])
    lines += ["", "二选一机会水平为50%，仅作为参照；高分证明这些事实题被正确回答，不能直接证明自由决策使用了该事实。事实题每个具体选项顺序只有8次，局部估计粗糙。"]
    control_detail(lines, reports, models, "goal_policy", "明示目标执行：各政策", POLICY_NAMES)
    lines += ["", "## 唯一最优行动切换子集", "",
        "这是两条件都只有一个最优行动、且正确行动必须改变的窄子集。冻结规模为：" + "；".join(POLICY_NAMES.get(p, p) + f" {n}组条件对" for p, n in manifest["goal_switch_pairs"].items()) + "。其结果不能推广为所有社会干预的响应正确率。每题16次。",
        "",
        "下表“双调用均正确”是pA×pC或pB×pD的插件估计，表示独立抽取两个调用同时正确的估计概率；并非同一智能体轨迹或实测配对转变率。“双条件最差”为min(pA,pC)或min(pB,pD)。"]
    switch_rows = []
    for policy in manifest["goal_switch_pairs"]:
        for m in models:
            row = reports.get(m, {}).get("ci", {}).get(("goal_switch", policy))
            if row:
                switch_rows.append([POLICY_NAMES.get(policy, policy), MODEL_NAMES.get(m, m), row["n_pairs"], cell(row, "accuracy", True), cell(row, "min_original", True), cell(row, "min_variant", True), cell(row, "dual_original", True), cell(row, "dual_variant", True)])
    table(lines, ["目标", "模型", "条件对", "平均正确率↑", "原呈现双条件最差↑", "变体双条件最差↑", "原呈现双调用均正确估计↑", "变体双调用均正确估计↑"], switch_rows)
    lines += ["", "## 角色编号识别"]
    role_rows = []
    for m in models:
        data = reports.get(m)
        if not data:
            continue
        rows = data["role_checks"]
        variants = [("all", rows)] + [(v, [x for x in rows if x["variant_id"] == v]) for v in sorted({x["variant_id"] for x in rows})]
        for variant, selected in variants:
            good, n = sum(int(x["correct"]) for x in selected), sum(int(x["n"]) for x in selected)
            accuracy = good / n
            # Each role input has the same frozen replication count. Compute the
            # same fixed-prompt pointwise Hoeffding bound for each variant table.
            if len({int(x['n']) for x in selected}) != 1:
                raise ValueError('unequal role replication counts; revise weights explicitly')
            radius = math.sqrt(.5 * math.log(40) * sum((1 / len(selected)) ** 2 / int(x['n']) for x in selected))
            summary = dict(accuracy=accuracy, accuracy_bound95_low=max(0, accuracy-radius), accuracy_bound95_high=min(1, accuracy+radius))
            role_rows.append([MODEL_NAMES.get(m, m), VARIANT_NAMES.get(variant, variant), len(selected), f"{good}/{n}", cell(summary, 'accuracy', True)])
    table(lines, ["模型", "呈现", "唯一输入", "正确/次数", "正确率↑"], role_rows)
    lines += ["", "角色表为冻结控制输入上的计数描述，不因全对而推断所有角色任务均已解决。"]
    behavior_detail(lines, reports, models, "task", "行为结果按社会任务细分", TASK_NAMES)
    behavior_detail(lines, reports, models, "variant_id", "行为结果按呈现操作细分", VARIANT_NAMES)
    behavior_detail(lines, reports, models, "intervention_kind", "行为结果按干预类型细分")
    behavior_detail(lines, reports, models, "role", "行为结果按角色细分")
    for typ, label in (("changed_fact", "应变事实"), ("stable_fact", "未变事实")):
        for group, title, labels in (("task", "任务", TASK_NAMES), ("variant_id", "呈现", VARIANT_NAMES), ("factor", "干预因子", None)):
            control_detail(lines, reports, models, typ + "_" + group, label + "按" + title + "细分", labels)
    control_detail(lines, reports, models, "goal_task", "明示目标执行按任务细分", TASK_NAMES)
    control_detail(lines, reports, models, "goal_variant_id", "明示目标执行按呈现细分", VARIANT_NAMES)
    behavior_detail(lines, reports, models, "source_id", "行为结果按来源追溯")
    behavior_detail(lines, reports, models, "leave_source_out", "留一来源敏感性：分组名表示被删除的来源")
    behavior_detail(lines, reports, models, "sensitivity", "去除关系改述的敏感性分析")
    lines += ["", "## 六任务宏平均与同提示重复参照", "",
        "六任务宏平均给每项任务相同权重，只作补充；独裁者只有1组条件对。宏平均文件只有点估计，不拼接任务区间来冒充宏平均区间。同提示重复W1只来自固定的127个分层输入，不能与全量E机械相减，也没有统一越界阈值。"]
    repeat_rows = []
    for m in models:
        data = reports.get(m)
        if not data:
            continue
        macro = data["macro"]
        dists = [number(x, "W1") for x in data["same_prompt_repeat"]]
        repeat_rows.append([MODEL_NAMES.get(m, m), *[f"{float(macro[k]):.4f}" if k in macro else "未生成" for k in ("E", "R", "G")], len(dists), f"{sum(dists)/len(dists):.4f}"])
    table(lines, ["模型", "宏平均E", "宏平均R", "宏平均G", "重复提示数", "重复W1均值"], repeat_rows)
    lines += ["", "## 附录：正确率的经验自助抽样区间", "",
        "以下仅列整体事实、目标政策与唯一最优切换子集。正式解释优先采用前文Hoeffding保守界；此处区间可能因全对样本而退化。完整分组经验区间在control_summary.csv中。"]
    empirical_rows = []
    for m in models:
        data = reports.get(m)
        if not data:
            continue
        for (group, value), row in sorted(data['ci'].items()):
            if group not in ('type', 'goal_policy', 'goal_switch'):
                continue
            empirical_rows.append([MODEL_NAMES.get(m, m), group, POLICY_NAMES.get(value, value), row['n_pairs'], cell(row, 'accuracy', True, empirical=True)])
    table(lines, ['模型', '汇总', '分组', '条件对', '平均正确率及经验区间'], empirical_rows)
    lines += ["", "## 复算与使用范围", "",
        "逐模型reports目录保留behavior_quartets.csv、behavior_summary.csv、control_quartets.csv、control_summary.csv、role_checks.csv与same_prompt_repeat.csv；completion_audit.json记载冻结输入、配置、批次计数与原始文件哈希。score_v2.py是冻结评分入口，本合并脚本只渲染结果。各分组中缺少条目表示未生成/不适用，不将其填作0。",
        "",
        "不要仅依据E/R非零宣布显著；不要将R较大称为社会机制更正确；不要将自助区间用于来源总体推断；不要把事实独立调用与主行为串成多轮轨迹。正确率下降可能反映规则阅读、计算、指令执行等基础环节，不能直接归因为独有社会心理机制。最终科研文字仍需结合可复现案例及适用范围解释。",
        "",
        "冻结输入SHA-256：`" + manifest["input_sha256"] + "`。",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--skip-raw-hash", action="store_true", help="Skip hashing raw files when present; report audits are still mandatory")
    args = ap.parse_args()
    root = args.root.resolve()
    manifest, reports, records, errors, verified, complete = inspect(root, verify_raw=not args.skip_raw_hash)
    filename = "最终结果.md" if complete else "阶段结果.md"
    atomic_write(root / filename, render(root, manifest, reports, records, errors, complete))
    result = {"status": "complete" if complete else "interim", "file": filename, "accepted_models": len(reports), "expected_models": 8, "errors": errors,
              "models": [{k: v for k, v in r.items() if k != "audit"} for r in records], "frozen_files_verified": verified, "raw_hash_check_enabled": not args.skip_raw_hash}
    atomic_write(root / "combine_audit.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
