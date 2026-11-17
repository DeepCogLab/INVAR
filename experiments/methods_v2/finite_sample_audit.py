"""Auxiliary unbiased squared-distance audit; never edits original E/G scores.

The estimands are integrated squared CDF distance and squared mean-interaction,
not W1 E/G. Negative estimates are retained. --self-test is CPU-only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

VERSION = "INVAR_finite_sample_squared_audit_v2_20261026"
DEFAULT_METHODS = ("baseline", "canonical_modules", "canonical_profile")
ALLOWED_METHODS = DEFAULT_METHODS + ("anchor_sft", "anchor_js", "anchor_kl", "anchor_js_kl")
REPETITIONS = {"behavior": 32, "changed_fact": 8, "stable_fact": 8, "goal": 16, "role": 16}


def read(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def filehash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        Path(path).write_text("", encoding="utf-8-sig")
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_samples(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("At least two finite scalar samples are required")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("Normalized behavior samples must lie in [0,1]")
    return array


def d2_unbiased(a, b, same_prompt=False):
    """Integral of empirical squared CDF difference minus sampling variance.

    Distinct prompt calls are independent; a reused prompt is one shared sample.
    For the latter the population distance and sample contrast are identically 0.
    """
    a, b = validate_samples(a), validate_samples(b)
    if same_prompt:
        if not np.array_equal(a, b):
            raise ValueError("A shared prompt must reference the same saved samples")
        return 0.0
    grid = np.unique(np.concatenate(([0.0, 1.0], a, b)))
    left = grid[:-1]
    fa = np.searchsorted(np.sort(a), left, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), left, side="right") / len(b)
    integrand = (fa - fb) ** 2 - fa * (1 - fa) / (len(a) - 1) - fb * (1 - fb) / (len(b) - 1)
    return float(integrand @ np.diff(grid))


def g2_unbiased(quartet, samples):
    """Square of D-B-C+A means, correcting after merging identical prompt IDs."""
    coefficients = collections.Counter()
    for side, coefficient in zip("ABCD", (1, -1, -1, 1)):
        coefficients[quartet[side]] += coefficient
    contrast = variance = 0.0
    for pid, coefficient in coefficients.items():
        if coefficient == 0:
            continue
        values = validate_samples(samples[pid])
        contrast += coefficient * float(values.mean())
        variance += coefficient ** 2 * float(values.var(ddof=1)) / len(values)
    return contrast ** 2 - variance, contrast, variance, dict(coefficients)


def quartet_estimates(quartet, samples):
    a, b, c, d = (quartet[side] for side in "ABCD")
    ab = d2_unbiased(samples[a], samples[b], same_prompt=a == b)
    cd = d2_unbiased(samples[c], samples[d], same_prompt=c == d)
    g2, contrast, removed, coefficients = g2_unbiased(quartet, samples)
    return dict(quartet, E2_unbiased=(ab + cd) / 2, G2_unbiased=g2,
                D2_AB_unbiased=ab, D2_CD_unbiased=cd, mean_interaction=contrast,
                G2_sampling_variance_removed=removed,
                G2_unique_prompt_coefficients=json.dumps(coefficients, sort_keys=True))


def pair_weights(rows):
    counts = collections.Counter(row["pair_id"] for row in rows)
    return np.array([1.0 / len(counts) / counts[row["pair_id"]] for row in rows])


def summarize(rows, model, method):
    groups = [("all", "all", rows)]
    for name in ("source_id", "task", "variant_id", "training_partition"):
        for value in sorted({row[name] for row in rows if name in row}):
            groups.append((name, value, [row for row in rows if row.get(name) == value]))
    result = []
    for group, value, subset in groups:
        if not subset:
            continue
        weights = pair_weights(subset)
        record = {"model": model, "method": method, "group": group, "value": value,
                  "n_pairs": len({row["pair_id"] for row in subset}), "n_quartets": len(subset),
                  "status": "complete_auxiliary_fixed_item_audit"}
        for metric in ("E2_unbiased", "G2_unbiased"):
            estimates = np.array([row[metric] for row in subset])
            record[metric] = float(weights @ estimates)
            record[metric + "_negative_quartets"] = int(np.sum(estimates < 0))
        result.append(record)
    return result


def load_helpers(baseline):
    sys.path.insert(0, str(baseline / "INVAR_dataset"))
    from evaluate import parse_response, normalize
    path = Path(os.environ.get("INVAR_METHODS_V1", Path(__file__).resolve().parent.parent / "methods_v1")) / "run_prompt_methods.py"
    spec = importlib.util.spec_from_file_location("invar_auxiliary_public_schema_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return parse_response, normalize, module.validate_output


def inspect_raw(file, model, method, full_inputs, selected, repetitions, parse_response, normalize, validate_output):
    result = {"model": model, "method": method, "status": "missing", "errors": [],
              "expected_behavior_prompts": sum(full_inputs[pid]["task_type"] == "behavior" for pid in selected),
              "behavior_repetitions_required": 32, "rows": 0}
    if not file.exists():
        return {}, result
    identity_path = file.parent / f"{method}.run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8-sig")) if identity_path.exists() else {}
    allowed = identity.get("plan", {}).get("selected_prompt_ids", list(selected))
    if not set(selected) <= set(allowed) or not set(allowed) <= full_inputs.keys():
        result["status"] = "invalid"
        result["errors"].append("own method plan does not contain frozen selected inputs")
        return {}, result
    counts, samples = collections.Counter(), collections.defaultdict(list)
    seen, configs = set(), set()
    with file.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pid, replicate = row["prompt_id"], row["replicate"]
                if pid not in allowed or type(replicate) is not int:
                    raise ValueError("unknown prompt or invalid replicate")
                prompt = full_inputs[pid]
                if not 0 <= replicate < repetitions[prompt["task_type"]]:
                    raise ValueError("replicate outside frozen plan")
                key = (pid, replicate)
                if key in seen:
                    raise ValueError("duplicate saved trial")
                seen.add(key)
                if row.get("model") != model or row.get("method") != method or row.get("batch", "main") != "main":
                    raise ValueError("mixed model/method/batch")
                if row.get("status") != "ok":
                    raise ValueError("non-ok saved response")
                validate_output(row["response_text"], prompt["output_schema"], row.get("finish_reason"))
                counts[pid] += 1
                configs.add(row["config_sha256"])
                result["rows"] += 1
                if pid in selected and prompt["task_type"] == "behavior":
                    action, error = parse_response(row["response_text"], prompt["action_space"])
                    if error:
                        raise ValueError(error)
                    value = normalize(action, prompt["outcome"])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("behavior outcome outside normalized [0,1]")
                    samples[pid].append(value)
            except (ValueError, KeyError, TypeError) as error:
                result["errors"].append(f"line {number}: {type(error).__name__}: {error}")
    expected_rows = sum(repetitions[full_inputs[pid]["task_type"]] for pid in allowed)
    result.update(expected_rows=expected_rows, file_sha256=filehash(file), configuration_hashes=sorted(configs))
    complete_counts = all(counts[pid] == repetitions[full_inputs[pid]["task_type"]] for pid in allowed)
    complete_behavior = all(len(samples[pid]) == 32 for pid in selected if full_inputs[pid]["task_type"] == "behavior")
    marker_path = file.with_suffix(".complete.json")
    marker_ok = False
    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
            marker_ok = (marker.get("status") == "complete" and marker["rows"] == expected_rows
                         and marker["responses_sha256"] == result["file_sha256"]
                         and marker["model"] == model and marker["method"] == method
                         and configs == {marker["config_sha256"]})
        except (ValueError, KeyError) as error:
            result["errors"].append("invalid completion marker: " + str(error))
    result["complete_marker_verified"] = marker_ok
    result["status"] = "complete" if complete_counts and complete_behavior and marker_ok and not result["errors"] else "partial"
    # Never analyze a complete-looking subset from a partial/invalid file.
    return (dict(samples) if result["status"] == "complete" else {}), result


def partition(source, variant, heldout):
    if source == heldout:
        return "heldout_source_" + heldout
    if variant == "V4_rules_table":
        return "heldout_variant_V4_seen_source"
    if variant == "V6_relation":
        return "heldout_variant_V6_seen_source"
    return "seen_source_seen_variant"


def run(args):
    methods = list(dict.fromkeys(args.methods.split(",")))
    if not methods or any(method not in ALLOWED_METHODS for method in methods):
        raise ValueError("This audit accepts sampled prompt/LoRA methods only; activation likelihood proxies are excluded")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if filehash(args.baseline / "inputs.jsonl") != manifest["baseline_inputs_sha256"]:
        raise ValueError("Baseline input hash changed")
    repetitions = manifest.get("repetitions", REPETITIONS)
    if repetitions.get("behavior") != 32:
        raise ValueError("This auxiliary audit is frozen to 32 behavior samples per prompt")
    full_inputs = {row["prompt_id"]: row for row in read(args.baseline / "inputs.jsonl")}
    selected = set(manifest["selected_prompt_ids"])
    quartet_ids = set(manifest["selected_quartet_ids"]["behavior"])
    quartets = [q for q in read(args.baseline / "INVAR_dataset/data/quartets.jsonl") if q["quartet_id"] in quartet_ids]
    if len(quartets) != len(quartet_ids):
        raise ValueError("Unknown or duplicate selected quartet IDs")
    if any(q[side] not in selected for q in quartets for side in "ABCD"):
        raise ValueError("Selected quartet references unselected input")
    heldout = manifest.get("learning", {}).get("heldout_source", "HPVBG")
    for quartet in quartets:
        quartet["training_partition"] = partition(quartet["source_id"], quartet["variant_id"], heldout)
    models = set(manifest.get("initial_models", []))
    if args.out.exists():
        models.update(directory.name for directory in args.out.iterdir() if directory.is_dir()
                      and any((directory / (method + ".jsonl")).exists() for method in methods))
    parse_response, normalize, validate_output = load_helpers(args.baseline)
    audits, quartet_rows, summaries, differences = [], [], [], []
    for model in sorted(models):
        model_summaries = {}
        for method in methods:
            samples, audit = inspect_raw(args.out / model / (method + ".jsonl"), model, method,
                                         full_inputs, selected, repetitions, parse_response, normalize, validate_output)
            audits.append(audit)
            if audit["status"] != "complete":
                continue
            rows = [quartet_estimates(q, samples) for q in quartets]
            quartet_rows.extend(dict(row, model=model, method=method) for row in rows)
            method_summary = summarize(rows, model, method)
            summaries.extend(method_summary)
            model_summaries[method] = {(row["group"], row["value"]): row for row in method_summary}
        baseline = model_summaries.get("baseline", {})
        for method, rows in model_summaries.items():
            if method == "baseline":
                continue
            for key, row in rows.items():
                old = baseline.get(key)
                if old is None or old["n_pairs"] != row["n_pairs"] or old["n_quartets"] != row["n_quartets"]:
                    continue
                differences.append({"model": model, "method": method, "group": key[0], "value": key[1],
                                    "n_pairs": row["n_pairs"], "n_quartets": row["n_quartets"],
                                    "E2_unbiased_method_minus_baseline": row["E2_unbiased"] - old["E2_unbiased"],
                                    "G2_unbiased_method_minus_baseline": row["G2_unbiased"] - old["G2_unbiased"],
                                    "comparison": "same_fixed_quartets_independent_sampled_methods"})
    out = args.out / "finite_sample_audit"
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "quartets.csv", quartet_rows)
    save_csv(out / "summary.csv", summaries)
    save_csv(out / "matched_differences.csv", differences)
    save_csv(out / "completion_audit.csv", [dict(row, errors=json.dumps(row["errors"], ensure_ascii=False)) for row in audits])
    payload = {"version": VERSION, "status": "auxiliary_squared_estimands_not_primary_W1",
               "declaration": "2026-10-25新增的辅助有限样本诊断；不冒称生成前预注册，不覆盖或改写冻结主指标",
               "scorer_sha256": filehash(Path(__file__)), "manifest_sha256": filehash(args.manifest),
               "methods": methods, "negative_estimates_truncated": False,
               "assumptions": ["iid responses per fixed prompt", "independent calls across distinct prompts",
                               "same prompt references reuse exactly one sample", "fixed normalized scale [0,1]"],
               "summaries": summaries, "matched_differences": differences, "completion_audit": audits}
    (out / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 有限采样辅助诊断：平方距离的无偏估计", "",
             "2026-10-25新增的辅助分析声明。主E/R/G评分、冻结输出和原scores文件均保留原样；不声称该辅助分析在响应生成之前预注册。", "",
             "32次采样下，W1等非负距离会包含采样波动，且波动随行为分布的熵变化。这里额外估计两个不同的平方总体量，检查方法表观改善是否主要来自有限采样噪声变化。它们不是原W1 E或绝对值G的纠偏版，禁止将新数值与原E/G直接相减。", "",
             "## 定义与假设", "",
             "对两个不同提示的独立样本，在归一支持[0,1]上计算：", "",
             "`D2 = ∫[(Fa−Fb)^2 − Fa(1−Fa)/(na−1) − Fb(1−Fb)/(nb−1)]dt`。", "",
             "Fa/Fb为经验CDF。由于E[Fa(1−Fa)/(na−1)]=F(1−F)/na，扣除项恰好去掉CDF平方差中两侧的抽样方差；其目标是总体CDF差平方的积分。`E2_unbiased=(D2_AB+D2_CD)/2`。相同prompt引用同一份样本时，该边的总体距离本来严格为0，直接记0，不能套独立样本式重复扣方差。", "",
             "`G2_unbiased = [(meanD−meanB)−(meanC−meanA)]² − Σ_unique_prompt c_pid² s_pid²/n_pid`，s²使用ddof=1。先合并A:+1、B:−1、C:−1、D:+1中相同prompt的系数，再计算方差；因此共享提示的协方差不会被遗漏。其目标是总体均值交互对比的平方。", "",
             "无偏结论要求每个固定提示下响应独立同分布、不同提示的随机调用相互独立、量尺固定，并正确复用同一prompt样本。温度固定并不能证明独立性；共用非随机账本解析、参数权重或固定提示本身不构成响应随机误差的重复抽样。若不同提示共享随机解析或随机适配过程，须另建协方差分析。", "",
             "估计可为负，不截断；负值只表示有限样本下估计落在0以下，不表示总体平方距离为负。平方距离强调较大CDF差，与W1的大小和排序可能不同。无偏不意味着方差小；n=32仍可能很不稳定。此处不提供零效应检验或用负值给模型排名，也不替代独立确认与大重复实验。", "",
             "## 已完成方法的辅助结果", "",
             "先在条件对内对变体等权，再对条件对等权。只在全冻结计划合法完成、完成标记哈希正确、每behavior提示恰好32条时纳入；partial不挑完整四元组子集汇总。", "",
             "| 模型 | 方法 | 条件对 | E2无偏估计 | G2无偏估计 |", "|---|---|---:|---:|---:|"]
    for row in summaries:
        if row["group"] == "all":
            lines.append(f"| {row['model']} | {row['method']} | {row['n_pairs']} | {row['E2_unbiased']:.6f} | {row['G2_unbiased']:.6f} |")
    lines += ["", "## 对匹配baseline的差值", "",
              "差值仅在同样平方estimand之间计算：方法−新采样baseline。负值表示估计距离减少；不能单独认定改善成立。须结合原E/G、U/Q、熵与独立复测。LoRA总体包含训练接触，迁移判断请看training_partition分项。", "",
              "| 模型 | 方法 | ΔE2 | ΔG2 |", "|---|---|---:|---:|"]
    for row in differences:
        if row["group"] == "all":
            lines.append(f"| {row['model']} | {row['method']} | {row['E2_unbiased_method_minus_baseline']:+.6f} | {row['G2_unbiased_method_minus_baseline']:+.6f} |")
    lines += ["", "## 完成性", "", "| 模型 | 方法 | 状态 | 响应数 |", "|---|---|---|---:|"]
    for row in audits:
        lines.append(f"| {row['model']} | {row['method']} | {row['status']} | {row['rows']} |")
    lines += ["", "逐四元组及来源/任务/变体/训练接触分项见本目录CSV。没有读入HF似然或激活代理结果。此辅助指标的定义与代码哈希单独保存，主评分未被修改。", ""]
    (out / "说明与结果.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"version": VERSION, "status": "auxiliary_audited", "complete": sum(a["status"] == "complete" for a in audits),
                      "summaries": len(summaries), "matched_differences": len(differences)}, ensure_ascii=False))


def binomial_probability(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def bernoulli_samples(n, k):
    return np.concatenate((np.zeros(n - k), np.ones(k)))


def exact_d2_expectation(pa, pb, na=4, nb=3):
    answer = 0.0
    for ka, kb in itertools.product(range(na + 1), range(nb + 1)):
        probability = binomial_probability(na, ka, pa) * binomial_probability(nb, kb, pb)
        answer += probability * d2_unbiased(bernoulli_samples(na, ka), bernoulli_samples(nb, kb))
    return answer


def exact_g2_expectation(probabilities, quartet, n=3):
    pids = sorted(probabilities)
    answer = 0.0
    for counts in itertools.product(range(n + 1), repeat=len(pids)):
        probability = math.prod(binomial_probability(n, k, probabilities[pid]) for pid, k in zip(pids, counts))
        samples = {pid: bernoulli_samples(n, k) for pid, k in zip(pids, counts)}
        answer += probability * g2_unbiased(quartet, samples)[0]
    return answer


def self_test():
    equal = exact_d2_expectation(.2, .2)
    different = exact_d2_expectation(.2, .8)
    if abs(equal) > 1e-12 or abs(different - .36) > 1e-12:
        raise AssertionError((equal, different))
    constant = np.zeros(32)
    if d2_unbiased(constant, constant) != 0 or d2_unbiased(constant, np.ones(32)) != 1:
        raise AssertionError("Constant sample distances incorrect")
    same = np.array([0, .5, 1, .5])
    if d2_unbiased(same, same, same_prompt=True) != 0:
        raise AssertionError("Shared-prompt distance must be zero")
    if d2_unbiased(same, same) >= 0:
        raise AssertionError("Independent empirical estimate may be negative and must not be clipped")
    quartet = dict(A="a", B="b", C="c", D="d")
    equal_g = exact_g2_expectation(dict(a=.2, b=.2, c=.2, d=.2), quartet)
    changed_g = exact_g2_expectation(dict(a=.2, b=.2, c=.8, d=.2), quartet)
    if abs(equal_g) > 1e-12 or abs(changed_g - .36) > 1e-12:
        raise AssertionError((equal_g, changed_g))
    shared_q = dict(A="a", B="a", C="b", D="b")
    value, contrast, variance, coefficients = g2_unbiased(shared_q, dict(a=same, b=same[::-1]))
    if value != 0 or variance != 0 or contrast != 0 or any(coefficients.values()):
        raise AssertionError("Shared coefficients failed to cancel")
    doubled_q = dict(A="a", B="b", C="b", D="a")
    if abs(exact_g2_expectation(dict(a=.2, b=.8), doubled_q) - 1.44) > 1e-12:
        raise AssertionError("Shared coefficients must be merged before squaring")
    weights = pair_weights([{"pair_id": "p1"}, {"pair_id": "p1"}, {"pair_id": "p2"}])
    if not np.allclose(weights, [.25, .25, .5]):
        raise AssertionError("Pair-equal aggregation failed")
    # Full independent n=32 sample simulation: mean estimates must recover the
    # population estimands within 5 Monte Carlo standard errors.
    rng = np.random.default_rng(20261025)
    trials, n = 50000, 32
    phat = rng.binomial(n, [.2, .2, .8, .2], size=(trials, 4)) / n
    variance = phat * (1 - phat) / (n - 1)
    d_ab = (phat[:, 0] - phat[:, 1]) ** 2 - variance[:, 0] - variance[:, 1]
    d_cd = (phat[:, 2] - phat[:, 3]) ** 2 - variance[:, 2] - variance[:, 3]
    e2 = (d_ab + d_cd) / 2
    g2 = (phat @ np.array([1, -1, -1, 1])) ** 2 - variance.sum(axis=1)
    for simulated, expected in ((e2, .18), (g2, .36)):
        se = simulated.std(ddof=1) / np.sqrt(trials)
        if abs(simulated.mean() - expected) > 5 * se:
            raise AssertionError((float(simulated.mean()), expected, float(se)))
    # At the null the positive plug-in term has positive noise; corrected mean 0.
    null = rng.binomial(n, .5, size=(trials, 2)) / n
    naive = (null[:, 0] - null[:, 1]) ** 2
    corrected = naive - (null * (1 - null)).sum(axis=1) / (n - 1)
    if naive.mean() <= 0 or abs(corrected.mean()) > 5 * corrected.std(ddof=1) / np.sqrt(trials):
        raise AssertionError("Null finite-sample simulation failed")
    print(json.dumps({"self_test": "passed", "version": VERSION,
                      "exact_D2_equal": equal, "exact_D2_point2_point8": different,
                      "exact_G2_equal": equal_g, "exact_G2_changed": changed_g,
                      "n32_simulated_E2": float(e2.mean()), "n32_simulated_G2": float(g2.mean()),
                      "null_naive_squared_CDF_mean": float(naive.mean()), "null_unbiased_D2_mean": float(corrected.mean()),
                      "checks": ["exact_binomial_enumeration", "constant_distributions", "negative_retained",
                                 "shared_prompt_D2_zero", "merged_G2_coefficients", "pair_weights", "n32_full_sampling_simulation"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.baseline, args.manifest, args.out)):
        parser.error("--baseline --manifest --out required unless --self-test")
    run(args)


if __name__ == "__main__":
    main()
