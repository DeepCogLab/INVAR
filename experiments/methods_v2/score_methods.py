"""Score sampled prompt-method pilots, keeping proxy/activation results separate.

Shared-prompt response bootstrap ranges describe fixed-item sampling only.
They are not zero-effect tests or intervals over unseen social settings.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("baseline", "canonical_modules", "canonical_profile")
LORA_METHODS = ("anchor_sft", "anchor_js", "anchor_kl", "anchor_js_kl")
SCORER_VERSION = "INVAR_normalization_anchor_score_v1_20261026"
REPETITIONS = {"behavior": 32, "changed_fact": 8, "stable_fact": 8, "goal": 16, "role": 16}


def training_partition(source_id: str, variant_id: str | None, heldout_source: str = "HPVBG") -> str:
    if source_id == heldout_source:
        return "heldout_source_" + heldout_source
    if variant_id == "V4_rules_table":
        return "heldout_variant_V4_seen_source"
    if variant_id == "V6_relation":
        return "heldout_variant_V6_seen_source"
    return "seen_source_seen_variant"


def partition_exposure(partition: str) -> str:
    if partition.startswith("heldout_source_"):
        return "该来源材料未用于适配器训练；仍是已观察数据的回溯开发迁移"
    if partition.startswith("heldout_variant_"):
        return "该变体未用于训练；来源与参考侧A/C可能已见，非全四侧未见"
    if partition == "seen_source_seen_variant":
        return "来源/变体已见；事实、目标及自由行为约束可能训练接触，不是未见迁移"
    return "混合训练接触集合，只作开发描述，不是未见迁移"


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def filehash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    if not keys:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_helpers(baseline: Path):
    sys.path.insert(0, str(baseline / "INVAR_dataset"))
    from evaluate import parse_response, normalize
    runner_path = Path(os.environ.get("INVAR_METHODS_V1", Path(__file__).resolve().parent.parent / "methods_v1")) / "run_prompt_methods.py"
    spec = importlib.util.spec_from_file_location("invar_prompt_method_validation", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return parse_response, normalize, runner.validate_output


def weights_for(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = collections.Counter(row["pair_id"] for row in rows)
    return np.array([1.0 / len(counts) / counts[row["pair_id"]] for row in rows])


def w1(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.union1d(a, b)
    if len(grid) < 2:
        return 0.0
    fa = np.searchsorted(np.sort(a), grid[:-1], side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid[:-1], side="right") / len(b)
    return float(np.abs(fa - fb) @ np.diff(grid))


class Samples:
    """One empirical resampling per unique prompt; all quartet uses reuse it."""
    def __init__(self, values: dict[str, list[float]], draws: int, seed: int):
        self.draws = draws
        self.items = {}
        self.edges = {}
        rng = np.random.default_rng(seed)
        for pid, values_at_prompt in sorted(values.items()):
            support, counts = np.unique(values_at_prompt, return_counts=True)
            n = len(values_at_prompt)
            probabilities = counts / n
            boot = rng.multinomial(n, probabilities, size=draws) / n
            self.items[pid] = (support, probabilities, np.cumsum(boot, axis=1), boot @ support)

    def mean(self, pid: str) -> tuple[float, np.ndarray]:
        support, probabilities, _, boot_means = self.items[pid]
        return float(probabilities @ support), boot_means

    def distance(self, a: str, b: str) -> tuple[float, np.ndarray]:
        edge = tuple(sorted((a, b)))
        if edge in self.edges:
            return self.edges[edge]
        xa, pa, ca, _ = self.items[a]
        xb, pb, cb, _ = self.items[b]
        grid = np.union1d(xa, xb)
        if len(grid) < 2:
            answer = (0.0, np.zeros(self.draws))
        else:
            ia = np.searchsorted(xa, grid[:-1], side="right") - 1
            ib = np.searchsorted(xb, grid[:-1], side="right") - 1
            ma, mb = np.maximum(ia, 0), np.maximum(ib, 0)
            fa = np.where(ia >= 0, np.cumsum(pa)[ma], 0.0)
            fb = np.where(ib >= 0, np.cumsum(pb)[mb], 0.0)
            ba = np.where(ia[None, :] >= 0, ca[:, ma], 0.0)
            bb = np.where(ib[None, :] >= 0, cb[:, mb], 0.0)
            widths = np.diff(grid)
            answer = (float(np.abs(fa - fb) @ widths), np.abs(ba - bb) @ widths)
        self.edges[edge] = answer
        return answer


def compute_quartets(behavior_qs: list[dict[str, Any]], control_qs: list[dict[str, Any]],
                     values: dict[str, list[float]], draws: int, seed: int):
    sample = Samples(values, draws, seed)
    points, boots = [], {}
    for quartet in behavior_qs:
        if not all(quartet[side] in sample.items for side in "ABCD"):
            continue
        a, b, c, d = (quartet[side] for side in "ABCD")
        ab, bab = sample.distance(a, b)
        cd, bcd = sample.distance(c, d)
        ac, bac = sample.distance(a, c)
        bd, bbd = sample.distance(b, d)
        ma, ba = sample.mean(a)
        mb, bb = sample.mean(b)
        mc, bc = sample.mean(c)
        md, bd_mean = sample.mean(d)
        point = {"E": (ab + cd) / 2, "R": (ac + bd) / 2,
                 "G": abs((md - mb) - (mc - ma)),
                 "delta_original": mc - ma, "delta_variant": md - mb}
        row = dict(quartet, kind="behavior", **point)
        points.append(row)
        boots[("behavior", quartet["quartet_id"])] = {
            "E": (bab + bcd) / 2, "R": (bac + bbd) / 2,
            "G": np.abs((bd_mean - bb) - (bc - ba)),
            "delta_original": bc - ba, "delta_variant": bd_mean - bb}
    for quartet in control_qs:
        if not all(quartet[side] in sample.items for side in "ABCD"):
            continue
        means = [sample.mean(quartet[side]) for side in "ABCD"]
        p = np.array([x[0] for x in means])
        boot = np.stack([x[1] for x in means])
        kind = quartet.get("probe_type", "goal")
        points.append(dict(quartet, kind=kind, accuracy=float(p.mean()), worst_side=float(p.min()),
                           min_original=float(min(p[0], p[2])), min_variant=float(min(p[1], p[3])),
                           dual_original=float(p[0] * p[2]), dual_variant=float(p[1] * p[3])))
        boots[(kind, quartet["quartet_id"])] = {
            "accuracy": boot.mean(axis=0), "worst_side": boot.min(axis=0),
            "min_original": np.minimum(boot[0], boot[2]), "min_variant": np.minimum(boot[1], boot[3]),
            "dual_original": boot[0] * boot[2], "dual_variant": boot[1] * boot[3]}
    return points, boots, sample


def group_rows(rows: list[dict[str, Any]]):
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        yield kind, "all", "all", subset
        for key in ("training_partition", "source_id", "task", "variant_id", "factor", "policy", "option_order_version"):
            for value in sorted({str(row[key]) for row in subset if key in row}):
                yield kind, key, value, [row for row in subset if str(row.get(key)) == value]
        if kind == "goal":
            for policy in sorted({row["policy"] for row in subset if row.get("unique_optimum_switch")}):
                yield kind, "goal_switch", policy, [row for row in subset if row.get("policy") == policy and row.get("unique_optimum_switch")]


def summarize(points: list[dict[str, Any]], boots: dict, model: str, method: str, status: str):
    summaries, summary_boot = [], {}
    for kind, group, value, subset in group_rows(points):
        weights = weights_for(subset)
        metrics = ("E", "R", "G", "delta_original", "delta_variant") if kind == "behavior" else (
            "accuracy", "worst_side", "min_original", "min_variant", "dual_original", "dual_variant")
        for metric in metrics:
            point = float(weights @ np.array([row[metric] for row in subset]))
            resamples = weights @ np.stack([boots[(kind, row["quartet_id"])][metric] for row in subset])
            low, high = np.quantile(resamples, [0.025, 0.975])
            key = (kind, group, value, metric)
            summary_boot[key] = resamples
            summaries.append({"model": model, "method": method, "status": status, "kind": kind,
                              "group": group, "value": value, "metric": metric, "estimate": point,
                              "empirical_low": float(low), "empirical_high": float(high),
                              "n_pairs": len({row["pair_id"] for row in subset}), "n_quartets": len(subset),
                              "training_contact": partition_exposure(value if group == "training_partition" else "mixed")})
    return summaries, summary_boot


def enrich(qs: list[dict[str, Any]], pair_metadata: dict[str, dict[str, Any]], heldout_source: str = "HPVBG") -> list[dict[str, Any]]:
    output = []
    for q in qs:
        new = dict(q)
        pair = pair_metadata.get(q["pair_id"], {})
        new.setdefault("source_id", pair.get("source_id", "unknown"))
        new.setdefault("task", pair.get("task", "unknown"))
        new["training_partition"] = training_partition(new["source_id"], new.get("variant_id"), heldout_source)
        output.append(new)
    return output


def inspect_method(file: Path, model: str, method: str, inputs: dict[str, dict[str, Any]],
                   repetitions: dict[str, int], parse_response, normalize, validate_output,
                   source_partitions: dict[str, str] | None = None):
    values = collections.defaultdict(list)
    answers = collections.defaultdict(collections.Counter)
    counts = collections.Counter()
    seen, configs, errors = set(), set(), []
    costs = {"decision_input_tokens": 0, "decision_completion_tokens": 0, "fallback_rows": 0,
             "fallback_prompt_ids": set(), "response_rows": 0}
    if not file.exists():
        return values, answers, counts, {**costs, "fallback_prompt_ids": []}, "missing", ["response file absent"]
    # A live file can end with an incomplete final line. Preserve it and mark partial.
    with file.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                pid, rep = record["prompt_id"], record["replicate"]
                if pid not in inputs or type(rep) is not int or not 0 <= rep < repetitions[inputs[pid]["task_type"]]:
                    raise ValueError("trial outside selected plan")
                key = (pid, rep)
                if key in seen:
                    raise ValueError("duplicate prompt/replicate")
                seen.add(key)
                if record.get("model") != model or record.get("method") != method:
                    raise ValueError("mixed model/method")
                if source_partitions is not None and record.get("source_partition") != source_partitions[pid]:
                    raise ValueError("LoRA row source partition mismatch")
                if record.get("status") != "ok":
                    raise ValueError("record status is not ok")
                prompt = inputs[pid]
                validate_output(record["response_text"], prompt["output_schema"], record.get("finish_reason"))
                configs.add(record["config_sha256"])
                obj = json.loads(record["response_text"])
                field = next(iter(prompt["output_schema"]["properties"]))
                answer = obj[field]
                answers[pid][str(answer)] += 1
                if prompt["task_type"] == "behavior":
                    action, error = parse_response(record["response_text"], prompt["action_space"])
                    if error is not None:
                        raise ValueError(error)
                    outcome = normalize(action, prompt["outcome"])
                elif prompt["task_type"] == "goal":
                    outcome = int(answer in prompt["expected_actions"])
                else:
                    outcome = int(answer == prompt["expected_answer"])
                values[pid].append(outcome)
                counts[pid] += 1
                costs["response_rows"] += 1
                costs["decision_input_tokens"] += record.get("input_tokens", 0)
                costs["decision_completion_tokens"] += record.get("completion_tokens", 0)
                if record.get("fallback"):
                    costs["fallback_rows"] += 1
                    costs["fallback_prompt_ids"].add(pid)
            except (ValueError, KeyError, TypeError) as error:
                errors.append(f"line {line_number}: {type(error).__name__}: {error}")
    expected = sum(repetitions[prompt["task_type"]] for prompt in inputs.values())
    count_complete = all(counts[pid] == repetitions[prompt["task_type"]] for pid, prompt in inputs.items())
    marker = file.with_suffix(".complete.json")
    marker_ok = False
    if marker.exists():
        try:
            data = json.loads(marker.read_text(encoding="utf-8-sig"))
            marker_ok = (data["rows"] == expected and data["responses_sha256"] == filehash(file)
                         and data["config_sha256"] in configs and data["model"] == model and data["method"] == method)
        except (ValueError, KeyError):
            marker_ok = False
    status = "complete" if count_complete and not errors and len(configs) == 1 and marker_ok else "partial"
    costs["fallback_prompt_ids"] = sorted(costs["fallback_prompt_ids"])
    costs["expected_rows"] = expected
    costs["complete_marker_verified"] = marker_ok
    costs["configuration_hashes"] = sorted(configs)
    costs["raw_sha256"] = filehash(file)
    return values, answers, counts, costs, status, errors


def prompt_statistics(inputs, values, answers, model, method, status, manifest):
    records = []
    for pid, arr in sorted(values.items()):
        prompt = inputs[pid]
        counts = answers[pid]
        probabilities = np.array(list(counts.values()), dtype=float) / len(arr)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        specification = next(iter(prompt["output_schema"]["properties"].values()))
        menu_size = len(specification["enum"]) if "enum" in specification else specification["maximum"] - specification["minimum"] + 1
        sources = manifest.get("provenance_sources", {}).get(pid, [])
        heldout_source = manifest.get("learning", {}).get("heldout_source", "HPVBG")
        exposure = training_partition(heldout_source if heldout_source in sources else "other", prompt.get("variant_id"), heldout_source)
        record = {"model": model, "method": method, "status": status, "prompt_id": pid,
                  "task_type": prompt["task_type"], "n": len(arr), "mean_normalized_or_correct": float(np.mean(arr)),
                  "entropy_nats": entropy, "entropy_normalized": entropy / math.log(menu_size) if menu_size > 1 else 0.0,
                  "action_or_answer_distribution": json.dumps({key: count / len(arr) for key, count in sorted(counts.items())}, ensure_ascii=False),
                  "source_ids": json.dumps(sources, ensure_ascii=False),
                  "training_partition": exposure,
                  "training_contact": partition_exposure(exposure),
                  "exact_training_membership": "not_verified_from_training_examples",
                  "behavior_label_policy": "未提供自由行为硬标签；参数方法可接受等价与冻结教师软分布约束，仍有训练接触"}
        records.append(record)
    return records


def report(args: argparse.Namespace) -> None:
    parse_response, normalize, validate_output = load_helpers(args.baseline)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if filehash(args.baseline / "inputs.jsonl") != manifest["baseline_inputs_sha256"]:
        raise ValueError("Baseline inputs hash mismatch")
    full_inputs = {row["prompt_id"]: row for row in read(args.baseline / "inputs.jsonl")}
    inputs = {pid: full_inputs[pid] for pid in manifest["selected_prompt_ids"]}
    repetitions = manifest.get("repetitions", REPETITIONS)
    selected_qs = manifest["selected_quartet_ids"]
    pair_metadata = {row["pair_id"]: row for row in manifest["selected_pair_details"]}
    heldout_source = manifest.get("learning", {}).get("heldout_source", "HPVBG")
    behavior_qs = enrich([q for q in read(args.baseline / "INVAR_dataset/data/quartets.jsonl")
                          if q["quartet_id"] in set(selected_qs["behavior"])], pair_metadata, heldout_source)
    fact_qs = enrich([q for q in read(args.baseline / "fact_quartets.jsonl")
                      if q["quartet_id"] in set(selected_qs["fact"])], pair_metadata, heldout_source)
    goal_qs = enrich([q for q in read(args.baseline / "goal_quartets.jsonl")
                      if q["quartet_id"] in set(selected_qs["goal"])], pair_metadata, heldout_source)
    if any(q[side] not in inputs for q in behavior_qs + fact_qs + goal_qs for side in "ABCD"):
        raise ValueError("Selected quartet references unselected prompt")
    methods = list(dict.fromkeys(getattr(args, "methods", ",".join(METHODS)).split(",")))
    if not methods or any(method not in METHODS + LORA_METHODS for method in methods):
        raise ValueError("Only sampled prompt and LoRA methods are accepted; HF activation/likelihood results require a separate scorer")
    models = set(manifest.get("initial_models", []))
    if args.out.exists():
        models.update(directory.name for directory in args.out.iterdir() if directory.is_dir()
                      and any((directory / f"{method}.jsonl").exists() for method in methods))
    summaries, differences, audits, all_quartets, all_prompts, ledger_costs = [], [], [], [], [], []
    for model in sorted(models):
        model_results = {}
        for method in methods:
            file = args.out / model / f"{method}.jsonl"
            own_identity_path = file.parent / f"{method}.run_identity.json"
            method_identity = (json.loads(own_identity_path.read_text(encoding="utf-8-sig"))
                               if own_identity_path.exists() else {})
            own_plan = method_identity.get("plan", {})
            allowed_ids = own_plan.get("selected_prompt_ids", list(inputs))
            if not set(inputs) <= set(allowed_ids) or not set(allowed_ids) <= full_inputs.keys():
                raise ValueError("Method plan must contain frozen screen and only known baseline inputs")
            allowed_inputs = {pid: full_inputs[pid] for pid in allowed_ids}
            expected_partitions = None
            if method in LORA_METHODS and file.exists():
                expected_partitions = {
                    pid: ("heldout_source" if heldout_source in manifest["provenance_sources"][pid]
                          else "development_seen_source") for pid in allowed_inputs}
                if own_plan.get("source_partitions") and own_plan["source_partitions"] != expected_partitions:
                    raise ValueError("LoRA identity source partitions differ from provenance closure")
            values, answers, counts, costs, status, errors = inspect_method(
                file, model, method, allowed_inputs, repetitions, parse_response, normalize, validate_output,
                expected_partitions)
            costs["identity_path"] = str(own_identity_path) if method_identity else None
            costs["training_identity"] = method_identity.get("config", {}).get("training_identity")
            costs["analysis_scope"] = "frozen_common_screen_only; additional heldout rows audited but not compared without matched baseline"
            costs["analysis_selected_prompts"] = len(inputs)
            costs["extra_plan_prompts"] = len(allowed_inputs) - len(inputs)
            audits.append(dict(model=model, method=method, status=status, errors=errors, **costs))
            values = {pid: arr for pid, arr in values.items() if pid in inputs}
            answers = {pid: answer for pid, answer in answers.items() if pid in inputs}
            if not values:
                continue
            seed = 20261025 + int(hashlib.sha256(f"{model}|{method}".encode()).hexdigest()[:8], 16)
            points, boots, distributions = compute_quartets(behavior_qs, fact_qs + goal_qs,
                                                            values, args.bootstrap, seed)
            # Role checks are unique-prompt averages, as in the baseline report.
            role_pids = [pid for pid in values if inputs[pid]["task_type"] == "role"]
            method_summaries, summary_boot = summarize(points, boots, model, method, status)
            role_groups = [("all", "all", role_pids)] if role_pids else []
            role_partitions = {}
            for pid in role_pids:
                sources = manifest["provenance_sources"][pid]
                source = heldout_source if heldout_source in sources else "other"
                role_partitions[pid] = training_partition(source, inputs[pid].get("variant_id"), heldout_source)
            role_groups.extend(("training_partition", partition,
                                [pid for pid in role_pids if role_partitions[pid] == partition])
                               for partition in sorted(set(role_partitions.values())))
            for group, group_value, pids in role_groups:
                role_point = np.mean([distributions.mean(pid)[0] for pid in pids])
                role_boot = np.mean([distributions.mean(pid)[1] for pid in pids], axis=0)
                row = {"model": model, "method": method, "status": status, "kind": "role", "group": group,
                       "value": group_value, "metric": "accuracy", "estimate": float(role_point),
                       "empirical_low": float(np.quantile(role_boot, .025)), "empirical_high": float(np.quantile(role_boot, .975)),
                       "n_pairs": len({inputs[pid].get("pair_id", pid) for pid in pids}), "n_prompts": len(pids),
                       "training_contact": partition_exposure(group_value if group == "training_partition" else "mixed")}
                method_summaries.append(row)
                summary_boot[("role", group, group_value, "accuracy")] = role_boot
            prompts = prompt_statistics(inputs, values, answers, model, method, status, manifest)
            all_prompts.extend(prompts)
            summaries.extend(method_summaries)
            all_quartets.extend(dict(point, model=model, method=method, status=status) for point in points)
            model_results[method] = {"status": status, "summaries": method_summaries, "boot": summary_boot}
        ledger_file = args.out / model / "ledger_cost.json"
        if ledger_file.exists():
            ledger_costs.append(dict(model=model, **json.loads(ledger_file.read_text(encoding="utf-8"))))
        baseline_result = model_results.get("baseline")
        if not baseline_result or baseline_result["status"] != "complete":
            continue
        baseline_summaries = {(r["kind"], r["group"], r["value"], r["metric"]): r for r in baseline_result["summaries"]}
        for method, result in model_results.items():
            if method == "baseline" or result["status"] != "complete":
                continue
            for row in result["summaries"]:
                key = (row["kind"], row["group"], row["value"], row["metric"])
                old = baseline_summaries.get(key)
                if old is None or old["n_pairs"] != row["n_pairs"]:
                    continue
                change = result["boot"][key] - baseline_result["boot"][key]
                low, high = np.quantile(change, [.025, .975])
                differences.append({"model": model, "method": method, "comparison": "matched_sampled_baseline",
                                    "kind": key[0], "group": key[1], "value": key[2], "metric": key[3],
                                    "difference_method_minus_baseline": row["estimate"] - old["estimate"],
                                    "empirical_low": float(low), "empirical_high": float(high),
                                    "n_pairs": row["n_pairs"], "status": "complete_development_only",
                                    "training_contact": row.get("training_contact", partition_exposure("mixed"))})
    args.out.mkdir(parents=True, exist_ok=True)
    scores = args.out / "scores"
    scores.mkdir(exist_ok=True)
    write_csv(scores / "summary.csv", summaries)
    write_csv(scores / "differences.csv", differences)
    write_csv(scores / "quartets.csv", all_quartets)
    write_csv(scores / "prompts.csv", all_prompts)
    write_csv(scores / "completion_audit.csv", [{**a, "errors": json.dumps(a["errors"], ensure_ascii=False),
                                                "fallback_prompt_ids": json.dumps(a["fallback_prompt_ids"])} for a in audits])
    payload = {"scorer_version": SCORER_VERSION, "methods": methods,
               "status": "development_screen_not_blind_confirmation", "bootstrap_draws": args.bootstrap,
               "manifest_sha256": filehash(args.manifest), "scorer_sha256": filehash(Path(__file__)),
               "ci_scope": "fixed-item empirical response resampling; independent methods; shared prompt references reuse draws",
               "zero_effect_test": False, "summaries": summaries, "differences": differences,
               "completion_audit": audits, "ledger_extraction_costs": ledger_costs,
               "activation_or_likelihood_proxy_included": False}
    (args.out / "development_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# INVAR 第二轮：保守归一化与条件内锚定", "",
             "本报告为已见数据上的开发筛选，不是独立确认实验。只比较相同生成接口、相同冻结题目的输入处理与参数学习方法；激活/似然代理结果不在此表。", "",
             "E/G越低表示呈现更稳定；R只描述条件变化幅度，没有越大越好的含义。U为事实准确率，Q为明示目标执行准确率。自由选择无唯一正确行动。", "",
             "汇总先在每个条件对内对适用变体等权，再对条件对等权；事实original/mirror选项顺序均保留并等权。角色为唯一检查题等权平均。", "",
             "## 完整性", "", "| 模型 | 方法 | 状态 | 有效响应/计划 | 回退响应 |", "|---|---|---|---:|---:|"]
    for audit in audits:
        lines.append(f"| {audit['model']} | {audit['method']} | {audit['status']} | {audit['response_rows']}/{audit.get('expected_rows',sum(repetitions[p['task_type']] for p in inputs.values()))} | {audit['fallback_rows']} |")
    lines += ["", "缺少响应、非法输出、重复试次或未通过完成文件哈希核验的结果标记partial/missing。它们只保留审计和探索性逐题文件，不进入下面的完成方法表或方法差表。", "",
              "## 已完成方法的总体表现（训练接触混合，仅开发描述）", "",
              "本轮LoRA在非HPVBG来源、排除V4/V6的材料上训练；自由行为还参与等价与条件内软分布约束。没有人类行为标签，但确实有训练接触。全1,378输入为接触混合的开发汇总，不能称未见泛化；主迁移分区另列。", "",
              "| 模型 | 方法 | 训练接触 | E↓ | R | G↓ | 应变U↑ | 未变U↑ | Q↑ | 角色↑ |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    lookup = {(r["model"], r["method"], r["kind"], r["metric"]): r["estimate"] for r in summaries
              if r["group"] == "all" and r["status"] == "complete"}
    def formatted(model, method, kind, metric):
        value = lookup.get((model, method, kind, metric))
        return "—" if value is None else f"{value:.4f}"
    for audit in audits:
        if audit["status"] != "complete":
            continue
        model, method = audit["model"], audit["method"]
        values = [formatted(model, method, kind, metric) for kind, metric in
                  (("behavior", "E"), ("behavior", "R"), ("behavior", "G"), ("changed_fact", "accuracy"),
                   ("stable_fact", "accuracy"), ("goal", "accuracy"), ("role", "accuracy"))]
        exposure = "已见/留出混合，非unseen" if method in LORA_METHODS else "无本阶段参数学习"
        lines.append("| " + " | ".join([model, method, exposure] + values) + " |")
    lines += ["", "## LoRA主要迁移比较：按训练接触分开", "",
              "HPVBG为留来源；非HPVBG的V4规则表格、V6关系改述分别为留变体；其余已见来源/变体只作训练接触开发参照。V4/V6四元组的A/C原呈现可能参加训练，不能说四侧全部未见。所有比较仍是研究者已看过的固定材料，不是新盲测。", "",
              "| 模型 | 方法 | 分区 | 检查类型 | 指标 | 数值 | 方法−匹配baseline | 训练接触 |", "|---|---|---|---|---|---:|---:|---|"]
    delta_lookup = {(r["model"], r["method"], r["kind"], r["group"], r["value"], r["metric"]): r for r in differences}
    for row in summaries:
        if row["method"] not in LORA_METHODS or row["status"] != "complete" or row["group"] != "training_partition":
            continue
        if row["metric"] not in {"E", "G", "accuracy"}:
            continue
        key = (row["model"], row["method"], row["kind"], row["group"], row["value"], row["metric"])
        diff = delta_lookup.get(key)
        shown = "—（baseline未完成）" if diff is None else f"{diff['difference_method_minus_baseline']:+.4f}"
        lines.append(f"| {row['model']} | {row['method']} | {row['value']} | {row['kind']} | {row['metric']} | {row['estimate']:.4f} | {shown} | {row['training_contact']} |")
    lines += ["", "## 相对同模型新采样基线的差值", "",
              "差值=方法−baseline；E/G负值为降低，U/Q正值为提高。不同方法输出独立采样，不把相同replicate编号当作同一随机决策的配对。配对发生在固定题目及汇总权重上。", "",
              "| 模型 | 方法 | 指标 | 差值 | 经验重采样范围 |", "|---|---|---|---:|---|"]
    for row in differences:
        if row["method"] not in LORA_METHODS and row["group"] == "all" and (row["metric"] in {"E", "G"} or row["metric"] == "accuracy"):
            lines.append(f"| {row['model']} | {row['method']} | {row['kind']}:{row['metric']} | {row['difference_method_minus_baseline']:+.4f} | [{row['empirical_low']:+.4f}, {row['empirical_high']:+.4f}] |")
    lines += ["", "## 使用边界与成本", "",
              f"采用{args.bootstrap}次唯一提示级响应自助抽样：每轮每个提示只抽一次，全部四元组引用复用，baseline在所有方法差中复用。不同方法之间使用独立抽样流。范围仅反映固定题目与经验分布下的采样，不代表新情境总体区间，不用于零效应检验；绝对值/最小值有边界偏差，全对或全错会退化。", "",
              "事实两种选项顺序以评分端正确语义评分，不能直接把A/B字母的变化当作知识更新。双调用指标是独立调用正确率乘积，不是一个智能体的实测轨迹。", "",
              "scores/prompts.csv保留行动/答案概率、行动熵及归一化熵；scores/quartets.csv保留逐四元组E/R/G与控制准确率；scores/summary.csv给出来源、任务、变体、选项顺序、目标政策和应变子集分项。scores/differences.csv给出相同分项相对匹配baseline的差值。", "",
              "费用必须同时看decision输入/输出token与ledger_extraction_costs。账本按唯一材料缓存，在两个架构和所有重复间共享，提取成本不能重复相加或漏报；主表回退包含在总体评分中，不只保留解析成功题。", "",
              "若LoRA另跑了完整HPVBG留来源扩展，完整性按其独立run_identity计划验收；本报告方法差仍限定原冻结screen的共同输入。额外材料在没有同规模匹配baseline时只记录完成性，不补成虚假的对比。各方法complete文件分别核对自身响应config_hash，LoRA不沿用提示方法的全局config。", "",
              "E降低须同时检查U/Q、行为熵及跨情境区分；恒定行动或统一偏好不构成可靠性改进。自由R变化不直接判好坏。", ""]
    (args.out / "development_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "scored_development", "audits": len(audits),
                      "complete": sum(a["status"] == "complete" for a in audits),
                      "summaries": len(summaries), "matched_differences": len(differences)}, ensure_ascii=False))


def self_test() -> None:
    assert training_partition("HPVBG", "V4_rules_table") == "heldout_source_HPVBG"
    assert training_partition("PIQ572", "V4_rules_table") == "heldout_variant_V4_seen_source"
    assert training_partition("PGG", "V6_relation") == "heldout_variant_V6_seen_source"
    assert training_partition("PGG", "V5_ids") == "seen_source_seen_variant"
    assert abs(w1(np.array([0, 0]), np.array([1, 1])) - 1) < 1e-12
    assert w1(np.array([0, 1]), np.array([0, 1])) == 0
    q = {"quartet_id": "q", "pair_id": "pair", "A": "a", "B": "b", "C": "c", "D": "d"}
    values = {"a": [0] * 16, "b": [0] * 16, "c": [1] * 16, "d": [1] * 16}
    points, boots, _ = compute_quartets([q], [], values, 500, 1)
    assert points[0]["E"] == 0 and points[0]["R"] == 1 and points[0]["G"] == 0
    flip = dict(values, b=[1] * 16, d=[0] * 16)
    points, boots, _ = compute_quartets([q], [], flip, 500, 1)
    assert points[0]["E"] == 1 and points[0]["G"] == 2
    # Different question-letter order, same semantic correctness after scoring.
    with tempfile.TemporaryDirectory(prefix="invar_score_test_") as temp:
        directory = Path(temp)
        file = directory / "baseline.jsonl"
        schema = {"properties": {"answer": {"type": "string", "enum": ["A", "B"]}}}
        prompts = {"original": {"task_type": "changed_fact", "expected_answer": "A", "output_schema": schema},
                   "mirror": {"task_type": "changed_fact", "expected_answer": "B", "output_schema": schema}}
        records = [{"model": "model", "method": "baseline", "prompt_id": pid, "replicate": 0,
                    "status": "ok", "response_text": json.dumps({"answer": answer}), "finish_reason": "stop",
                    "config_sha256": "config", "fallback": False, "input_tokens": 1, "completion_tokens": 1}
                   for pid, answer in (("original", "A"), ("mirror", "B"))]
        file.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        marker = {"rows": 2, "responses_sha256": filehash(file), "config_sha256": "config",
                  "model": "model", "method": "baseline"}
        file.with_suffix(".complete.json").write_text(json.dumps(marker), encoding="utf-8")
        def valid_answer(text, schema, reason):
            assert json.loads(text)["answer"] in {"A", "B"}
        got = inspect_method(file, "model", "baseline", prompts, {"changed_fact": 1}, None, None, valid_answer)
        assert got[0] == {"original": [1], "mirror": [1]} and got[4] == "complete"
        # A letter-only policy fails after reversal although its surface is stable.
        records[1]["response_text"] = '{"answer":"A"}'
        file.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        marker["responses_sha256"] = filehash(file)
        file.with_suffix(".complete.json").write_text(json.dumps(marker), encoding="utf-8")
        got = inspect_method(file, "model", "baseline", prompts, {"changed_fact": 1}, None, None, valid_answer)
        assert got[0]["mirror"] == [0] and got[4] == "complete"
        # An incomplete/live response file must not enter the completed table.
        file.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
        got = inspect_method(file, "model", "baseline", prompts, {"changed_fact": 1}, None, None, valid_answer)
        assert got[4] == "partial"
        # LoRA has its own per-method configuration; an unrelated prompt config
        # must neither invalidate it nor make a wrong source partition acceptable.
        lora_file = directory / "sft.jsonl"
        lora_records = [dict(row, method="sft", config_sha256="adapter-config", source_partition="heldout_source")
                        for row in records]
        lora_file.write_text("".join(json.dumps(row) + "\n" for row in lora_records), encoding="utf-8")
        (directory / "config.json").write_text('{"config_sha256":"prompt-config"}', encoding="utf-8")
        marker.update(method="sft", config_sha256="adapter-config", responses_sha256=filehash(lora_file))
        lora_file.with_suffix(".complete.json").write_text(json.dumps(marker), encoding="utf-8")
        expected = {pid: "heldout_source" for pid in prompts}
        got = inspect_method(lora_file, "model", "sft", prompts, {"changed_fact": 1}, None, None, valid_answer, expected)
        assert got[4] == "complete"
        got = inspect_method(lora_file, "model", "sft", prompts, {"changed_fact": 1}, None, None,
                             valid_answer, {pid: "development_seen_source" for pid in prompts})
        assert got[4] == "partial" and got[5]
    # Duplicate quartet references must share bootstrap draws exactly.
    _, _, samples = compute_quartets([], [], {"shared": [0, 0, 1, 1]}, 500, 3)
    assert np.array_equal(samples.mean("shared")[1], samples.mean("shared")[1])
    subset = [{"pair_id": "p1"}, {"pair_id": "p1"}, {"pair_id": "p2"}]
    assert np.allclose(weights_for(subset), [.25, .25, .5])
    print(json.dumps({"self_test": "passed", "scorer_version": SCORER_VERSION,
                      "checks": ["known_W1", "E_R_G_bounds", "semantic_letter_reversal", "partial_exclusion",
                                 "shared_prompt_bootstrap", "pair_then_variant_weights", "four_training_partitions",
                                 "independent_lora_config", "source_partition_validation"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="Comma-separated sampled methods; additionally supports four anchor learning arms")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.baseline, args.manifest, args.out)):
        parser.error("--baseline, --manifest, --out required")
    if args.bootstrap < 500:
        parser.error("Use at least 500 resamples")
    report(args)


if __name__ == "__main__":
    main()
