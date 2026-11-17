"""Frozen four-arm INVAR pilot: SFT, free-behavior JS, conditional anchor KL.

Default preparation is CPU/standard-library only. --teacher-only EXPLICITLY
loads the unadapted frozen base model on GPU and publishes a complete cache.
Training never creates a missing cache. Every arm shares the same frozen
supervised/free-behavior exposure schedule. No free-behavior hard labels.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROTOCOL = "INVAR_condition_anchor_four_arm_20261026_v1"
ARMS = {"anchor_sft": (0.0, 0.0), "anchor_js": (1.0, 0.0),
        "anchor_kl": (0.0, 1.0), "anchor_js_kl": (1.0, 1.0)}
SEED = 20261026
EXCLUDED_VARIANTS = {"V4_rules_table", "V6_relation"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    import hashlib
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_file(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def load_legacy(directory):
    """Read frozen V1 helpers without altering them or using their training entry point."""
    path = Path(directory).resolve() / "train_lora.py"
    require(path.is_file(), f"Frozen V1 helper absent: {path}")
    spec = importlib.util.spec_from_file_location("invar_frozen_lora_v1_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_closure(baseline, inputs, pairs):
    refs = collections.defaultdict(set)
    for path in (baseline / "INVAR_dataset/data/quartets.jsonl", baseline / "fact_quartets.jsonl",
                 baseline / "goal_quartets.jsonl"):
        for row in read(path):
            for side in "ABCD":
                refs[row[side]].add(row["pair_id"])
    for pid, row in inputs.items():
        if row.get("pair_id"):
            refs[pid].add(row["pair_id"])
        for use in row.get("uses", []):
            refs[pid].add(use["pair_id"])
    for pid in inputs:
        require(refs[pid] and refs[pid] <= pairs.keys(), f"Unresolved source closure: {pid}")
    return refs, {pid: sorted({pairs[p]["source_id"] for p in ids}) for pid, ids in refs.items()}


def public_menu(row):
    props = row["output_schema"]["properties"]
    require(len(props) == 1, "Exactly one public response field required")
    field, spec = next(iter(props.items()))
    if "enum" in spec:
        values = list(spec["enum"])
    else:
        require(spec.get("type") == "integer" and "minimum" in spec and "maximum" in spec,
                "A finite public action menu is required")
        values = list(range(spec["minimum"], spec["maximum"] + 1))
    require(1 < len(values) <= 65536 and len(set(values)) == len(values), "Invalid public action menu")
    return field, values


def connected_conditions(free_rows, edges, interventions):
    """Union ONLY audited same-condition edges; reject real-condition bridges."""
    parent = {row["prompt_id"]: row["prompt_id"] for row in free_rows}
    by_id = {row["prompt_id"]: row for row in free_rows}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for edge in edges:
        a, b = find(edge["left"]), find(edge["right"])
        parent[max(a, b)] = min(a, b)
    groups = collections.defaultdict(list)
    for pid in sorted(parent):
        groups[find(pid)].append(pid)
    pid_component, components = {}, []
    for members in sorted(groups.values(), key=lambda x: x[0]):
        first = by_id[members[0]]
        for pid in members:
            row = by_id[pid]
            require(row["source_id"] == first["source_id"] and row["field"] == first["field"]
                    and row["candidates"] == first["candidates"], "Incompatible candidate/source component")
        cid = "condition_" + digest(members)[:24]
        for pid in members:
            pid_component[pid] = cid
        components.append({"condition_id": cid, "source_id": first["source_id"],
                           "prompt_ids": members, "field": first["field"],
                           "candidates": first["candidates"], "candidate_semantics": first["candidate_semantics"]})
    for a, c, pair_id in interventions:
        if a in parent and c in parent:
            require(pid_component[a] != pid_component[c],
                    f"Equivalence component accidentally joins real conditions: {pair_id}")
    for edge in edges:
        edge["condition_id"] = pid_component[edge["left"]]
        require(edge["condition_id"] == pid_component[edge["right"]], "Disconnected equivalence edge")
    require(all(len(c["prompt_ids"]) >= 2 for c in components), "Condition without distinct equivalent presentations")
    return components


def build_schedule(supervised_edges, free_edges, components, steps, seed):
    """All four arms get the identical immutable exposure list."""
    sup_groups = {kind: [e for e in supervised_edges if e["task_type"] == kind]
                  for kind in ("changed_fact", "stable_fact", "goal")}
    require(all(sup_groups.values()), "Empty supervised task group")
    free_groups = {c["condition_id"]: [] for c in components}
    for edge in free_edges:
        free_groups[edge["condition_id"]].append(edge)
    require(all(free_groups.values()), "Empty free-behavior condition")
    rng = random.Random(seed)
    tasks, cids = sorted(sup_groups), sorted(free_groups)
    schedule = []
    for step in range(1, steps + 1):
        for micro in range(8):
            task = rng.choice(tasks)
            sup = rng.choice(sup_groups[task])
            cid = rng.choice(cids)
            free = rng.choice(free_groups[cid])
            schedule.append({"optimizer_step": step, "micro_step": micro,
                             "supervised_edge_id": sup["edge_id"], "free_edge_id": free["edge_id"],
                             "condition_id": cid, "supervised_task": task})
    return schedule


def prepare(args):
    legacy = load_legacy(args.legacy_helper_dir)
    baseline = Path(args.baseline).resolve()
    supervised, supervised_edges, old_audit = legacy.prepare(baseline, args.heldout_source)
    inputs = {row["prompt_id"]: row for row in read(baseline / "inputs.jsonl")}
    pairs = {row["pair_id"]: row for row in read(baseline / "INVAR_dataset/data/pairs.jsonl")}
    refs, sources = source_closure(baseline, inputs, pairs)
    free_by_id, edge_map, interventions = {}, {}, []
    rejected = collections.Counter()
    for q in read(baseline / "INVAR_dataset/data/quartets.jsonl"):
        if q["source_id"] == args.heldout_source:
            rejected["heldout_source_quartet"] += 1
            continue
        if q["variant_id"] in EXCLUDED_VARIANTS:
            rejected["heldout_variant_quartet"] += 1
            continue
        interventions.append((q["A"], q["C"], q["pair_id"]))
        for side in ("AB", "CD"):
            left, right = q[side[0]], q[side[1]]
            if any(sources[pid] != [q["source_id"]] for pid in (left, right)):
                rejected["mixed_or_heldout_reference_closure"] += 1
                continue
            if left == right:
                rejected["identical_equivalent_input"] += 1
                continue
            if any(len(public_menu(inputs[pid])[1]) > args.max_free_candidates for pid in (left, right)):
                rejected["complete_menu_exceeds_free_candidate_budget"] += 1
                continue
            for pid in (left, right):
                row = inputs[pid]
                require(row["task_type"] == "behavior", "Non-behavior input in free-behavior edge")
                field, menu = public_menu(row)
                require(field == "action", "Unexpected behavior response field")
                messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
                require([m["role"] for m in messages] == ["system", "user"], "Unexpected prompt roles")
                free_by_id[pid] = {"prompt_id": pid, "task_type": "behavior", "messages": messages,
                                   "source_id": sources[pid][0], "pair_references": sorted(refs[pid]),
                                   "field": field, "candidates": menu, "candidate_semantics": menu,
                                   "messages_sha256": digest(messages)}
            a, b = free_by_id[left], free_by_id[right]
            require(a["candidates"] == b["candidates"], "Equivalent action menus disagree")
            key = tuple(sorted((left, right)))
            if key not in edge_map:
                edge_map[key] = {"edge_id": "free_eq_" + digest(key)[:20], "left": key[0], "right": key[1],
                                 "source_id": q["source_id"], "references": []}
            edge_map[key]["references"].append({"pair_id": q["pair_id"], "quartet_id": q["quartet_id"],
                                                 "sides": side, "variant_id": q["variant_id"]})
    free_rows = [free_by_id[pid] for pid in sorted(free_by_id)]
    free_edges = sorted(edge_map.values(), key=lambda e: e["edge_id"])
    require(free_rows and free_edges, "No eligible behavior training data")
    components = connected_conditions(free_rows, free_edges, interventions)
    require(all(r["source_id"] != args.heldout_source for r in supervised + free_rows), "Test source in training")
    require(all(q["variant_id"] not in EXCLUDED_VARIANTS for e in free_edges for q in e["references"]),
            "Held-out presentation used as an equivalence edge")
    schedule = build_schedule(supervised_edges, free_edges, components, args.max_steps, args.seed)
    # task is present in condition-pair metadata, not necessarily in prompt uses.
    menu_coverage = collections.Counter()
    for pid, row in inputs.items():
        if row["task_type"] != "behavior":
            continue
        n = len(public_menu(row)[1])
        source_label = "+".join(sources[pid])
        tasks = "+".join(sorted({pairs[p]["task"] for p in refs[pid]}))
        status = "included_complete_menu" if pid in free_by_id else (
            "excluded_mixed_source" if len(sources[pid]) != 1 else
            "evaluation_only_heldout_source" if args.heldout_source in sources[pid] else
            "evaluation_only_large_menu" if n > args.max_free_candidates else
            "excluded_no_allowed_equivalence_edge")
        menu_coverage[(source_label, tasks, n, status)] += 1
    audit = {
        "protocol": PROTOCOL, "heldout_source": args.heldout_source, "seed": args.seed,
        "max_optimizer_steps": args.max_steps, "heldout_variants": sorted(EXCLUDED_VARIANTS),
        "max_free_candidates": args.max_free_candidates,
        "input_sha256": old_audit["input_sha256"], "supervised_split_sha256": old_audit["split_content_sha256"],
        "legacy_helper_sha256": sha_file(Path(args.legacy_helper_dir) / "train_lora.py"),
        "supervised_inputs": len(supervised), "supervised_edges": len(supervised_edges),
        "free_inputs": len(free_rows), "free_edges": len(free_edges), "semantic_conditions": len(components),
        "free_inputs_by_source": dict(collections.Counter(r["source_id"] for r in free_rows)),
        "conditions_by_source": dict(collections.Counter(c["source_id"] for c in components)),
        "free_menu_coverage": [{"source_id": s, "task": t, "legal_candidates": n, "status": status,
                                "unique_prompts": count}
                               for (s, t, n, status), count in sorted(menu_coverage.items())],
        "supervised_legal_menu_histogram": dict(sorted(collections.Counter(len(r["candidates"]) for r in supervised).items())),
        "supervised_correct_set_maximum": max(len(r["expected"]) for r in supervised),
        "large_menu_policy": "public menus above fixed budget excluded only from free training/teacher; all candidates retained at evaluation",
        "rejected": dict(rejected), "heldout_source_rows_in_training": 0, "mixed_source_rows_in_training": 0,
        "heldout_variant_edges_in_training": 0, "free_behavior_hard_labels": 0,
        "real_condition_bridges": 0, "supervised_js": False,
        "components_sha256": digest(components), "free_inputs_sha256": digest(free_rows),
        "free_edges_sha256": digest(free_edges), "exposure_schedule_sha256": digest(schedule),
        "supervised_loss": "D0 frozen: negative log complete correct-set probability / mean correct suffix length",
        "proxy": "HF unnormalized complete candidate+EOS probabilities normalized over public legal candidates; no length normalization",
        "teacher_pool": "arithmetic mean of unique training presentations within each separate semantic condition",
        "source_claim": "retrospective source holdout; V2 has already been inspected; not new confirmation",
    }
    audit["split_content_sha256"] = digest(audit)
    return legacy, supervised, supervised_edges, free_rows, free_edges, components, schedule, audit


def save_preparation(output, prepared):
    _, sup, sup_edges, free, free_edges, conditions, schedule, audit = prepared
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old_path = output / "data_audit.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        require(old["split_content_sha256"] == audit["split_content_sha256"], "Preparation changed; use new output directory")
    files = {"supervised_inputs.jsonl": sup, "supervised_edges.jsonl": sup_edges,
             "free_inputs.jsonl": free, "free_edges.jsonl": free_edges,
             "semantic_conditions.jsonl": conditions, "exposure_schedule.jsonl": schedule}
    for name, rows in files.items():
        write_rows(output / name, rows)
    atomic_json(output / "condition_split_audit.json", {
        "split_content_sha256": audit["split_content_sha256"],
        "component_membership_sha256": audit["components_sha256"],
        "semantic_conditions": len(conditions), "real_condition_bridges": 0,
        "equivalence_sides_only": ["AB", "CD"], "heldout_source": audit["heldout_source"],
        "heldout_variants": audit["heldout_variants"],
        "free_menu_coverage": audit["free_menu_coverage"], "max_free_candidates": audit["max_free_candidates"],
        "candidate_sampling": False,
    })
    atomic_json(output / "candidate_audit.json", {
        "max_free_candidates": audit["max_free_candidates"],
        "free_candidate_counts": dict(sorted(collections.Counter(len(r["candidates"]) for r in free).items())),
        "supervised_legal_menu_histogram": audit["supervised_legal_menu_histogram"],
        "supervised_correct_set_maximum": audit["supervised_correct_set_maximum"],
        "all_selected_legal_candidates_enumerated": True, "large_menu_policy": audit["large_menu_policy"],
    })
    saved_audit = dict(audit, prepared_file_hashes={name: sha_file(output / name) for name in files})
    atomic_json(output / "data_audit.json", saved_audit)


def js_and_anchor(torch, logqa, logqb, logteacher):
    """Differentiable tiny-vector losses; all dimensions represent the same actions."""
    mixture = torch.logsumexp(torch.stack((logqa, logqb)), dim=0) - math.log(2)
    js = 0.5 * ((logqa.exp() * (logqa - mixture)).sum() + (logqb.exp() * (logqb - mixture)).sum())
    teacher = logteacher.detach()
    kl = 0.5 * ((teacher.exp() * (teacher - logqa)).sum() + (teacher.exp() * (teacher - logqb)).sum())
    return js, kl


def initialize_backend(args, prepared, output, for_teacher=False):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    legacy, sup, _, free, _, _, _, audit = prepared
    require(args.model is not None, "--model must point to existing local Qwen3-8B weights")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    # Audit tokenizer only, then load weights; no silent truncation.
    all_rows = free if for_teacher else sup + free
    token_cache, token_audit = legacy.tokenize_examples(all_rows, tokenizer, args.max_length)
    atomic_json(Path(output) / "tokenization_audit.json", token_audit)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "One BF16-capable CUDA GPU required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                                dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    require(model.config.model_type == "qwen3", "Initial pilot requires Qwen3")
    require(getattr(model.config, "attention_dropout", 0.0) == 0.0, "Deterministic candidate scoring requires zero attention dropout")
    require("logits_to_keep" in inspect.signature(model.forward).parameters, "Native suffix-only logits support required")
    model.config.use_cache = False
    weight_metadata = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                       for p in sorted(Path(args.model).glob("*.safetensors"))]
    identity = {"base_model_path": str(Path(args.model).resolve()),
                "base_model_config_sha256": sha_file(Path(args.model) / "config.json"),
                "base_weight_file_metadata": weight_metadata,
                "base_weight_metadata_sha256": digest(weight_metadata),
                "weight_identity_note": "config hash and weight file metadata; no claim of full weight-byte hashes",
                "chat_template_sha256": digest(tokenizer.chat_template),
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "input_sha256": audit["input_sha256"], "split_content_sha256": audit["split_content_sha256"],
                "max_length": args.max_length, "dtype": "bfloat16", "enable_thinking": False,
                "candidate_likelihood_policy": audit["proxy"],
                "runner_sha256": sha_file(Path(__file__)), "legacy_helper_sha256": audit["legacy_helper_sha256"]}
    return torch, model, tokenizer, token_cache, identity


def candidate_logps(torch, model, cache, legacy, row, actions=None):
    import torch.nn.functional as F
    actions = row["candidates"] if actions is None else actions
    scores, lengths = [], []
    for action in actions:
        tokens, labels = cache[(row["prompt_id"], canonical(action))]
        keep, suffix = legacy.answer_projection_spec(labels)
        ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        mask = torch.ones_like(ids)
        result = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep)
        require(result.logits.shape[1] == keep, "Model ignored suffix-only projection")
        logits = result.logits[0, :-1].float()
        target = torch.tensor(suffix, dtype=torch.long, device="cuda")
        require(logits.shape[0] == target.shape[0], "Answer suffix misalignment")
        scores.append(-F.cross_entropy(logits, target, reduction="sum"))
        lengths.append(len(suffix))
    return torch.stack(scores), lengths


def cache_config(identity, audit):
    # Exposure step budget does not alter the teacher, but including the split
    # hash makes cross-run use deliberate and rejects ambiguous changed plans.
    return {"protocol": PROTOCOL, "teacher": "frozen_base_no_adapter_eval_no_grad",
            "identity": identity, "components_sha256": audit["components_sha256"],
            "free_inputs_sha256": audit["free_inputs_sha256"], "pool": audit["teacher_pool"],
            "heldout_source": audit["heldout_source"], "excluded_variants": audit["heldout_variants"],
            "max_free_candidates": audit["max_free_candidates"]}


def build_teacher(args, prepared):
    directory = Path(args.teacher_cache)
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / "teacher_manifest.json").exists(), "Completed teacher exists; do not overwrite")
    lock = directory / "BUILD.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(canonical({"pid": os.getpid(), "unix": time.time()}))
    # A failed build keeps its lock and failure record; never silently reuse a partial cache.
    try:
        legacy, _, _, free, _, conditions, _, audit = prepared
        torch, model, _, cache, identity = initialize_backend(args, prepared, directory, for_teacher=True)
        require(not any("lora_" in name for name, _ in model.named_parameters()), "Teacher contains adapter parameters")
        model.requires_grad_(False).cuda().eval()
        config = cache_config(identity, audit)
        by_id, prompt_records = {}, []
        with torch.no_grad():
            for index, row in enumerate(free):
                complete_logps, lengths = candidate_logps(torch, model, cache, legacy, row)
                logq = torch.log_softmax(complete_logps.double(), dim=0)
                require(bool(torch.isfinite(logq).all()), "Nonfinite teacher probability")
                record = {"prompt_id": row["prompt_id"], "source_id": row["source_id"],
                          "candidates": row["candidates"], "candidate_semantics": row["candidate_semantics"],
                          "complete_candidate_logps": complete_logps.cpu().double().tolist(),
                          "suffix_token_lengths": lengths, "logq": logq.cpu().tolist(),
                          "messages_sha256": row["messages_sha256"]}
                by_id[row["prompt_id"]] = record
                prompt_records.append(record)
                if (index + 1) % 25 == 0:
                    atomic_json(directory / "teacher_progress.json", {"status": "building",
                                "prompts_done": index + 1, "prompts_total": len(free)})
        condition_records = []
        for condition in conditions:
            vectors = torch.tensor([by_id[pid]["logq"] for pid in condition["prompt_ids"]], dtype=torch.float64)
            mean_logq = torch.logsumexp(vectors, dim=0) - math.log(len(condition["prompt_ids"]))
            require(abs(float(mean_logq.exp().sum()) - 1) < 1e-8, "Teacher consensus not normalized")
            condition_records.append({**condition, "teacher_logq": mean_logq.tolist(),
                                      "pool": "arithmetic mean of unique presentation probabilities"})
        write_rows(directory / "teacher_prompt_distributions.jsonl", prompt_records)
        write_rows(directory / "teacher_condition_distributions.jsonl", condition_records)
        manifest = {"status": "complete", "config": config, "config_sha256": digest(config),
                    "input_sha256": audit["input_sha256"], "split_content_sha256": audit["split_content_sha256"],
                    "prompt_rows": len(prompt_records), "condition_rows": len(condition_records),
                    "heldout_source": args.heldout_source, "excluded_variants": sorted(EXCLUDED_VARIANTS),
                    "max_free_candidates": args.max_free_candidates,
                    "files_sha256": {name: sha_file(directory / name) for name in (
                        "teacher_prompt_distributions.jsonl", "teacher_condition_distributions.jsonl")},
                    "human_truth": False, "vllm_frequencies_used": False,
                    "heldout_sources_used": False, "finished_unix": time.time()}
        # Publication barrier: this immutable complete manifest is written LAST.
        atomic_json(directory / "teacher_manifest.json", manifest)
        atomic_json(directory / "teacher_progress.json", {"status": "complete", "prompts_done": len(free),
                                                            "conditions_done": len(condition_records)})
        lock.unlink()
        print(canonical({"teacher": "complete", "prompts": len(free), "conditions": len(condition_records),
                         "manifest_sha256": sha_file(directory / "teacher_manifest.json")}), flush=True)
    except Exception as error:
        atomic_json(directory / "teacher_FAILED.json", {"error": str(error), "type": type(error).__name__,
                                                        "lock_retained": True, "unix": time.time()})
        raise


def load_teacher(args, prepared):
    directory = Path(args.teacher_cache)
    manifest_path = directory / "teacher_manifest.json"
    require(manifest_path.is_file(), "Teacher cache not published complete; build explicitly with --teacher-only")
    require(not (directory / "BUILD.lock").exists(), "Teacher cache still has an active/failed build lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "complete", "Incomplete teacher manifest")
    require(manifest["config_sha256"] == digest(manifest["config"]), "Teacher config identity mismatch")
    audit = prepared[-1]
    require(manifest["split_content_sha256"] == audit["split_content_sha256"], "Teacher split changed")
    require(manifest["input_sha256"] == audit["input_sha256"], "Teacher input hash changed")
    require(manifest["config"]["identity"]["base_model_path"] == str(Path(args.model).resolve()), "Teacher base weights differ")
    require(manifest["config"]["identity"]["base_model_config_sha256"] == sha_file(Path(args.model) / "config.json"),
            "Teacher base configuration changed")
    for name, expected in manifest["files_sha256"].items():
        require(sha_file(directory / name) == expected, "Teacher cache byte hash mismatch")
    prompt_rows = read(directory / "teacher_prompt_distributions.jsonl")
    prompts = {row["prompt_id"]: row for row in prompt_rows}
    expected_free = {row["prompt_id"]: row for row in prepared[3]}
    require(len(prompts) == len(prompt_rows) == manifest["prompt_rows"] and prompts.keys() == expected_free.keys(),
            "Teacher prompt coverage mismatch")
    for pid, row in prompts.items():
        expected = expected_free[pid]
        require(row["candidates"] == expected["candidates"] and row["candidate_semantics"] == expected["candidate_semantics"]
                and row["messages_sha256"] == expected["messages_sha256"]
                and row["source_id"] == expected["source_id"] != args.heldout_source,
                "Teacher prompt source/menu/content mismatch")
        require(len(row["complete_candidate_logps"]) == len(row["logq"]) == len(expected["candidates"])
                and len(row["suffix_token_lengths"]) == len(expected["candidates"]), "Incomplete candidate cache")
        require(all(math.isfinite(x) for x in row["complete_candidate_logps"]) and
                all(type(n) is int and n > 0 for n in row["suffix_token_lengths"]), "Invalid full-candidate likelihood")
        top = max(row["complete_candidate_logps"])
        norm = top + math.log(sum(math.exp(x - top) for x in row["complete_candidate_logps"]))
        require(all(abs(a - (b - norm)) < 1e-10 for a, b in zip(row["logq"], row["complete_candidate_logps"])),
                "Teacher cache used another normalization or length correction")
    rows = read(directory / "teacher_condition_distributions.jsonl")
    teacher = {row["condition_id"]: row for row in rows}
    require(len(teacher) == len(rows) == len(prepared[5]), "Teacher condition coverage differs")
    for condition in prepared[5]:
        row = teacher[condition["condition_id"]]
        require(row["prompt_ids"] == condition["prompt_ids"] and row["candidates"] == condition["candidates"],
                "Teacher candidate/presentation alignment changed")
        require(row["source_id"] != args.heldout_source, "Held-out source in teacher")
        require(all(math.isfinite(x) for x in row["teacher_logq"]), "Invalid cached teacher vector")
        require(abs(sum(math.exp(x) for x in row["teacher_logq"]) - 1) < 1e-7, "Cached teacher mass invalid")
        for index, logp in enumerate(row["teacher_logq"]):
            average = sum(math.exp(prompts[pid]["logq"][index]) for pid in row["prompt_ids"]) / len(row["prompt_ids"])
            require(abs(math.exp(logp) - average) < 1e-9, "Teacher consensus is not arithmetic presentation mean")
    return teacher, manifest, sha_file(manifest_path)


def run_training(args, prepared, output):
    teacher, teacher_manifest, teacher_hash = load_teacher(args, prepared)
    require(not (output / "training_started.json").exists(), "Already started; no automatic optimizer/RNG resume")
    legacy, sup, sup_edges, free, free_edges, conditions, schedule, audit = prepared
    torch, model, tokenizer, cache, identity = initialize_backend(args, prepared, output)
    # Tokenizer and scoring identity must match the frozen teacher, not just its model name.
    teacher_identity = teacher_manifest["config"]["identity"]
    for name in ("base_model_path", "base_model_config_sha256", "chat_template_sha256", "dtype",
                 "enable_thinking", "candidate_likelihood_policy", "legacy_helper_sha256", "runner_sha256",
                 "torch_version", "transformers_version", "base_weight_metadata_sha256"):
        require(identity[name] == teacher_identity[name], f"Teacher/student interface differs: {name}")
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                            target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    require(trainable and all("lora_" in name for name, _ in trainable), "Non-LoRA parameters would update")
    model.cuda().train()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=0.01)
    by_id = {row["prompt_id"]: row for row in sup + free}
    sup_by_id = {edge["edge_id"]: edge for edge in sup_edges}
    free_by_id = {edge["edge_id"]: edge for edge in free_edges}
    lambda_js, lambda_kl = ARMS[args.arm]
    config = {"protocol": PROTOCOL, "mode": args.arm, "arm": args.arm, "seed": args.seed,
              "model": identity["base_model_path"], "base_model_config_sha256": identity["base_model_config_sha256"],
              "input_sha256": audit["input_sha256"], "split_content_sha256": audit["split_content_sha256"],
              "teacher_manifest_sha256": teacher_hash, "identity": identity,
              "exposure_schedule_sha256": audit["exposure_schedule_sha256"], "rank": 8,
              "alpha": 16, "target_modules": ["q_proj", "v_proj"], "dropout": 0.0,
              "learning_rate": 1e-4, "gradient_accumulation": 8, "batch_size": 1,
              "bf16": True, "gradient_checkpointing": True, "max_steps": args.max_steps,
              "lambda_free_js": lambda_js, "lambda_free_anchor": lambda_kl,
              "supervised_js": False, "supervised_loss": audit["supervised_loss"],
              "same_exposure": "all arms compute both free distributions and both penalties; zero coefficients retain zero-gradient path",
              "trainable_parameters": sum(p.numel() for _, p in trainable),
              "confirmation_status": "not_new_confirmation; first-stage evidence already inspected"}
    atomic_json(output / "training_config.json", config)
    atomic_json(output / "training_started.json", {"unix": time.time(), "config_sha256": digest(config)})
    logteacher = {cid: torch.tensor(row["teacher_logq"], dtype=torch.float32, device="cuda")
                  for cid, row in teacher.items()}
    def supervised_loss(row):
        scores, lengths = candidate_logps(torch, model, cache, legacy, row, row["expected"])
        # EXACT D0 from the frozen first-stage SFT. No supervised-JS term.
        return -torch.logsumexp(scores, dim=0) / (sum(lengths) / len(lengths))
    def free_distribution(row):
        scores, _ = candidate_logps(torch, model, cache, legacy, row)
        return torch.log_softmax(scores, dim=0)
    started, last_checkpoint = time.time(), None
    exposures = collections.Counter()
    optimizer.zero_grad(set_to_none=True)
    atomic_json(output / "progress.json", {"status": "training", "completed_optimizer_steps": 0,
                                            "planned_optimizer_steps": args.max_steps, "resumable": False})
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(1, args.max_steps + 1):
            metrics = collections.defaultdict(float)
            for event in schedule[(step - 1) * 8:step * 8]:
                se, fe = sup_by_id[event["supervised_edge_id"]], free_by_id[event["free_edge_id"]]
                sa, sb = by_id[se["left"]], by_id[se["right"]]
                fa, fb = by_id[fe["left"]], by_id[fe["right"]]
                require(all(r["source_id"] != args.heldout_source for r in (sa, sb, fa, fb)), "Runtime source violation")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sup_loss = (supervised_loss(sa) + supervised_loss(sb)) / 2
                    pa, pb = free_distribution(fa), free_distribution(fb)
                    js, kl = js_and_anchor(torch, pa, pb, logteacher[event["condition_id"]])
                    # All arms keep the same free forward/backward computation graph,
                    # even when its coefficients are zero. This adds NO hard behavior label.
                    total = sup_loss + lambda_js * js + lambda_kl * kl + 0.0 * (pa.sum() + pb.sum())
                require(bool(torch.isfinite(total)), "Nonfinite loss")
                (total / 8).backward()
                for key, value in (("D0_supervised", sup_loss), ("free_js_proxy", js), ("free_anchor_kl_proxy", kl), ("total_loss", total)):
                    metrics[key] += float(value.detach()) / 8
                entropy = -0.5 * ((pa.exp() * pa).sum() + (pb.exp() * pb).sum())
                metrics["free_entropy_proxy"] += float(entropy.detach()) / 8
                for pid in (se["left"], se["right"], fe["left"], fe["right"]):
                    exposures[pid] += 1
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            require(bool(torch.isfinite(norm)), "Nonfinite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, **metrics, "gradient_norm": float(norm), "elapsed_seconds": time.time() - started,
                      "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated()}
            stream.write(canonical(record) + "\n")
            stream.flush()
            if step % 25 == 0 or step == args.max_steps:
                target = output / f"adapter_checkpoint_{step:03d}"
                require(not target.exists(), "Checkpoint exists; cannot overwrite")
                model.save_pretrained(target, safe_serialization=True)
                atomic_json(target / "checkpoint_metadata.json", {
                    "completed_optimizer_steps": step, "training_config_sha256": sha_file(output / "training_config.json"),
                    "split_content_sha256": audit["split_content_sha256"], "teacher_manifest_sha256": teacher_hash,
                    "adapter_files": {p.name: sha_file(p) for p in target.iterdir() if p.is_file()},
                    "resumable": False, "optimizer_saved": False, "rng_saved": False})
                last_checkpoint = target.name
            atomic_json(output / "progress.json", {"status": "training", "completed_optimizer_steps": step,
                        "planned_optimizer_steps": args.max_steps, "last_adapter_checkpoint": last_checkpoint,
                        "resumable": False, "record": record})
            if step == 1 or step % 10 == 0:
                print(canonical(record), flush=True)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "tokenizer")
    atomic_json(output / "actual_exposures.json", {"schedule_sha256": audit["exposure_schedule_sha256"],
                "prompt_exposure_counts": dict(sorted(exposures.items())), "total_prompt_exposures": sum(exposures.values())})
    complete = {"status": "training_complete_not_yet_evaluated", "steps": args.max_steps,
                "config_sha256": sha_file(output / "training_config.json"),
                "split_content_sha256": audit["split_content_sha256"], "teacher_manifest_sha256": teacher_hash,
                "adapter_files": {p.name: sha_file(p) for p in (output / "adapter").iterdir() if p.is_file()},
                "exposures_sha256": sha_file(output / "actual_exposures.json"),
                "claim": "No claim of improved generation reliability or new confirmation success"}
    atomic_json(output / "training_complete.json", complete)
    atomic_json(output / "progress.json", {**complete, "completed_optimizer_steps": args.max_steps,
                                            "last_adapter_checkpoint": last_checkpoint, "resumable": False})


def self_test():
    # Small CPU tensors only; no model weights and no CUDA initialization.
    import torch
    a = torch.tensor([1.0, -0.5, 0.2], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-0.1, 0.4, 0.0], dtype=torch.float64, requires_grad=True)
    teacher = torch.log(torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64, requires_grad=True))
    js, kl = js_and_anchor(torch, torch.log_softmax(a, 0), torch.log_softmax(b, 0), teacher)
    require(0 <= float(js.detach()) <= math.log(2) + 1e-12 and float(kl.detach()) >= 0, "Invalid divergence")
    (js + kl).backward()
    require(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all(), "Nonfinite loss gradient")
    # Anchor must not propagate into the teacher, even if caller accidentally enabled its gradient.
    t = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    _, k = js_and_anchor(torch, torch.log_softmax(c, 0), torch.log_softmax(c, 0), t.log())
    k.backward()
    require(t.grad is None, "Teacher received gradient")
    # When student equals its condition anchor, both divergences vanish.
    target = torch.log(torch.tensor([0.25, 0.75], dtype=torch.float64))
    j0, k0 = js_and_anchor(torch, target, target, target)
    require(abs(float(j0)) < 1e-12 and abs(float(k0)) < 1e-12, "Zero-divergence identity failed")
    # Correct-set D0 includes both legal correct answers and differentiates.
    scores = torch.tensor([-2.0, -3.0], dtype=torch.float64, requires_grad=True)
    d0 = -torch.logsumexp(scores, 0) / 5.5
    d0.backward()
    require(torch.all(scores.grad < 0) and torch.isfinite(scores.grad).all(), "Correct-set D0 gradients invalid")
    # Real-condition separation: equivalence unions cannot bridge A and C.
    rows = [{"prompt_id": p, "source_id": "train", "field": "action", "candidates": [0, 1],
             "candidate_semantics": [0, 1]} for p in "ABCD"]
    edges = [{"left": "A", "right": "B"}, {"left": "C", "right": "D"}]
    conditions = connected_conditions(rows, edges, [("A", "C", "pair")])
    require(len(conditions) == 2, "Different conditions merged")
    try:
        connected_conditions(rows, [{"left": "A", "right": "B"}, {"left": "B", "right": "C"},
                                     {"left": "C", "right": "D"}], [("A", "C", "pair")])
    except ValueError:
        pass
    else:
        raise AssertionError("Real-condition bridge accepted")
    require(not torch.cuda.is_initialized(), "CPU loss self-test initialized CUDA")
    print(canonical({"self_test": "passed", "device": "cpu", "checks": ["JS_KL_finite_backward",
                    "teacher_detached", "identity_zero_loss", "D0_correct_set_gradient", "no_AC_union", "no_CUDA"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE.parent / "baseline_v2")
    parser.add_argument("--legacy-helper-dir", type=Path,
                        default=Path(os.environ.get("INVAR_METHODS_V1", str(HERE.parent / "methods_v1"))))
    parser.add_argument("--model", help="Existing local Qwen3-8B weights, no downloads")
    parser.add_argument("--heldout-source", default="HPVBG")
    parser.add_argument("--arm", choices=sorted(ARMS), default="anchor_sft")
    parser.add_argument("--teacher-cache", type=Path, default=HERE / "teacher_cache")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-free-candidates", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tokenizer-audit", action="store_true",
                        help="With --prepare-only --model: inspect all training candidate lengths on CPU, no CUDA/model weights")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(0 < args.max_steps <= 150, "Frozen pilot budget is 1-150 optimizer steps")
    require(1 < args.max_free_candidates <= 16, "This pilot permits at most 16 COMPLETE legal free candidates")
    require(not (args.prepare_only and args.teacher_only), "Preparation and GPU teacher build are different modes")
    output = args.output or HERE / "learning" / f"{args.arm}_seed{args.seed}"
    prepared = prepare(args)
    save_preparation(output, prepared)
    print(canonical(prepared[-1]), flush=True)
    if args.prepare_only:
        if args.tokenizer_audit:
            require(args.model is not None, "--tokenizer-audit requires --model")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
            _, token_audit = prepared[0].tokenize_examples(prepared[1] + prepared[3], tokenizer, args.max_length)
            atomic_json(output / "tokenization_audit.json", token_audit)
            print(canonical(token_audit), flush=True)
        return
    require(args.model is not None, "--model is required for GPU operations")
    if args.teacher_only:
        build_teacher(args, prepared)
    else:
        try:
            run_training(args, prepared, output)
        except Exception as error:
            progress_path = output / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
            progress.update(status="failed_no_automatic_resume", error=str(error), resumable=False)
            atomic_json(progress_path, progress)
            raise


if __name__ == "__main__":
    main()
