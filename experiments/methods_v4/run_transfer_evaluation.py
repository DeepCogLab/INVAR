"""Actual vLLM source/model/seed transfer evaluation, no oracle inference input.

Independent method sampling follows V3. A full-data baseline can be generated
once per model/inference-seed and projected by a separately audited staging
step; this runner never calls a projected response newly generated.

Each invocation runs one method. Every engine, including baseline, enables the
same rank-8 LoRA machinery. No frozen V1/V2/V3 source file is modified.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1 = Path(os.environ.get("INVAR_METHODS_V1", HERE.parent / "methods_v1")).resolve()
V2 = Path(os.environ.get("INVAR_METHODS_V2", HERE.parent / "methods_v2")).resolve()
sys.path.insert(0, str(V1))
sys.path.insert(0, str(V2))
from run_prompt_methods import (read_rows, digest, sanitized_messages, validate_output,
                               append_rows, atomic_json, checkpoint_fingerprint, load_baseline_helpers)


VERSION = "INVAR_transfer_actual_generation_v4"
TRAINING_PROTOCOL = "INVAR_retrospective_source_model_seed_transfer_20261026_v1"
V2_TRAINING_PROTOCOL = "INVAR_condition_anchor_four_arm_20261026_v1"
METHODS = ("baseline", "anchor_sft", "anchor_js")
MODEL_TYPES = {"qwen3_8b": "qwen3", "qwen25_14b_instruct": "qwen2"}
SOURCES = {"HPVBG", "IGVOFO", "PGG", "PIQ572"}
TRAINING_SEEDS = {20261026, 20261028, 20261029}
EXCLUDED_VARIANTS = {"V4_rules_table", "V6_relation"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def import_file(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def seed_for(model, method, run_key, pid, replicate, inference_seed):
    """Independent arm/run streams; training seed never overrides inference seed."""
    identity = [VERSION, inference_seed, model, method, run_key, pid, replicate]
    return int(digest(identity)[:8], 16) % (2 ** 31 - 1)


def current_messages(row):
    # No serialization of row IDs, provenance, correct answers, target outcomes,
    # source conditions, teacher distributions, or training examples.
    return sanitized_messages({"messages": row["messages"]})


def provenance(baseline, inputs):
    pairs = {row["pair_id"]: row for row in read_rows(baseline / "INVAR_dataset/data/pairs.jsonl")}
    references = collections.defaultdict(set)
    for file in (baseline / "INVAR_dataset/data/quartets.jsonl", baseline / "fact_quartets.jsonl",
                 baseline / "goal_quartets.jsonl"):
        for q in read_rows(file):
            for side in "ABCD":
                references[q[side]].add(q["pair_id"])
    for pid, row in inputs.items():
        if row.get("pair_id"):
            references[pid].add(row["pair_id"])
        for use in row.get("uses", []):
            references[pid].add(use["pair_id"])
    sources = {}
    for pid in inputs:
        require(references[pid] and references[pid] <= pairs.keys(), f"Unresolved source closure: {pid}")
        sources[pid] = sorted({pairs[pair]["source_id"] for pair in references[pid]})
    return sources


def load_plan(baseline, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(sha(baseline / "inputs.jsonl") == manifest["baseline_inputs_sha256"], "Frozen baseline input hash changed")
    all_rows = read_rows(baseline / "inputs.jsonl")
    full = {row["prompt_id"]: row for row in all_rows}
    require(len(full) == len(all_rows), "Duplicate baseline prompt ID")
    selected = manifest["selected_prompt_ids"]
    require(isinstance(selected, list) and selected and len(set(selected)) == len(selected)
            and set(selected) <= full.keys(), "Manifest requires a nonempty unique subset of actual prompt IDs")
    repetitions = manifest["repetitions"]
    rows = [full[pid] for pid in selected]
    for row in rows:
        count = repetitions.get(row["task_type"])
        require(type(count) is int and count > 0, f"Invalid task budget: {row['task_type']}")
        require(isinstance(row.get("output_schema"), dict), "Public JSON schema missing")
        current_messages(row)
    total = sum(repetitions[row["task_type"]] for row in rows)
    if type(manifest.get("calls_per_model_method")) is int:
        require(total == manifest["calls_per_model_method"], "Manifest call count differs from selected subset/budgets")
    sources = provenance(baseline, full)
    if "provenance_sources" in manifest:
        for pid in selected:
            require(manifest["provenance_sources"][pid] == sources[pid], "Manifest source closure changed")
    plan = {"selected_prompt_ids": selected, "repetitions": repetitions, "planned_rows": total,
            "unique_inputs": len(rows), "inputs_by_type": dict(collections.Counter(row["task_type"] for row in rows)),
            "provenance_sources": {pid: sources[pid] for pid in selected},
            "scope": "retrospective_development_transfer_not_new_confirmation"}
    return rows, manifest, plan


def validate_v4_training(adapter, base_path, fingerprint, method, model, baseline, validator):
    directory = adapter.parent
    training_path, audit_path = directory / "training_config.json", directory / "data_audit.json"
    training = json.loads(training_path.read_text(encoding="utf-8-sig"))
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    require(training["protocol"] == audit["protocol"] == TRAINING_PROTOCOL, "Unknown transfer training protocol")
    require(training["mode"] == training["arm"] == method, "Adapter arm label differs from training")
    require(training["model_id"] == audit["model_id"] == model, "Adapter trained for a different model")
    heldout = training["heldout_source"]
    require(heldout in SOURCES and audit["heldout_source"] == heldout, "Held-out source differs")
    require(training["seed"] == training["training_seed"] == audit["training_seed"] == audit["seed"] in TRAINING_SEEDS,
            "Training seed differs from the frozen replication design")
    require(training["train_sources"] == audit["train_sources"] == sorted(SOURCES - {heldout}), "Training source pool differs")
    require(training["teacher_required"] is False and training["teacher_manifest_sha256"] is None
            and audit["teacher_required"] is False and audit["teacher_manifest_sha256"] is None,
            "This V4 SFT/JS comparison has no teacher/KL dependency")
    require(training["lambda_free_anchor"] == 0.0 and training["lambda_free_js"] == (0.0 if method == "anchor_sft" else 1.0),
            "Training arm coefficients differ")
    require(training["supervised_js"] is False and audit["supervised_js"] is False, "Unexpected supervised JS")
    require(training["rank"] == 8 and set(training["target_modules"]) == {"q_proj", "v_proj"}, "Unexpected LoRA rank/targets")
    require(training["max_steps"] == audit["max_optimizer_steps"] == 150, "Transfer training step budget changed")
    require(set(audit["heldout_variants"]) == EXCLUDED_VARIANTS and audit["max_free_candidates"] == 16,
            "Training variant/support exclusion changed")
    for key in ("heldout_source_rows_in_training", "mixed_source_rows_in_training", "heldout_variant_edges_in_training",
                "free_behavior_hard_labels", "real_condition_bridges"):
        require(audit[key] == 0, f"Training isolation audit failed: {key}")
    require(training["input_sha256"] == audit["input_sha256"] == sha(baseline / "inputs.jsonl"), "Training inputs differ")
    require(training["split_content_sha256"] == audit["split_content_sha256"] and training["source_pool_sha256"] == audit["source_pool_sha256"],
            "Training split/source pool identity differs")
    audit_body = {key: value for key, value in audit.items() if key not in {"split_content_sha256", "prepared_file_hashes"}}
    require(digest(audit_body) == audit["split_content_sha256"], "Split audit content hash mismatch")
    require(training["exposure_schedule_sha256"] == audit["exposure_schedule_sha256"], "Training exposure schedule differs")
    require(Path(training["model"]).resolve() == base_path.resolve()
            and training["base_model_config_sha256"] == sha(base_path / "config.json"), "Base weight/config identity differs")
    identity = training["identity"]
    require(identity["model_id"] == model and identity["model_type"] == MODEL_TYPES[model]
            and identity["base_model_path"] == str(base_path.resolve()), "Native model identity differs")
    require(identity["base_model_config_sha256"] == sha(base_path / "config.json"), "Training native configuration changed")
    require(identity["template_kwargs"] == ({"enable_thinking": False} if model == "qwen3_8b" else {})
            and identity["native_template"] is True and identity["dtype"] == "bfloat16", "Training template/precision differs")
    weight_metadata = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(base_path.glob("*.safetensors"))]
    require(identity["base_weight_metadata_sha256"] == digest(weight_metadata)
            and digest(identity["base_weight_file_metadata"]) == digest(weight_metadata), "Base weight file metadata changed")
    # Read-only reuse of the frozen independently implemented provenance checker.
    prepared = validator.verify_prepared_source_closure(directory, audit, baseline, heldout)
    pool = {"supervised_inputs": prepared["supervised_inputs.jsonl"], "supervised_edges": prepared["supervised_edges.jsonl"],
            "free_inputs": prepared["free_inputs.jsonl"], "free_edges": prepared["free_edges.jsonl"],
            "conditions": prepared["semantic_conditions.jsonl"]}
    require(digest(pool) == audit["source_pool_sha256"], "Prepared source pool differs")
    metadata_path = adapter / "checkpoint_metadata.json"
    is_checkpoint = metadata_path.exists()
    if not is_checkpoint:
        metadata_path = directory / "training_complete.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    require(metadata["adapter_files"]["adapter_model.safetensors"] == fingerprint["adapter_weights_sha256"]
            and metadata["adapter_files"]["adapter_config.json"] == fingerprint["adapter_config_sha256"], "Saved adapter byte hashes differ")
    config_key = "training_config_sha256" if is_checkpoint else "config_sha256"
    require(metadata[config_key] == sha(training_path), "Saved adapter training config hash differs")
    require(metadata["split_content_sha256"] == audit["split_content_sha256"]
            and metadata["model_id"] == model and metadata["heldout_source"] == heldout
            and metadata["training_seed"] == training["seed"] and metadata["teacher_required"] is False
            and metadata["teacher_manifest_sha256"] is None, "Saved adapter split/model/seed differs")
    if is_checkpoint:
        require(0 < metadata["completed_optimizer_steps"] <= 150, "Invalid checkpoint step")
    else:
        require(metadata["steps"] == 150 and metadata["status"] == "training_complete_not_yet_evaluated", "Training is incomplete")
        actual_path = directory / "actual_exposures.json"
        require(metadata["exposures_sha256"] == sha(actual_path), "Actual exposure file hash differs")
        actual = json.loads(actual_path.read_text(encoding="utf-8-sig"))
        require(actual["schedule_sha256"] == audit["exposure_schedule_sha256"], "Actual exposure schedule differs")
        edge_maps = [{edge["edge_id"]: edge for edge in prepared[name]}
                     for name in ("supervised_edges.jsonl", "free_edges.jsonl")]
        counts = collections.Counter()
        for event in prepared["exposure_schedule.jsonl"]:
            for key, edges in zip(("supervised_edge_id", "free_edge_id"), edge_maps):
                edge = edges[event[key]]
                counts.update((edge["left"], edge["right"]))
        require(actual["prompt_exposure_counts"] == dict(counts) and actual["total_prompt_exposures"] == sum(counts.values()),
                "Actual exposures differ from the declared common schedule")
    return {"training_protocol": TRAINING_PROTOCOL, "training_mode": method, "model_id": model,
            "training_seed": training["seed"], "heldout_source": heldout, "teacher_required": False,
            "training_input_sha256": audit["input_sha256"], "training_split_sha256": audit["split_content_sha256"],
            "source_pool_sha256": audit["source_pool_sha256"], "training_chat_template_sha256": identity["chat_template_sha256"],
            "training_config_sha256": sha(training_path), "data_audit_sha256": sha(audit_path),
            "adapter_checkpoint_step": metadata.get("completed_optimizer_steps", metadata.get("steps")),
            "source_closure_verified": True, "scope": "retrospective_transfer_all_sources_previously_inspected"}


def adapter_identity(adapter, method, model, weights, baseline, teacher_cache):
    validator = import_file("invar_frozen_v2_transfer_validation", V2 / "run_anchor_evaluation.py")
    fingerprint = validator.adapter_fingerprint(adapter)
    training_path = adapter.parent / "training_config.json"
    training = json.loads(training_path.read_text(encoding="utf-8-sig"))
    if training["protocol"] == TRAINING_PROTOCOL:
        identity = validate_v4_training(adapter, weights, fingerprint, method, model, baseline, validator)
    elif training["protocol"] == V2_TRAINING_PROTOCOL:
        require(model == "qwen3_8b", "Existing V2 anchor adapters belong to Qwen3-8B")
        identity = validator.validate_training_base(adapter, weights, fingerprint, method, "HPVBG", baseline, teacher_cache)
        identity.update(training_protocol=V2_TRAINING_PROTOCOL, training_seed=training["seed"], model_id=model,
                        heldout_source="HPVBG", teacher_required=True)
    else:
        raise ValueError("Unknown adapter training protocol")
    return {"path": str(adapter.resolve()), "fingerprint": fingerprint, "training_identity": identity}


def inspect_saved(file, config_hash, model, method, run_key, inference_seed, rows, plan, metadata):
    by_id = {row["prompt_id"]: row for row in rows}
    seen = set()
    if not file.exists():
        return seen
    for record in read_rows(file):
        pid, replicate = record["prompt_id"], record["replicate"]
        require(record["config_sha256"] == config_hash and record["model"] == model and record["method"] == method,
                "Saved response belongs to another immutable configuration")
        require(pid in by_id and type(replicate) is int and 0 <= replicate < plan["repetitions"][by_id[pid]["task_type"]],
                "Saved response is outside current plan")
        trial_id = f"{model}|{method}|{pid}|{replicate}"
        require(record["trial_id"] == trial_id and trial_id not in seen, "Duplicate or malformed trial")
        require(record["status"] == "ok" and record.get("batch", "main") == "main", "Saved failure must be investigated, never silently rerolled")
        require(record["sampling_seed"] == seed_for(model, method, run_key, pid, replicate, inference_seed), "Saved inference seed differs")
        require(record["messages_sha256"] == digest(current_messages(by_id[pid])), "Saved prompt differs from current original messages")
        require(record["source_partition"] == plan["source_partitions"][pid] and record["fallback"] is False, "Saved partition/prompt policy differs")
        if metadata:
            require(record["adapter_fingerprint_sha256"] == metadata["fingerprint"]["content_sha256"], "Saved adapter fingerprint differs")
        else:
            require(record.get("adapter_fingerprint_sha256") is None and record.get("lora_request_id") is None,
                    "Baseline response contains an adapter binding")
        validate_output(record["response_text"], by_id[pid]["output_schema"], record.get("finish_reason"))
        seen.add(trial_id)
    return seen


def complete(file, identity, rows, metadata, run_key, inference_seed):
    config = identity["config"]
    plan = identity["plan"]
    seen = inspect_saved(file, identity["config_sha256"], config["model"], config["method"], run_key,
                         inference_seed, rows, plan, metadata)
    require(len(seen) == plan["planned_rows"], "Cannot complete a partial generation file")
    saved = read_rows(file)
    summary = {"protocol": VERSION, "model": config["model"], "method": config["method"], "status": "complete",
               "rows": len(saved), "config_sha256": identity["config_sha256"], "responses_sha256": sha(file),
               "inference_seed": inference_seed, "adapter_run_key": run_key,
               "fallback_prompts": 0, "fallback_rows": 0,
               "decision_input_tokens": sum(row["input_tokens"] for row in saved),
               "decision_completion_tokens": sum(row["completion_tokens"] for row in saved),
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "generation_origin": "actual_vllm_calls_not_subset_projection",
               "confirmation_status": "retrospective_development_transfer"}
    atomic_json(file.with_suffix(".complete.json"), summary)
    failure = file.parent / f"{config['method']}.FAILED.json"
    if failure.exists():
        failure.replace(file.parent / f"{config['method']}.FAILED.resolved_{time.time_ns()}.json")
    return summary


def run(args):
    require(args.method in METHODS and args.model in MODEL_TYPES, "Unapproved model/method")
    require(type(args.inference_seed) is int and 0 <= args.inference_seed < 2 ** 31, "Explicit nonnegative inference seed required")
    require((args.adapter is None) == (args.method == "baseline"), "Baseline uses no adapter; each other method requires --adapter")
    rows, manifest, plan = load_plan(args.baseline, args.manifest)
    if "model_methods" in manifest:
        require(args.method in manifest["model_methods"].get(args.model, []), "Method absent from model manifest")
    model_paths, render_prompt = load_baseline_helpers(args.baseline)
    weights = Path(model_paths[args.model]).resolve()
    native = json.loads((weights / "config.json").read_text(encoding="utf-8-sig"))
    require(native["model_type"] == MODEL_TYPES[args.model], "Unexpected native model architecture")
    context = min(8192, native.get("max_position_embeddings", 8192))
    max_batch = 64 if args.model == "qwen25_14b_instruct" else 128
    batch_size = args.batch_size or max_batch
    require(type(batch_size) is int and 1 <= batch_size <= max_batch, f"Batch-size must be between 1 and {max_batch}")
    metadata = adapter_identity(args.adapter.resolve(), args.method, args.model, weights, args.baseline.resolve(),
                                args.teacher_cache.resolve()) if args.adapter else None
    heldout = metadata["training_identity"]["heldout_source"] if metadata else manifest.get("heldout_source")
    declared = manifest.get("heldout_source", manifest.get("learning", {}).get("heldout_source"))
    if metadata and declared:
        require(declared == heldout, "Manifest held-out source differs from trained adapter")
    if metadata and manifest.get("evaluation_scope") == "heldout_source_only":
        require(all(heldout in sources for sources in plan["provenance_sources"].values()), "Heldout-only manifest contains another source")
    plan["heldout_source"] = heldout
    plan["source_partitions"] = {pid: ("heldout_source" if heldout and heldout in sources else
                                      "development_seen_source" if metadata else "baseline_reference")
                                 for pid, sources in plan["provenance_sources"].items()}
    plan["content_sha256"] = digest(plan)
    run_key = metadata["fingerprint"]["content_sha256"] if metadata else "baseline_unadapted_shared_reference"
    code_files = [Path(__file__), V1 / "run_prompt_methods.py", args.baseline / "INVAR_dataset/run_vllm_batch.py"]
    if metadata:
        code_files.extend([V2 / "run_anchor_evaluation.py", V2 / "run_normalized_methods.py"])
    config = {"protocol": VERSION, "model": args.model, "method": args.method,
              "base_seed": args.inference_seed, "inference_seed": args.inference_seed,
              "adapter_run_key": run_key, "adapter": metadata,
              "weights": checkpoint_fingerprint(weights), "temperature": .7, "top_p": .95,
              "top_k": -1, "repetition_penalty": 1., "max_tokens": 128, "dtype": "bfloat16",
              "decoder": "json_schema_xgrammar_compact", "batch_size": batch_size, "context": context,
              "engine_enable_lora": True, "max_lora_rank": 8, "max_loras": 1,
              "gpu_memory_utilization": .90, "enable_prefix_caching": True, "generation_config": "vllm",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "input_sha256": manifest["baseline_inputs_sha256"], "manifest_sha256": sha(args.manifest),
              "code_hashes": {str(path.resolve()): sha(path) for path in code_files},
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")},
              "seed_policy": "sha256(protocol,inference_seed,model,method,adapter_run_key,prompt_id,replicate); independent methods",
              "correctness_labels_in_inference": False, "prompt_policy": "original current messages and public legal menu only",
              "scope": "retrospective_development_transfer; all source materials previously inspected"}
    config = json.loads(json.dumps(config))
    directory = args.out / args.model
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / f"{args.method}.run_identity.json"
    identity = {"config": config, "config_sha256": digest(config), "plan": plan}
    if identity_path.exists():
        prior = json.loads(identity_path.read_text(encoding="utf-8-sig"))
        require(prior["config"] == config and prior["plan"] == plan and prior["config_sha256"] == identity["config_sha256"],
                "Configuration/adapter/plan changed; use another output directory")
        identity = prior
    else:
        atomic_json(identity_path, identity)
    raw = directory / f"{args.method}.jsonl"
    done = inspect_saved(raw, identity["config_sha256"], args.model, args.method, run_key,
                         args.inference_seed, rows, plan, metadata)
    if len(done) == plan["planned_rows"]:
        print(json.dumps(complete(raw, identity, rows, metadata, run_key, args.inference_seed)), flush=True)
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.lora.request import LoRARequest
    tokenizer = AutoTokenizer.from_pretrained(str(weights), trust_remote_code=True, local_files_only=True)
    if metadata:
        require(digest(tokenizer.chat_template) == metadata["training_identity"]["training_chat_template_sha256"],
                "Generation and training tokenizer templates differ")
    rendered, info = {}, {}
    for row in rows:
        pid = row["prompt_id"]
        messages = current_messages(row)
        text = render_prompt(tokenizer, messages, args.model)
        count = len(tokenizer.encode(text, add_special_tokens=False))
        require(count + 128 <= context, f"Context overflow at {pid}; no truncation permitted")
        rendered[pid] = text
        info[pid] = {"messages_sha256": digest(messages), "rendered_sha256": digest(text), "input_tokens": count,
                     "fallback": False, "fallback_reason": None}
    prompt_path = directory / f"{args.method}.prompts.json"
    if prompt_path.exists():
        require(json.loads(prompt_path.read_text(encoding="utf-8-sig")) == info, "Rendered prompt identity changed")
    else:
        atomic_json(prompt_path, info)
    llm = LLM(model=str(weights), tokenizer=str(weights), trust_remote_code=True, dtype="bfloat16",
              max_model_len=context, gpu_memory_utilization=.90, max_num_seqs=batch_size,
              enable_prefix_caching=True, generation_config="vllm", seed=args.inference_seed,
              enable_lora=True, max_lora_rank=8, max_loras=1,
              structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True})
    request = (LoRARequest("transfer_" + args.method + "_" + metadata["fingerprint"]["adapter_weights_sha256"][:12],
                           1, metadata["path"]) if metadata else None)
    grammars = {row["prompt_id"]: StructuredOutputsParams(json=row["output_schema"], disable_any_whitespace=True) for row in rows}
    todo = [(row, rep) for row in rows for rep in range(plan["repetitions"][row["task_type"]])
            if f"{args.model}|{args.method}|{row['prompt_id']}|{rep}" not in done]
    begun = time.time()
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        seeds = [seed_for(args.model, args.method, run_key, row["prompt_id"], rep, args.inference_seed) for row, rep in chunk]
        parameters = [SamplingParams(temperature=.7, top_p=.95, top_k=-1, repetition_penalty=1., max_tokens=128,
                                     seed=seed, structured_outputs=grammars[row["prompt_id"]]) for (row, _), seed in zip(chunk, seeds)]
        began_batch = time.time()
        outputs = None
        try:
            outputs = llm.generate([rendered[row["prompt_id"]] for row, _ in chunk], parameters,
                                   lora_request=request, use_tqdm=False)
            require(len(outputs) == len(chunk), "vLLM generated response count differs")
        except Exception as error:
            atomic_json(directory / f"{args.method}.failed_batch.json", {
                "stage": "generate", "config_sha256": identity["config_sha256"], "error": str(error),
                "trial_ids": [f"{args.model}|{args.method}|{row['prompt_id']}|{rep}" for row, rep in chunk],
                "partial_returned_outputs": ([{"request_id": getattr(output, "request_id", None),
                    "completions": [{"text": item.text, "finish_reason": item.finish_reason}
                                    for item in getattr(output, "outputs", [])]} for output in outputs]
                    if outputs is not None else None)})
            raise
        seconds = time.time() - began_batch
        records, invalid = [], []
        for (row, rep), output, seed in zip(chunk, outputs, seeds):
            pid = row["prompt_id"]
            binding = getattr(output, "lora_request", None)
            binding_ok = ((binding is None) if request is None else
                          binding is not None and binding.lora_int_id == request.lora_int_id and binding.lora_name == request.lora_name)
            completion = output.outputs[0] if len(output.outputs) == 1 else None
            record = {"model": args.model, "method": args.method, "prompt_id": pid, "task_type": row["task_type"],
                      "replicate": rep, "trial_id": f"{args.model}|{args.method}|{pid}|{rep}", "batch": "main",
                      "response_text": completion.text if completion else "", "status": "ok",
                      "finish_reason": completion.finish_reason if completion else None,
                      "completion_tokens": len(completion.token_ids) if completion else 0,
                      "sampling_seed": seed, "inference_seed": args.inference_seed,
                      "config_sha256": identity["config_sha256"], "source_partition": plan["source_partitions"][pid],
                      "adapter_fingerprint_sha256": metadata["fingerprint"]["content_sha256"] if metadata else None,
                      "lora_request_name": binding.lora_name if binding else None,
                      "lora_request_id": binding.lora_int_id if binding else None,
                      "batch_seconds": seconds, "batch_size_actual": len(chunk), **info[pid]}
            try:
                require(binding_ok and completion is not None, "Requested LoRA/baseline binding or completion is not confirmed")
                validate_output(record["response_text"], row["output_schema"], record["finish_reason"])
            except Exception as error:
                record.update(status="invalid_output", error=str(error))
                invalid.append(record["trial_id"])
            records.append(record)
        append_rows(raw, records)
        require(not invalid, f"Invalid output retained; stop without reroll: {invalid[:3]}")
        progress = {"model": args.model, "method": args.method, "status": "running",
                    "done": len(done) + start + len(chunk), "total": plan["planned_rows"],
                    "elapsed_seconds": time.time() - begun, "updated_unix": time.time(),
                    "config_sha256": identity["config_sha256"]}
        atomic_json(directory / f"{args.method}.progress.json", progress)
        if start // batch_size % 20 == 0:
            print(json.dumps(progress), flush=True)
    summary = complete(raw, identity, rows, metadata, run_key, args.inference_seed)
    failure = directory / f"{args.method}.FAILED.json"
    if failure.exists():
        failure.replace(directory / f"{args.method}.FAILED.resolved_{time.time_ns()}.json")
    print(json.dumps(summary), flush=True)


def self_test():
    schema = {"type": "object", "additionalProperties": False, "required": ["action"],
              "properties": {"action": {"type": "string", "enum": ["X", "Y"]}}}
    messages = [{"role": "system", "content": "按原规则作答。"}, {"role": "user", "content": "只选择X或Y。"}]
    poisoned = {"messages": copy.deepcopy(messages), "expected_answer": "SECRET_TRUTH", "variant_id": "SECRET_VARIANT"}
    require(current_messages(poisoned) == messages and "SECRET" not in json.dumps(current_messages(poisoned)), "Private metadata leakage")
    require(seed_for("m", "baseline", "base", "p", 0, 20261030) != seed_for("m", "anchor_js", "adapter", "p", 0, 20261030), "Method streams are not independent")
    require(seed_for("m", "anchor_js", "adapter1", "p", 0, 20261030) != seed_for("m", "anchor_js", "adapter2", "p", 0, 20261030), "Adapter runs reuse the same stream")
    require(seed_for("m", "baseline", "base", "p", 0, 20261030) == seed_for("m", "baseline", "base", "p", 0, 20261030), "Non-reproducible seed")
    with tempfile.TemporaryDirectory(prefix="invar_transfer_cpu_") as temp:
        root = Path(temp)
        (root / "INVAR_dataset/data").mkdir(parents=True)
        rows = [{"prompt_id": f"p{i}", "task_type": "behavior", "messages": copy.deepcopy(messages),
                 "output_schema": schema, "pair_id": "pair", "expected_answer": "SECRET"} for i in range(2)]
        def save_rows(path, objects):
            path.write_text("".join(json.dumps(row) + "\n" for row in objects), encoding="utf-8")
        save_rows(root / "inputs.jsonl", rows)
        save_rows(root / "INVAR_dataset/data/pairs.jsonl", [{"pair_id": "pair", "source_id": "PGG"}])
        save_rows(root / "INVAR_dataset/data/quartets.jsonl", [dict(A="p0", B="p0", C="p1", D="p1", pair_id="pair")])
        save_rows(root / "fact_quartets.jsonl", [])
        save_rows(root / "goal_quartets.jsonl", [])
        manifest_file = root / "manifest.json"
        atomic_json(manifest_file, {"baseline_inputs_sha256": sha(root / "inputs.jsonl"), "selected_prompt_ids": ["p1"],
                                   "repetitions": {"behavior": 3}, "calls_per_model_method": 3})
        chosen, _, plan = load_plan(root, manifest_file)
        require(len(chosen) == 1 and plan["planned_rows"] == 3 and plan["inputs_by_type"] == {"behavior": 1},
                "Arbitrary subset/budget or missing role/goal task handling failed")
        plan["source_partitions"] = {"p1": "baseline_reference"}
        config = {"model": "qwen3_8b", "method": "baseline"}
        identity = {"config": config, "config_sha256": "cfg", "plan": plan}
        file = root / "baseline.jsonl"
        records = [{"model": "qwen3_8b", "method": "baseline", "prompt_id": "p1", "replicate": rep,
                    "trial_id": f"qwen3_8b|baseline|p1|{rep}", "batch": "main", "config_sha256": "cfg",
                    "status": "ok", "response_text": '{"action":"X"}', "finish_reason": "stop",
                    "sampling_seed": seed_for("qwen3_8b", "baseline", "base", "p1", rep, 20261030),
                    "messages_sha256": digest(messages), "source_partition": "baseline_reference", "fallback": False,
                    "adapter_fingerprint_sha256": None, "lora_request_id": None, "input_tokens": 12, "completion_tokens": 5}
                   for rep in range(3)]
        append_rows(file, records)
        summary = complete(file, identity, chosen, None, "base", 20261030)
        require(summary["rows"] == 3 and summary["generation_origin"] == "actual_vllm_calls_not_subset_projection", "Completion manifest incompatible")
        for mutation in ({"config_sha256": "mixed"}, {"sampling_seed": -1}, {"messages_sha256": "wrong"},
                         {"adapter_fingerprint_sha256": "hidden-adapter"}, {"status": "invalid_output"},
                         {"replicate": 3}, {"fallback": True}):
            save_rows(file, [dict(records[0], **mutation)])
            try:
                inspect_saved(file, "cfg", "qwen3_8b", "baseline", "base", 20261030, chosen, plan, None)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid resume accepted: {mutation}")
        save_rows(file, [records[0], records[0]])
        try:
            inspect_saved(file, "cfg", "qwen3_8b", "baseline", "base", 20261030, chosen, plan, None)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate trial accepted")
    require(not any(name in sys.modules for name in ("torch", "transformers", "vllm")), "CPU tests imported GPU/model libraries")
    print(json.dumps({"self_test": "passed", "protocol": VERSION,
                      "checks": ["metadata_excluded", "arbitrary_subset_and_budget", "empty_role_goal_allowed",
                                 "independent_method_adapter_seed_streams", "resume_identity_schema_seed_validation",
                                 "duplicate_and_hidden_lora_rejected", "actual_generation_completion_schema", "no_gpu_imports"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_TYPES))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inference-seed", type=int)
    parser.add_argument("--teacher-cache", type=Path, default=V2 / "teacher_cache", help="Required only for existing V2 adapters")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("baseline", "manifest", "model", "method", "out", "inference_seed")):
        parser.error("--baseline --manifest --model --method --out --inference-seed required")
    try:
        run(args)
    except Exception as error:
        directory = args.out / args.model
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{args.method}.FAILED.json"
        if path.exists():
            path.replace(directory / f"{args.method}.FAILED.previous_{time.time_ns()}.json")
        atomic_json(path, {"protocol": VERSION, "model": args.model, "method": args.method,
                          "inference_seed": args.inference_seed, "error": f"{type(error).__name__}: {error}",
                          "unix": time.time(), "invalid_saved_outputs_never_rerolled": True})
        raise


if __name__ == "__main__":
    main()
