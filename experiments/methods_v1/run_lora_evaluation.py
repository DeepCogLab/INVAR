"""Evaluate one trained INVAR adapter through the unchanged vLLM interface.

No prompt intervention, truth labels, canonical case text, or trained outputs
are inserted into inference. --self-test imports no torch/transformers/vLLM.

Example, from the project root after training:
  python experiments/methods_v1/run_lora_evaluation.py \
    --baseline experiments/baseline_v2 --manifest experiments/methods_v1/split_manifest.json \
    --model qwen3_8b --method sft --adapter /path/to/run/adapter \
    --out experiments/methods_v1/prompt_screen

Outputs are <out>/<model>/<method>.jsonl and <method>.complete.json.
Identity/config/progress/failure files are method-specific and never replace
the prompt-method runner's identity.json, config.json or COMPLETE.json.
"""
from __future__ import annotations

import argparse
import collections
import copy
import importlib.metadata
import json
import sys
import tempfile
import time
from pathlib import Path

from run_prompt_methods import (
    BASE_SEED, REPETITIONS, append_rows, atomic_json, checkpoint_fingerprint,
    digest, json_text, load_baseline_helpers, read_rows, sanitized_messages,
    sha_bytes, trial_seed, validate_output,
)


METHODS = ("sft", "sft_consistency")
PROTOCOL = "INVAR_lora_generation_evaluation_v1_20261025"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    import hashlib
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def select_plan(baseline, manifest_path, include_all_heldout_source=False, heldout_source="HPVBG"):
    baseline = Path(baseline)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(sha_file(baseline / "inputs.jsonl") == manifest["baseline_inputs_sha256"],
            "Baseline inputs differ from frozen method manifest")
    rows = read_rows(baseline / "inputs.jsonl")
    by_id = {row["prompt_id"]: row for row in rows}
    require(len(rows) == len(by_id), "Duplicate baseline prompt IDs")
    selected = manifest["selected_prompt_ids"]
    require(isinstance(selected, list) and selected and len(set(selected)) == len(selected),
            "selected_prompt_ids must be nonempty and unique")
    require(set(selected) <= by_id.keys(), "Selected prompt ID absent from baseline")
    repetitions = manifest.get("repetitions", REPETITIONS)
    require(repetitions == REPETITIONS, "Repetition budget changed from prompt generation screen")
    # Verify source closure from original references rather than trusting a row's first pair_id.
    pairs = {row["pair_id"]: row for row in read_rows(baseline / "INVAR_dataset/data/pairs.jsonl")}
    require(heldout_source in {row["source_id"] for row in pairs.values()}, "Unknown held-out source")
    references = collections.defaultdict(set)
    for path in (baseline / "INVAR_dataset/data/quartets.jsonl",
                 baseline / "fact_quartets.jsonl", baseline / "goal_quartets.jsonl"):
        for quartet in read_rows(path):
            for side in "ABCD":
                references[quartet[side]].add(quartet["pair_id"])
    for pid, row in by_id.items():
        if row.get("pair_id"):
            references[pid].add(row["pair_id"])
        for use in row.get("uses", []):
            references[pid].add(use["pair_id"])
    sources = {}
    for pid in by_id:
        require(references[pid], f"Unresolved provenance: {pid}")
        require(references[pid] <= pairs.keys(), f"Unknown referenced condition pair: {pid}")
        sources[pid] = sorted({pairs[p]["source_id"] for p in references[pid]})
        require(sources[pid] == manifest["provenance_sources"][pid],
                f"Provenance closure changed: {pid}")
    selected = list(selected)
    extra = []
    if include_all_heldout_source:
        extra = sorted(pid for pid in by_id if heldout_source in sources[pid] and pid not in selected)
        selected.extend(extra)
    selected_rows = [by_id[pid] for pid in selected]
    for row in selected_rows:
        require(row["task_type"] in repetitions, "Unknown task type")
        require(type(repetitions[row["task_type"]]) is int and repetitions[row["task_type"]] > 0,
                "Invalid repetition count")
        sanitized_messages(row)
        require(isinstance(row.get("output_schema"), dict), "Missing public output schema")
    plan = {
        "selected_prompt_ids": selected, "screen_prompt_ids": manifest["selected_prompt_ids"],
        "extra_heldout_prompt_ids": extra, "heldout_source": heldout_source,
        "include_all_heldout_source": include_all_heldout_source,
        "repetitions": repetitions, "unique_inputs": len(selected_rows),
        "planned_rows": sum(repetitions[row["task_type"]] for row in selected_rows),
        "inputs_by_type": dict(collections.Counter(row["task_type"] for row in selected_rows)),
        "source_partitions": {pid: ("heldout_source" if heldout_source in sources[pid]
                                    else "development_seen_source") for pid in selected},
    }
    plan["content_sha256"] = digest(plan)
    return selected_rows, plan


def adapter_fingerprint(adapter):
    adapter = Path(adapter).resolve()
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    require(config_path.is_file() and weights_path.is_file(),
            "Expected PEFT adapter_config.json and adapter_model.safetensors")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM",
            "Expected causal-LM LoRA adapter")
    require(config.get("r") == 8, "This pilot is frozen at rank 8")
    require(set(config.get("target_modules", [])) == {"q_proj", "v_proj"},
            "Adapter targets must be q_proj and v_proj")
    require(not config.get("modules_to_save"), "Unexpected fully trained modules")
    require(config.get("bias", "none") == "none", "Unexpected trained bias parameters")
    require(not config.get("use_dora", False), "DoRA is outside this LoRA pilot")
    metadata_files = {}
    for name in ("training_config.json", "data_audit.json", "training_complete.json"):
        path = adapter.parent / name
        if path.is_file():
            metadata_files[name] = sha_file(path)
    metadata = adapter / "checkpoint_metadata.json"
    if metadata.is_file():
        metadata_files["checkpoint_metadata.json"] = sha_file(metadata)
    fingerprint = {
        "resolved_adapter": str(adapter), "adapter_config_sha256": sha_file(config_path),
        "adapter_weights_sha256": sha_file(weights_path), "weight_bytes": weights_path.stat().st_size,
        "training_metadata_sha256": metadata_files, "rank": 8,
        "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": config.get("base_model_name_or_path"),
    }
    fingerprint["content_sha256"] = digest(fingerprint)
    return fingerprint


def validate_training_base(adapter, base_path, fingerprint, method, heldout_source):
    adapter = Path(adapter).resolve()
    training_path = adapter.parent / "training_config.json"
    audit_path = adapter.parent / "data_audit.json"
    require(training_path.is_file() and audit_path.is_file(),
            "Training configuration and split audit required beside adapter/checkpoint directory")
    training = json.loads(training_path.read_text(encoding="utf-8-sig"))
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    require(training["mode"] == method, "Method label does not match trained adapter")
    require(audit["heldout_source"] == heldout_source, "Training held-out source differs from evaluation")
    require(training["split_content_sha256"] == audit["split_content_sha256"], "Training split hash mismatch")
    require(audit["heldout_source_rows_in_training"] == 0 and audit["mixed_source_rows_in_training"] == 0,
            "Training provenance audit contains contamination")
    require(audit["behavior_labels_used"] == 0, "Free-choice behavior labels used during training")
    require(training.get("base_model_config_sha256") == sha_file(Path(base_path) / "config.json"),
            "Adapter was trained against a different base-model configuration")
    require(Path(training["model"]).resolve() == Path(base_path).resolve(),
            "Adapter training weight path differs from generation base weights")
    checkpoint = adapter / "checkpoint_metadata.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
        require(saved["adapter_files"]["adapter_model.safetensors"] == fingerprint["adapter_weights_sha256"],
                "Checkpoint adapter weight hash mismatch")
        require(saved["adapter_files"]["adapter_config.json"] == fingerprint["adapter_config_sha256"],
                "Checkpoint adapter configuration hash mismatch")
        require(saved["training_config_sha256"] == sha_file(training_path), "Checkpoint training config hash mismatch")
    else:
        complete_path = adapter.parent / "training_complete.json"
        require(complete_path.is_file(), "Final adapter lacks training completion record")
        complete = json.loads(complete_path.read_text(encoding="utf-8-sig"))
        require(complete["adapter_files"]["adapter_model.safetensors"] == fingerprint["adapter_weights_sha256"],
                "Final adapter weight hash mismatch")
        require(complete["adapter_files"]["adapter_config.json"] == fingerprint["adapter_config_sha256"],
                "Final adapter configuration hash mismatch")
        require(complete["config_sha256"] == sha_file(training_path), "Final training configuration hash mismatch")
    return {"training_split_sha256": audit["split_content_sha256"],
            "training_input_sha256": audit["input_sha256"], "training_mode": training["mode"],
            "confirmation_status": "not_evaluated_on_new_confirmation_data"}


def inspect_saved(file, model, method, config_hash, rows, plan, adapter_hash=None):
    by_id = {row["prompt_id"]: row for row in rows}
    seen = set()
    if not Path(file).exists():
        return seen
    for record in read_rows(Path(file)):
        require(record.get("config_sha256") == config_hash and record.get("method") == method,
                "Saved output has mixed configuration or method")
        require(record.get("model") == model, "Saved output has another model")
        pid, replicate = record["prompt_id"], record["replicate"]
        require(pid in by_id and type(replicate) is int
                and 0 <= replicate < plan["repetitions"][by_id[pid]["task_type"]],
                "Saved trial not in frozen generation plan")
        trial_id = f"{model}|{method}|{pid}|{replicate}"
        require(record["trial_id"] == trial_id and trial_id not in seen, "Duplicate or malformed saved trial")
        require(record.get("status") == "ok", "Saved invalid output requires investigation; no silent reroll")
        require(record.get("sampling_seed") == trial_seed(model, method, pid, replicate), "Saved seed mismatch")
        require(record.get("messages_sha256") == digest(sanitized_messages(by_id[pid])),
                "Saved prompt was changed from original messages")
        require(record.get("fallback") is False, "LoRA evaluation must not use a prompt fallback")
        if adapter_hash is not None:
            require(record.get("adapter_fingerprint_sha256") == adapter_hash,
                    "Saved row references another adapter fingerprint")
            require(record.get("source_partition") == plan["source_partitions"][pid],
                    "Saved provenance partition mismatch")
        validate_output(record["response_text"], by_id[pid]["output_schema"], record.get("finish_reason"))
        seen.add(trial_id)
    return seen


def mark_complete(directory, model, method, config_hash, rows, plan, adapter_hash):
    file = Path(directory) / f"{method}.jsonl"
    seen = inspect_saved(file, model, method, config_hash, rows, plan, adapter_hash)
    require(len(seen) == plan["planned_rows"], "Incomplete output cannot be marked complete")
    records = read_rows(file)
    summary = {
        "model": model, "method": method, "status": "complete", "rows": len(records),
        "config_sha256": config_hash, "responses_sha256": sha_file(file),
        "adapter_fingerprint_sha256": adapter_hash, "plan_content_sha256": plan["content_sha256"],
        "decision_input_tokens": sum(r["input_tokens"] for r in records),
        "decision_completion_tokens": sum(r["completion_tokens"] for r in records),
        "fallback_rows": 0, "fallback_prompts": 0,
        "heldout_source": plan["heldout_source"],
        "include_all_heldout_source": plan["include_all_heldout_source"],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confirmation_status": "retrospective_development_not_new_confirmation",
    }
    atomic_json(Path(directory) / f"{method}.complete.json", summary)
    atomic_json(Path(directory) / f"{method}.lora_progress.json", summary)
    failure = Path(directory) / f"{method}.FAILED.json"
    if failure.exists():
        failure.replace(failure.with_name(f"{method}.FAILED.resolved_{time.time_ns()}.json"))
    return summary


def run(args):
    require(args.model == "qwen3_8b", "This initial adapter evaluation is restricted to qwen3_8b")
    rows, plan = select_plan(args.baseline, args.manifest,
                             args.include_all_heldout_source, args.heldout_source)
    model_paths, render_prompt = load_baseline_helpers(args.baseline)
    base_path = Path(model_paths[args.model]).resolve()
    fingerprint = adapter_fingerprint(args.adapter)
    training_identity = validate_training_base(args.adapter, base_path, fingerprint,
                                                args.method, args.heldout_source)
    require(training_identity["training_input_sha256"] == sha_file(args.baseline / "inputs.jsonl"),
            "Adapter training inputs differ from generation baseline")
    batch_size = args.batch_size
    require(batch_size > 0, "batch-size must be positive")
    directory = args.out / args.model
    directory.mkdir(parents=True, exist_ok=True)
    native = json.loads((base_path / "config.json").read_text(encoding="utf-8-sig"))
    require(native.get("model_type") == "qwen3", "Expected native Qwen3 architecture")
    context = min(8192, native.get("max_position_embeddings", 8192))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "peft")}
    config = {
        "protocol": PROTOCOL, "model": args.model, "method": args.method,
        "base_model_fingerprint": checkpoint_fingerprint(base_path), "adapter_fingerprint": fingerprint,
        "training_identity": training_identity, "versions": versions,
        "baseline_inputs_sha256": sha_file(args.baseline / "inputs.jsonl"),
        "manifest_sha256": sha_file(args.manifest), "plan_content_sha256": plan["content_sha256"],
        "runner_sha256": sha_file(Path(__file__)),
        "shared_prompt_helpers_sha256": sha_file(Path(__file__).with_name("run_prompt_methods.py")),
        "baseline_renderer_sha256": sha_file(args.baseline / "INVAR_dataset/run_vllm_batch.py"),
        "temperature": 0.7, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0,
        "max_tokens": 128, "base_seed": BASE_SEED, "batch_size": batch_size,
        "repetitions": plan["repetitions"], "context": context,
        "decoder": "json_schema_xgrammar_compact", "enable_lora": True, "max_lora_rank": 8,
        "max_loras": 1, "gpu_memory_utilization": 0.90,
        "seed_policy": "sha256(base_seed,model,method,prompt_id,replicate)",
        "prompt_policy": "unchanged sanitized original baseline messages; original public schema only",
        "correctness_labels_given_to_model_or_decoder": False,
        "baseline_interface": "native Qwen3 template; enable_thinking=False; vLLM generation_config=vllm",
    }
    config_hash = digest(config)
    identity_path = directory / f"{args.method}.run_identity.json"
    if identity_path.exists():
        prior = json.loads(identity_path.read_text(encoding="utf-8-sig"))
        require(prior["config_sha256"] == config_hash, "Configuration/adapter changed; use new output directory")
    identity = {"config_sha256": config_hash, "config": config, "plan": plan}
    atomic_json(identity_path, identity)
    file = directory / f"{args.method}.jsonl"
    done = inspect_saved(file, args.model, args.method, config_hash, rows, plan, fingerprint["content_sha256"])
    if len(done) == plan["planned_rows"]:
        print(json_text(mark_complete(directory, args.model, args.method, config_hash,
                                      rows, plan, fingerprint["content_sha256"])), flush=True)
        return

    # GPU/model imports occur only after hashes and every saved output are validated.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True, local_files_only=True)
    prepared = {}
    for row in rows:
        messages = sanitized_messages(row)
        text = render_prompt(tokenizer, messages, args.model)
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        require(tokens + 128 <= context, f"Context overflow: {row['prompt_id']}; truncation forbidden")
        prepared[row["prompt_id"]] = {"text": text, "input_tokens": tokens,
                                       "messages_sha256": digest(messages), "rendered_sha256": digest(text)}
    prompt_hashes = {pid: {key: value for key, value in record.items() if key != "text"}
                     for pid, record in prepared.items()}
    prompt_path = directory / f"{args.method}.prompts.json"
    if prompt_path.exists():
        require(json.loads(prompt_path.read_text(encoding="utf-8-sig")) == prompt_hashes,
                "Rendered prompt hashes changed")
    atomic_json(prompt_path, prompt_hashes)
    llm = LLM(
        model=str(base_path), tokenizer=str(base_path), trust_remote_code=True, dtype="bfloat16",
        max_model_len=context, gpu_memory_utilization=0.90, max_num_seqs=batch_size,
        enable_prefix_caching=True, generation_config="vllm", seed=BASE_SEED,
        structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
        enable_lora=True, max_lora_rank=8, max_loras=1,
    )
    request = LoRARequest("invar_" + args.method + "_" + fingerprint["adapter_weights_sha256"][:12],
                          1, str(Path(args.adapter).resolve()))
    grammars = {r["prompt_id"]: StructuredOutputsParams(json=r["output_schema"], disable_any_whitespace=True)
                for r in rows}
    pending = [(row, replicate, f"{args.model}|{args.method}|{row['prompt_id']}|{replicate}")
               for row in rows for replicate in range(plan["repetitions"][row["task_type"]])
               if f"{args.model}|{args.method}|{row['prompt_id']}|{replicate}" not in done]
    started = time.time()
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        seeds = [trial_seed(args.model, args.method, row["prompt_id"], replicate)
                 for row, replicate, _ in batch]
        params = [SamplingParams(temperature=0.7, top_p=0.95, top_k=-1, repetition_penalty=1.0,
                                 max_tokens=128, seed=seed, structured_outputs=grammars[row["prompt_id"]])
                  for (row, _, _), seed in zip(batch, seeds)]
        began = time.time()
        generated = llm.generate([prepared[row["prompt_id"]]["text"] for row, _, _ in batch],
                                 params, use_tqdm=False, lora_request=request)
        require(len(generated) == len(batch), "Adapter generation count mismatch")
        seconds = time.time() - began
        records, invalid = [], []
        for (row, replicate, trial_id), result, seed in zip(batch, generated, seeds):
            require(len(result.outputs) == 1, "Missing/ambiguous adapter completion")
            binding = getattr(result, "lora_request", None)
            require(binding is not None and binding.lora_int_id == request.lora_int_id
                    and binding.lora_name == request.lora_name,
                    "vLLM result does not confirm the requested adapter binding")
            completion = result.outputs[0]
            data = prepared[row["prompt_id"]]
            record = {
                "model": args.model, "method": args.method, "prompt_id": row["prompt_id"],
                "task_type": row["task_type"], "replicate": replicate, "batch": "main", "trial_id": trial_id,
                "response_text": completion.text, "status": "ok", "finish_reason": completion.finish_reason,
                "input_tokens": data["input_tokens"], "completion_tokens": len(completion.token_ids),
                "sampling_seed": seed, "config_sha256": config_hash,
                "messages_sha256": data["messages_sha256"], "rendered_sha256": data["rendered_sha256"],
                "ledger_cache_key": None, "fallback": False, "fallback_reason": None,
                "batch_seconds": seconds, "batch_size_actual": len(batch),
                "adapter_fingerprint_sha256": fingerprint["content_sha256"],
                "lora_request_name": binding.lora_name, "lora_request_id": binding.lora_int_id,
                "source_partition": plan["source_partitions"][row["prompt_id"]],
            }
            try:
                validate_output(completion.text, row["output_schema"], completion.finish_reason)
            except (ValueError, TypeError, KeyError) as error:
                record.update(status="invalid_output", validation_error=str(error))
                invalid.append(trial_id)
            records.append(record)
        append_rows(file, records)
        require(not invalid, f"Invalid outputs retained; no automatic reroll: {invalid[:3]}")
        progress = {"model": args.model, "method": args.method, "status": "running",
                    "done": len(done) + start + len(batch), "total": plan["planned_rows"],
                    "config_sha256": config_hash, "seconds": time.time() - started,
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(directory / f"{args.method}.lora_progress.json", progress)
        if start // batch_size % 20 == 0:
            print(json_text(progress), flush=True)
    print(json_text(mark_complete(directory, args.model, args.method, config_hash,
                                  rows, plan, fingerprint["content_sha256"])), flush=True)


def self_test():
    """CPU-only boundary/metadata/resume/adapter-hash tests using temporary fixtures."""
    clean = [{"role": "system", "content": "只输出JSON。"}, {"role": "user", "content": "选择A或B。"}]
    schema = {"type": "object", "additionalProperties": False, "required": ["answer"],
              "properties": {"answer": {"type": "string", "enum": ["A", "B"]}}}
    row = {"prompt_id": "p", "task_type": "changed_fact", "messages": copy.deepcopy(clean),
           "output_schema": schema, "expected_answer": "SECRET_TRUTH", "expected_actions": ["SECRET_ACTION"]}
    require(sanitized_messages(row) == clean and "SECRET" not in json_text(sanitized_messages(row)),
            "Private labels leaked into inference")
    require(row["messages"] == clean, "Sanitization mutated original messages")
    plan = {"repetitions": REPETITIONS, "planned_rows": 8, "heldout_source": "HPVBG",
            "include_all_heldout_source": False, "content_sha256": "planhash",
            "source_partitions": {"p": "heldout_source"}}
    with tempfile.TemporaryDirectory(prefix="invar_lora_eval_test_") as temp:
        root = Path(temp)
        adapter = root / "adapter"
        adapter.mkdir()
        adapter_config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8,
                          "target_modules": ["q_proj", "v_proj"], "bias": "none"}
        (adapter / "adapter_config.json").write_text(json_text(adapter_config), encoding="utf-8")
        weights = adapter / "adapter_model.safetensors"
        weights.write_bytes(b"hash-fixture-not-real-model")
        before = adapter_fingerprint(adapter)
        weights.write_bytes(b"different-weight-bytes")
        require(before["content_sha256"] != adapter_fingerprint(adapter)["content_sha256"],
                "Changed adapter weights not detected")
        adapter_config["target_modules"] = ["q_proj", "lm_head"]
        (adapter / "adapter_config.json").write_text(json_text(adapter_config), encoding="utf-8")
        try:
            adapter_fingerprint(adapter)
        except ValueError:
            pass
        else:
            raise AssertionError("Unexpected adapter target accepted")
        file = root / "sft.jsonl"
        records = [{"model": "qwen3_8b", "method": "sft", "prompt_id": "p", "replicate": i,
                    "trial_id": f"qwen3_8b|sft|p|{i}", "response_text": '{"answer":"A"}',
                    "status": "ok", "finish_reason": "stop", "config_sha256": "cfg",
                    "sampling_seed": trial_seed("qwen3_8b", "sft", "p", i),
                    "messages_sha256": digest(clean), "fallback": False,
                    "adapter_fingerprint_sha256": "adapterhash", "source_partition": "heldout_source",
                    "input_tokens": 12, "completion_tokens": 5} for i in range(8)]
        append_rows(file, records)
        require(len(inspect_saved(file, "qwen3_8b", "sft", "cfg", [row], plan)) == 8, "Valid resume rejected")
        summary = mark_complete(root, "qwen3_8b", "sft", "cfg", [row], plan, "adapterhash")
        require(summary["rows"] == 8 and summary["decision_input_tokens"] == 96, "Completion counts wrong")
        require(not (root / "COMPLETE.json").exists() and not (root / "identity.json").exists(),
                "Prompt-runner identity or complete file overwritten")
        for mutation in ({"config_sha256": "other"}, {"status": "invalid_output"},
                         {"sampling_seed": -1}, {"messages_sha256": "wrong"},
                         {"replicate": 8}, {"fallback": True}):
            bad = [dict(records[0], **mutation)]
            file.write_text(json_text(bad[0]) + "\n", encoding="utf-8")
            try:
                inspect_saved(file, "qwen3_8b", "sft", "cfg", [row], plan)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid saved trial accepted: {mutation}")
        file.write_text(json_text(records[0]) + "\n" + json_text(records[0]) + "\n", encoding="utf-8")
        try:
            inspect_saved(file, "qwen3_8b", "sft", "cfg", [row], plan)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate trial accepted")
    require(not any(name in sys.modules for name in ("torch", "transformers", "vllm")),
            "CPU self-test imported GPU/model dependencies")
    print(json_text({"self_test": "passed", "checks": ["unchanged_messages", "private_metadata_excluded",
                      "adapter_weight_hash", "rank_target_rejection", "resume_config_seed_prompt_validation",
                      "invalid_output_not_rerolled", "duplicate_rejection", "compatible_completion_manifest",
                      "separate_run_identity", "no_model_imports"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", choices=("qwen3_8b",))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--include-all-heldout-source", action="store_true")
    parser.add_argument("--heldout-source", default="HPVBG")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("baseline", "manifest", "model", "adapter", "method", "out")):
        parser.error("--baseline --manifest --model --adapter --method --out are required unless --self-test")
    try:
        run(args)
    except Exception as error:
        directory = args.out / args.model
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / f"{args.method}.FAILED.json", {
            "model": args.model, "method": args.method, "error": f"{type(error).__name__}: {error}",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        raise


if __name__ == "__main__":
    main()
