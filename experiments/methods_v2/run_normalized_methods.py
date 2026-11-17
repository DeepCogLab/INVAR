"""Independent V2 conservative-normalization screen, seed 20261026.

Uses frozen V1 helper functions by an explicit path; no helper globals are
modified. Only current messages and public legal JSON menus reach the model.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1 = Path(os.environ.get("INVAR_METHODS_V1", HERE.parent / "methods_v1")).resolve()
sys.path.insert(0, str(V1))
from run_prompt_methods import (append_rows, atomic_json, checkpoint_fingerprint,
                               digest, json_text, load_baseline_helpers, read_rows,
                               sanitized_messages, sha_bytes, validate_output)
from conservative_normalizer import VERSION as NORMALIZER_VERSION
from conservative_normalizer import audit_corpus, normalize_messages
from conservative_normalizer import self_test as normalizer_self_test


VERSION = "INVAR_conservative_normalization_screen_v2_20261026"
BASE_SEED = 20261026
METHODS = ("baseline", "canonical_modules", "canonical_profile")
MODELS = ("qwen3_8b", "qwen25_14b_instruct", "glm4_9b")
REPETITIONS = {"behavior": 32, "changed_fact": 8, "stable_fact": 8, "goal": 16, "role": 16}


def source_hashes(baseline):
    paths = {"driver": Path(__file__), "normalizer": V1 / "conservative_normalizer.py",
             "v1_helpers": V1 / "run_prompt_methods.py",
             "baseline_renderer": baseline / "INVAR_dataset/run_vllm_batch.py"}
    return {name: {"path": str(path.resolve()), "sha256": sha_bytes(path.read_bytes())}
            for name, path in paths.items()}


def seed_for(model, method, prompt_id, replicate):
    value = f"{BASE_SEED}|{VERSION}|{model}|{method}|{prompt_id}|{replicate}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def transform(row, method):
    """Explicit current-message boundary; no full input row is normalized."""
    original = sanitized_messages(row)
    if method == "baseline":
        return {"messages": original, "transformation_audit": {
            "version": NORMALIZER_VERSION, "status": "no_change", "baseline_identity": True,
            "input_messages_sha256": digest(original), "output_messages_sha256": digest(original),
            "private_metadata_read": False, "retention_verified": True,
            "question_preserved_exactly": True, "reordered": False,
            "persona_table_converted": False, "fallback_reason": None,
        }}
    if method not in METHODS:
        raise ValueError("Unknown normalization method")
    result = normalize_messages(original, convert_persona_table=method == "canonical_profile")
    audit = result["transformation_audit"]
    if not audit["retention_verified"] or not audit["question_preserved_exactly"]:
        raise ValueError("Normalizer did not verify information retention")
    if audit["status"] == "fallback" and result["messages"] != original:
        raise ValueError("Fallback changed the original messages")
    return result


def select_inputs(baseline, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    inputs_file = baseline / "inputs.jsonl"
    if sha_bytes(inputs_file.read_bytes()) != manifest["baseline_inputs_sha256"]:
        raise ValueError("Baseline input hash differs from frozen manifest")
    rows = read_rows(inputs_file)
    by_id = {row["prompt_id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("Duplicate input prompt IDs")
    selected = manifest["selected_prompt_ids"]
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= by_id.keys():
        raise ValueError("Invalid selected prompt list")
    if manifest.get("repetitions", REPETITIONS) != REPETITIONS:
        raise ValueError("Repetition policy differs from the matched first screen")
    chosen = [by_id[pid] for pid in selected]
    for row in chosen:
        if row["task_type"] not in REPETITIONS or not isinstance(row.get("output_schema"), dict):
            raise ValueError("Invalid task type or missing public schema")
        sanitized_messages(row)
    calls = sum(REPETITIONS[row["task_type"]] for row in chosen)
    if "calls_per_model_method" in manifest and calls != manifest["calls_per_model_method"]:
        raise ValueError("Frozen call count mismatch")
    return chosen, manifest, calls


def write_normalizer_audit(directory, rows, methods):
    """Persist exact original/transformed text separately from model responses."""
    original_records = [{"prompt_id": row["prompt_id"], "task_type": row["task_type"],
                         "messages": sanitized_messages(row),
                         "messages_sha256": digest(sanitized_messages(row)),
                         "public_output_schema": row["output_schema"]} for row in rows]
    (directory / "normalizer_input.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in original_records), encoding="utf-8")
    prepared, summaries = {}, {}
    for method in methods:
        records, mapping, counts, reasons = [], {}, collections.Counter(), collections.Counter()
        for row in rows:
            result = transform(row, method)
            # Independently poison metadata to check the actual driver boundary.
            poisoned = copy.deepcopy(row)
            poisoned.update(prompt_id="NORMALIZER_SECRET_PID_26", case_id="NORMALIZER_SECRET_CASE_26",
                            pair_id="NORMALIZER_SECRET_PAIR_26", variant_id="NORMALIZER_SECRET_VARIANT_26",
                            expected_answer="NORMALIZER_SECRET_TRUTH_26", expected_actions=["NORMALIZER_SECRET_ACTION_26"])
            if transform(poisoned, method) != result:
                raise ValueError("Private metadata influenced preprocessing")
            if "NORMALIZER_SECRET_" in json.dumps(result["messages"], ensure_ascii=False):
                raise ValueError("Private metadata leaked into messages")
            audit = result["transformation_audit"]
            counts[audit["status"]] += 1
            counts["reordered"] += bool(audit["reordered"])
            counts["persona_table_converted"] += bool(audit["persona_table_converted"])
            if audit["fallback_reason"]:
                reasons[audit["fallback_reason"]] += 1
            pid = row["prompt_id"]
            mapping[pid] = result
            records.append({"prompt_id": pid, "task_type": row["task_type"],
                            "method": method, "messages": result["messages"],
                            "metadata_sentinel_check": "passed", "transformation_audit": audit})
        file = directory / f"{method}.normalizer_changes.jsonl"
        file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        prepared[method] = mapping
        summaries[method] = {"prompts": len(rows), **dict(counts), "fallback_reasons": dict(reasons),
                             "audit_file_sha256": sha_bytes(file.read_bytes())}
    atomic_json(directory / "normalizer_summary.json", {
        "normalizer_version": NORMALIZER_VERSION, "base_seed": BASE_SEED,
        "input_audit_sha256": sha_bytes((directory / "normalizer_input.jsonl").read_bytes()),
        "methods": summaries, "all_metadata_sentinel_checks": "passed",
        "fallback_policy": "original current messages retained; no sample discarded or semantic reroll"})
    return prepared


def inspect_saved(file, model, method, config_hash, by_id, transformed):
    done = set()
    if not file.exists():
        return done
    for row in read_rows(file):
        pid, replicate = row["prompt_id"], row["replicate"]
        if row.get("config_sha256") != config_hash or row.get("model") != model or row.get("method") != method:
            raise ValueError("Different protocol/model/method in saved responses")
        if pid not in by_id or type(replicate) is not int or not 0 <= replicate < REPETITIONS[by_id[pid]["task_type"]]:
            raise ValueError("Saved response outside frozen plan")
        trial_id = f"{model}|{method}|{pid}|{replicate}"
        if row.get("trial_id") != trial_id or trial_id in done:
            raise ValueError("Duplicate or malformed saved trial")
        if row.get("sampling_seed") != seed_for(model, method, pid, replicate):
            raise ValueError("Saved response uses a different seed policy")
        if row.get("messages_sha256") != digest(transformed[pid]["messages"]):
            raise ValueError("Saved response uses different normalized messages")
        if row.get("status") != "ok":
            raise ValueError("Saved invalid output requires investigation; never silently reroll")
        validate_output(row["response_text"], by_id[pid]["output_schema"], row.get("finish_reason"))
        done.add(trial_id)
    return done


def mark_complete(directory, model, method, config_hash, by_id, transformed, expected):
    file = directory / f"{method}.jsonl"
    done = inspect_saved(file, model, method, config_hash, by_id, transformed)
    if len(done) != expected:
        raise ValueError("Incomplete method cannot be marked complete")
    rows = read_rows(file)
    summary = {"protocol": VERSION, "model": model, "method": method, "status": "complete",
               "rows": len(rows), "config_sha256": config_hash,
               "responses_sha256": sha_bytes(file.read_bytes()), "base_seed": BASE_SEED,
               "decision_input_tokens": sum(row["input_tokens"] for row in rows),
               "decision_completion_tokens": sum(row["completion_tokens"] for row in rows),
               "fallback_rows": sum(row["fallback"] for row in rows),
               "fallback_prompts": sum(result["transformation_audit"]["status"] == "fallback" for result in transformed.values()),
               "normalizer_changes_sha256": sha_bytes((directory / f"{method}.normalizer_changes.jsonl").read_bytes()),
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(directory / f"{method}.complete.json", summary)
    return summary


def run(args):
    if args.model not in MODELS:
        raise ValueError(f"This screen is restricted to {MODELS}")
    methods = list(dict.fromkeys(args.methods.split(",")))
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"Allowed methods: {METHODS}")
    rows, manifest, expected = select_inputs(args.baseline, args.manifest)
    by_id = {row["prompt_id"]: row for row in rows}
    paths, render_prompt = load_baseline_helpers(args.baseline)
    path = Path(paths[args.model])
    batch_size = args.batch_size or (96 if args.model == "qwen25_14b_instruct" else 192)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    native = json.loads((path / "config.json").read_text(encoding="utf-8"))
    context = min(8192, native.get("max_position_embeddings", 8192))
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")}
    config = {"protocol": VERSION, "model": args.model, "base_seed": BASE_SEED,
              "resolved_v1_helper_directory": str(V1),
              "code_hashes": source_hashes(args.baseline), "model_fingerprint": checkpoint_fingerprint(path),
              "baseline_inputs_sha256": sha_bytes((args.baseline / "inputs.jsonl").read_bytes()),
              "manifest_sha256": sha_bytes(args.manifest.read_bytes()), "versions": versions,
              "temperature": .7, "top_p": .95, "top_k": -1, "repetition_penalty": 1.,
              "max_tokens": 128, "batch_size": batch_size, "context": context,
              "repetitions": REPETITIONS, "normalizer_version": NORMALIZER_VERSION,
              "normalizer_policy": "current-message conservative module reorder; optional strict persona-table conversion",
              "seed_policy": "sha256(20261026,protocol,model,method,prompt_id,replicate)",
              "decoder": "json_schema_xgrammar_compact", "dtype": "bfloat16",
              "generation_config": "vllm", "prefix_caching": True, "gpu_memory_utilization": .90,
              "correctness_metadata_given_to_model_or_decoder": False,
              "baseline_policy": "exact original messages, new matched budget and fresh seeds; never reuse V1 outputs"}
    config_hash = digest(config)
    directory = args.out / args.model
    directory.mkdir(parents=True, exist_ok=True)
    config_file = directory / "config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8-sig"))["config_sha256"] != config_hash:
        raise ValueError("Protocol/configuration changed; use a separate output directory")
    atomic_json(config_file, {"config": config, "config_sha256": config_hash})
    transformed = write_normalizer_audit(directory, rows, methods)
    done_by_method = {method: inspect_saved(directory / f"{method}.jsonl", args.model, method,
                                           config_hash, by_id, transformed[method]) for method in methods}
    identity = {"config_sha256": config_hash, "config": config,
                "selected_inputs": len(rows), "planned_rows_per_method": expected,
                "methods_requested": methods, "confirmation_status": "retrospective_development_screen"}
    identity_path = directory / "identity.json"
    if identity_path.exists():
        prior = json.loads(identity_path.read_text(encoding="utf-8-sig"))
        if prior.get("config_sha256") != config_hash:
            raise ValueError("Saved identity differs from current protocol")
        identity = {**prior, **identity}
    atomic_json(directory / "identity.json", identity)
    if not all(len(done) == expected for done in done_by_method.values()):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        texts, lengths = {}, {}
        for method in methods:
            texts[method], lengths[method] = {}, {}
            for pid, result in transformed[method].items():
                text = render_prompt(tokenizer, result["messages"], args.model)
                length = len(tokenizer.encode(text, add_special_tokens=False))
                if length + 128 > context:
                    raise ValueError(f"Context overflow {method}/{pid}; input truncation forbidden")
                texts[method][pid], lengths[method][pid] = text, length
        identity["max_input_tokens_by_method"] = {method: max(lengths[method].values()) for method in methods}
        identity["rendered_prompts_sha256_by_method"] = {method: digest(texts[method]) for method in methods}
        atomic_json(directory / "identity.json", identity)
        llm = LLM(model=str(path), tokenizer=str(path), trust_remote_code=True, dtype="bfloat16",
                  max_model_len=context, gpu_memory_utilization=.90, max_num_seqs=batch_size,
                  enable_prefix_caching=True, generation_config="vllm", seed=BASE_SEED,
                  structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True})
        grammars = {pid: StructuredOutputsParams(json=row["output_schema"], disable_any_whitespace=True)
                    for pid, row in by_id.items()}
        began_all = time.time()
        for method in methods:
            done = done_by_method[method]
            todo = [(row, rep, f"{args.model}|{method}|{row['prompt_id']}|{rep}")
                    for row in rows for rep in range(REPETITIONS[row["task_type"]])
                    if f"{args.model}|{method}|{row['prompt_id']}|{rep}" not in done]
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                seeds = [seed_for(args.model, method, row["prompt_id"], rep) for row, rep, _ in batch]
                parameters = [SamplingParams(temperature=.7, top_p=.95, top_k=-1, repetition_penalty=1.,
                                             max_tokens=128, seed=seed, structured_outputs=grammars[row["prompt_id"]])
                              for (row, _, _), seed in zip(batch, seeds)]
                began = time.time()
                generated = llm.generate([texts[method][row["prompt_id"]] for row, _, _ in batch], parameters, use_tqdm=False)
                if len(generated) != len(batch):
                    raise RuntimeError("Generated response count mismatch")
                seconds = time.time() - began
                records, invalid = [], []
                for (row, rep, trial_id), result, seed in zip(batch, generated, seeds):
                    if len(result.outputs) != 1:
                        raise RuntimeError("Missing or ambiguous completion")
                    completion = result.outputs[0]
                    pid = row["prompt_id"]
                    norm = transformed[method][pid]
                    audit = norm["transformation_audit"]
                    record = {"model": args.model, "method": method, "prompt_id": pid,
                              "task_type": row["task_type"], "replicate": rep, "batch": "main", "trial_id": trial_id,
                              "response_text": completion.text, "status": "ok", "finish_reason": completion.finish_reason,
                              "input_tokens": lengths[method][pid], "completion_tokens": len(completion.token_ids),
                              "sampling_seed": seed, "base_seed": BASE_SEED, "config_sha256": config_hash,
                              "messages_sha256": digest(norm["messages"]), "rendered_sha256": digest(texts[method][pid]),
                              "ledger_cache_key": None, "fallback": audit["status"] == "fallback",
                              "fallback_reason": audit["fallback_reason"], "normalizer_status": audit["status"],
                              "normalizer_changed": audit["input_messages_sha256"] != audit["output_messages_sha256"],
                              "batch_seconds": seconds, "batch_size_actual": len(batch)}
                    try:
                        validate_output(completion.text, row["output_schema"], completion.finish_reason)
                    except (ValueError, KeyError, TypeError) as error:
                        record.update(status="invalid_output", validation_error=str(error))
                        invalid.append(trial_id)
                    records.append(record)
                append_rows(directory / f"{method}.jsonl", records)
                if invalid:
                    raise RuntimeError(f"Invalid output retained, no silent retry: {invalid[:3]}")
                progress = {"protocol": VERSION, "model": args.model, "method": method, "status": "running",
                            "done": len(done) + start + len(batch), "total": expected,
                            "seconds": round(time.time() - began_all, 2), "config_sha256": config_hash,
                            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                atomic_json(directory / "progress.json", progress)
                if start // batch_size % 20 == 0:
                    print(json_text(progress), flush=True)
            mark_complete(directory, args.model, method, config_hash, by_id, transformed[method], expected)
    completed = {method: mark_complete(directory, args.model, method, config_hash, by_id, transformed[method], expected)
                 for method in methods}
    atomic_json(directory / "COMPLETE.json", {"protocol": VERSION, "model": args.model, "status": "complete",
                "base_seed": BASE_SEED, "config_sha256": config_hash, "methods": completed})
    failure = directory / "FAILED.json"
    if failure.exists():
        failure.replace(directory / f"FAILED.resolved_{time.time_ns()}.json")
    print(json_text({"status": "complete", "model": args.model, "methods": methods,
                     "rows_per_method": expected, "base_seed": BASE_SEED}), flush=True)


def self_test():
    normalizer_self_test()
    role = "【个体设定】\n你是P2。\n\n"
    rules = "【实验规则】\nP1自主选择，不能由你代替决定。\n\n"
    situation = "【当前情况】\nP1尚未选择。\n\n"
    question = "【独立事实检查】\nA：P1自主选择。B：计算机决定。"
    clean = [{"role": "system", "content": "原系统指令。"},
             {"role": "user", "content": situation + rules + role + question}]
    row = {"messages": copy.deepcopy(clean), "expected_answer": "SECRET_TRUTH", "pair_id": "SECRET_PAIR"}
    for method in METHODS:
        result = transform(row, method)
        poisoned = dict(row, expected_answer="DIFFERENT_TRUTH", variant_id="SECRET_VARIANT")
        if transform(poisoned, method) != result or "SECRET_" in json.dumps(result["messages"]):
            raise AssertionError("Metadata affected driver normalization")
        if result["messages"][0] != clean[0] or not result["messages"][1]["content"].endswith(question):
            raise AssertionError("System or question changed")
    if transform(row, "baseline")["messages"] != clean or row["messages"] != clean:
        raise AssertionError("Baseline/input identity failed")
    strange = {"messages": [{"role": "user", "content": "【未知模块】\n不能删除此信息。\n" + clean[1]["content"]}]}
    for method in METHODS[1:]:
        result = transform(strange, method)
        if result["transformation_audit"]["status"] != "fallback" or result["messages"] != strange["messages"]:
            raise AssertionError("Unknown module fallback failed")
    if BASE_SEED != 20261026 or seed_for("model", "baseline", "pid", 0) == seed_for("model", "canonical_modules", "pid", 0):
        raise AssertionError("Fresh protocol seeds failed")
    if seed_for("model", "baseline", "pid", 0) != seed_for("model", "baseline", "pid", 0):
        raise AssertionError("Non-reproducible seed")
    print(json_text({"driver_self_test": "passed", "protocol": VERSION,
                     "checks": ["baseline_exact", "metadata_boundary", "question_and_system_unchanged",
                                "input_not_mutated", "unknown_module_fallback", "independent_seed20261026"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit-corpus", action="store_true", help="CPU-only all-input retention and metadata audit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.audit_corpus:
        if not all((args.baseline, args.manifest, args.out)):
            parser.error("--audit-corpus requires --baseline --manifest --out")
        rows, manifest, calls = select_inputs(args.baseline, args.manifest)
        audit_corpus(args.baseline, args.out)
        atomic_json(args.out / "driver_audit_identity.json", {
            "protocol": VERSION, "base_seed": BASE_SEED, "code_hashes": source_hashes(args.baseline),
            "resolved_v1_helper_directory": str(V1),
            "manifest_sha256": sha_bytes(args.manifest.read_bytes()), "selected_inputs": len(rows),
            "selected_calls_per_model_method": calls, "gpu_initialized": False})
        return
    if not all((args.baseline, args.manifest, args.model, args.out)):
        parser.error("--baseline --manifest --model --out required unless using a CPU mode")
    try:
        run(args)
    except Exception as error:
        directory = args.out / args.model
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "FAILED.json", {"protocol": VERSION, "model": args.model,
                    "error": f"{type(error).__name__}: {error}",
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise


if __name__ == "__main__":
    main()
