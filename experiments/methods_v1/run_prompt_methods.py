"""Frozen-input INVAR prompt/ledger pilots; --self-test needs no GPU packages.

Only allowlisted message text and the public output schema reach vLLM.
All correctness metadata remains outside model prompts and decoding constraints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


METHODS = ("baseline", "generic", "social_audit", "ledger_original", "ledger_only")
QUESTION_MARKERS = ("【独立事实检查】", "【事实理解检查】", "【本次决策】")
REPETITIONS = {"behavior": 32, "changed_fact": 8, "stable_fact": 8, "goal": 16, "role": 16}
BASE_SEED = 20261025
GENERIC = (
    "作答前请认真阅读全部材料，核对题目实际要求，不要遗漏信息或加入没有依据的假设。"
    "仔细检查自己的判断是否与原文一致，最后仅输出原题要求的JSON对象。"
)
SOCIAL_AUDIT = (
    "作答前请核对：你对应的参与者是谁；哪些行动由谁决定、哪些已固定；"
    "每个合法行动对各人的点数后果；当前事实与未知信息；本题是否明确指定目标。"
    "匿名编号只用于指代，不据编号本身推断角色。"
    "自由选择时不额外假设最大化本人、总点数或公平；"
    "明示目标时按该目标比较全部合法行动。最后仅输出原题要求的JSON对象。"
)
LEDGER_SYSTEM = (
    "你是材料整理器。只整理下方当前社会情境材料，不作行为选择，不回答任何问题。"
    "输出JSON对象，只有claims和unknowns两个字段。"
    "claims是数组，每项只有fact和evidence两个字符串字段。"
    "fact用简短句子整理一个材料明确支持的事实；evidence必须是材料中的连续原文片段，逐字引用。"
    "覆盖参与者编号与角色、谁是当前决策者、权限与行动决定来源、已发生或尚未发生的行动、"
    "全部行动分支及收益规则、当前情况、信息限制，以及材料明确写出的本次目标。"
    "个人与群体、自主选择与计算机随机或强制决定的区别必须保留。"
    "规则中出现的条件分支要保留条件，不能当成已经发生的事实。"
    "unknowns是字符串数组，只记录材料明确未提供或无法确定的事项；没有则为空数组。"
    "没有明示目标时不得编造自利、公平或总收益最大化目标。不得猜测他人意图或未来行动。"
    "不得引入原文以外的事实、答案、建议或背景常识。"
)
LEDGER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["claims", "unknowns"],
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["fact", "evidence"],
            "properties": {"fact": {"type": "string"}, "evidence": {"type": "string"}},
        }},
        "unknowns": {"type": "array", "items": {"type": "string"}},
    },
}


def json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(obj: Any) -> str:
    return sha_bytes(json_text(obj).encode("utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path: Path, obj: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sanitized_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Missing messages")
    clean = []
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Unknown message role")
        if not isinstance(message.get("content"), str):
            raise ValueError("Non-string message content")
        clean.append({"role": message["role"], "content": message["content"]})
    return clean


def split_material_question(messages: list[dict[str, str]]) -> tuple[str, str, int]:
    """Earliest question marker; the explicit-goal section remains material."""
    user_positions = [i for i, message in enumerate(messages) if message["role"] == "user"]
    if len(user_positions) != 1:
        raise ValueError("Expected exactly one user message")
    user_index = user_positions[0]
    text = messages[user_index]["content"]
    positions = [text.find(marker) for marker in QUESTION_MARKERS if marker in text]
    if not positions:
        raise ValueError("Question boundary marker missing")
    boundary = min(positions)
    material, question = text[:boundary], text[boundary:]
    if not material.strip():
        raise ValueError("Empty material")
    return material, question, user_index


def validate_schema(obj: Any, schema: dict[str, Any], where: str = "$") -> None:
    """Validate the types used by baseline schemas without optional dependencies."""
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool}
    if kind in types:
        expected = types[kind]
        if isinstance(expected, tuple):
            if type(obj) not in expected:
                raise ValueError(f"{where}: expected {kind}")
        elif type(obj) is not expected:
            raise ValueError(f"{where}: expected {kind}")
    else:
        raise ValueError(f"{where}: unsupported schema type {kind!r}")
    if "enum" in schema and obj not in schema["enum"]:
        raise ValueError(f"{where}: value outside legal menu")
    if "minimum" in schema and obj < schema["minimum"]:
        raise ValueError(f"{where}: below minimum")
    if "maximum" in schema and obj > schema["maximum"]:
        raise ValueError(f"{where}: above maximum")
    if kind == "object":
        properties = schema.get("properties", {})
        if not set(schema.get("required", [])).issubset(obj):
            raise ValueError(f"{where}: missing required property")
        if schema.get("additionalProperties") is False and not set(obj).issubset(properties):
            raise ValueError(f"{where}: additional property")
        for key, value in obj.items():
            if key in properties:
                validate_schema(value, properties[key], f"{where}.{key}")
    elif kind == "array":
        if "minItems" in schema and len(obj) < schema["minItems"]:
            raise ValueError(f"{where}: not enough items")
        for index, value in enumerate(obj):
            validate_schema(value, schema["items"], f"{where}[{index}]")


def validate_output(response: str, schema: dict[str, Any], finish_reason: str | None) -> None:
    if finish_reason == "length":
        raise ValueError("Output token budget exhausted")
    validate_schema(json.loads(response), schema)


def check_ledger(response: str, material: str, finish_reason: str | None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        validate_output(response, LEDGER_SCHEMA, finish_reason)
        obj = json.loads(response)
        if not obj["claims"]:
            raise ValueError("No claims extracted")
        for claim in obj["claims"]:
            if not claim["fact"].strip() or not claim["evidence"].strip():
                raise ValueError("Empty claim or evidence")
            if claim["evidence"] not in material:
                raise ValueError("Evidence is not a verbatim material substring")
        return obj, None
    except (ValueError, TypeError, KeyError) as error:
        return None, f"{type(error).__name__}: {error}"


def add_system(messages: list[dict[str, str]], addition: str) -> list[dict[str, str]]:
    messages = copy.deepcopy(messages)
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + addition
    else:
        messages.insert(0, {"role": "system", "content": addition})
    return messages


def method_messages(row: dict[str, Any], method: str,
                    ledger: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """The sole decision-prompt builder; never serializes the full input row."""
    messages = sanitized_messages(row)
    if method == "baseline" or (method.startswith("ledger_") and ledger is None):
        return messages
    if method == "generic":
        return add_system(messages, GENERIC)
    if method == "social_audit":
        return add_system(messages, SOCIAL_AUDIT)
    if method not in {"ledger_original", "ledger_only"}:
        raise ValueError(f"Unknown method {method}")
    material, question, index = split_material_question(messages)
    ledger_text = "【当前材料整理账本】\n" + json_text(ledger)
    if method == "ledger_original":
        # Preserve material and question byte-for-byte; insert only between them.
        messages[index]["content"] = material + ledger_text + "\n\n" + question
    else:
        messages[index]["content"] = ledger_text + "\n\n" + question
    return messages


def extractor_messages(material: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": LEDGER_SYSTEM}, {"role": "user", "content": material}]


def trial_seed(model: str, method: str, prompt_id: str, replicate: int) -> int:
    key = f"{BASE_SEED}|{model}|{method}|{prompt_id}|{replicate}"
    return int(sha_bytes(key.encode())[:8], 16) % (2 ** 31 - 1)


def load_baseline_helpers(baseline: Path) -> tuple[dict[str, str], Any]:
    directory = baseline / "INVAR_dataset"
    if not (directory / "run_vllm_batch.py").is_file():
        raise FileNotFoundError(f"Baseline runner missing: {directory}")
    sys.path.insert(0, str(directory))
    from run_vllm_batch import MODEL_PATHS, prompt_text
    return MODEL_PATHS, prompt_text


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    """Config hashes plus checkpoint file identity; not a full weight-byte hash."""
    config_files = [path / name for name in (
        "config.json", "tokenizer_config.json", "generation_config.json", "model.safetensors.index.json")]
    file_hashes = {file.name: sha_bytes(file.read_bytes()) for file in config_files if file.is_file()}
    weights = sorted((file.name, file.stat().st_size, file.stat().st_mtime_ns)
                     for file in path.glob("*.safetensors"))
    return {"resolved_weights": str(path.resolve()), "config_file_sha256": file_hashes,
            "weight_file_metadata": weights,
            "fingerprint_note": "Weight metadata/config fingerprint; full weight bytes not hashed"}


def mark_complete(output_dir: Path, model: str, methods: list[str], config_hash: str,
                  planned_per_method: int) -> None:
    """Rebuild final cost/count manifests even after a last-batch interruption."""
    summaries = {}
    for method in methods:
        file = output_dir / f"{method}.jsonl"
        count = input_tokens = completion_tokens = fallback_rows = 0
        fallback_prompts: set[str] = set()
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["config_sha256"] != config_hash or row["status"] != "ok":
                    raise ValueError("Cannot mark invalid or mixed outputs complete")
                count += 1
                input_tokens += row["input_tokens"]
                completion_tokens += row["completion_tokens"]
                fallback_rows += bool(row["fallback"])
                if row["fallback"]:
                    fallback_prompts.add(row["prompt_id"])
        if count != planned_per_method:
            raise ValueError("Completion count does not match plan")
        summary = {"model": model, "method": method, "status": "complete", "rows": count,
                   "config_sha256": config_hash, "responses_sha256": sha_bytes(file.read_bytes()),
                   "decision_input_tokens": input_tokens, "decision_completion_tokens": completion_tokens,
                   "fallback_rows": fallback_rows, "fallback_prompts": len(fallback_prompts)}
        atomic_json(output_dir / f"{method}.complete.json", summary)
        summaries[method] = summary
    atomic_json(output_dir / "COMPLETE.json", {"model": model, "status": "complete",
                "config_sha256": config_hash, "methods": summaries,
                "ledger_cost_file": "ledger_cost.json" if (output_dir / "ledger_cost.json").exists() else None,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    old_failure = output_dir / "FAILED.json"
    if old_failure.exists():
        # Keep the failure history; an infrastructure resume may now be complete.
        old_failure.replace(output_dir / ("FAILED.resolved_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".json"))


def run(args: argparse.Namespace) -> None:
    model_paths, render_prompt = load_baseline_helpers(args.baseline)
    if args.model not in model_paths:
        raise ValueError(f"Model must be one of {list(model_paths)}")
    methods = list(dict.fromkeys(args.methods.split(",")))
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"Allowed methods: {METHODS}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    selected_ids = manifest["selected_prompt_ids"]
    if not isinstance(selected_ids, list) or not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected_prompt_ids must be a nonempty unique list")
    all_rows = read_rows(args.baseline / "inputs.jsonl")
    by_id = {row["prompt_id"]: row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("Duplicate baseline prompt IDs")
    unknown = set(selected_ids) - by_id.keys()
    if unknown:
        raise ValueError(f"Unknown selected prompt IDs: {sorted(unknown)[:3]}")
    rows = [by_id[prompt_id] for prompt_id in selected_ids]
    for row in rows:
        if row["task_type"] not in REPETITIONS:
            raise ValueError("Unknown task type")
        sanitized_messages(row)
        if not isinstance(row.get("output_schema"), dict):
            raise ValueError("Missing public output schema")
    batch_size = args.batch_size or (96 if args.model == "qwen25_14b_instruct" else 192)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    output_dir = args.out / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    path = Path(model_paths[args.model])
    fingerprint = checkpoint_fingerprint(path)
    # Read package metadata without importing torch or initializing CUDA.
    versions = {package: importlib.metadata.version(package) for package in ("torch", "transformers", "vllm")}
    config = {
        "protocol": "INVAR_prompt_methods_v1_20261025", "model": args.model,
        "versions": versions,
        "model_fingerprint": fingerprint, "baseline_inputs_sha256": sha_bytes((args.baseline / "inputs.jsonl").read_bytes()),
        "manifest_sha256": sha_bytes(args.manifest.read_bytes()), "runner_sha256": sha_bytes(Path(__file__).read_bytes()),
        "baseline_renderer_sha256": sha_bytes((args.baseline / "INVAR_dataset" / "run_vllm_batch.py").read_bytes()),
        "temperature": 0.7, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0,
        "max_tokens": 128, "base_seed": BASE_SEED, "batch_size": batch_size,
        "repetitions": REPETITIONS, "ledger_temperature": 0.0, "ledger_max_tokens": 768,
        "ledger_seed": BASE_SEED, "decoder": "json_schema_xgrammar_compact",
        "seed_policy": "sha256(base_seed,model,method,prompt_id,replicate)",
        "correctness_labels_given_to_model_or_decoder": False,
        "method_prompt_sha256": {"generic": digest(GENERIC), "social_audit": digest(SOCIAL_AUDIT),
                                 "ledger": digest(LEDGER_SYSTEM)},
    }
    config_hash = digest(config)
    config_path = output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8"))["config_sha256"] != config_hash:
        raise ValueError("Configuration changed; use a new output directory")
    atomic_json(config_path, {"config": config, "config_sha256": config_hash})

    # Validate saved rows before deciding whether CUDA is necessary.
    done_by_method: dict[str, set[str]] = {}
    for method in methods:
        seen: set[str] = set()
        file = output_dir / f"{method}.jsonl"
        if file.exists():
            for record in read_rows(file):
                if record.get("config_sha256") != config_hash or record.get("method") != method:
                    raise ValueError("Mixed saved configurations/methods")
                pid, rep = record["prompt_id"], record["replicate"]
                if pid not in selected_ids or type(rep) is not int or not 0 <= rep < REPETITIONS[by_id[pid]["task_type"]]:
                    raise ValueError("Saved trial not in frozen plan")
                expected_id = f"{args.model}|{method}|{pid}|{rep}"
                if record["trial_id"] != expected_id or expected_id in seen:
                    raise ValueError("Duplicate or malformed saved trial")
                if record.get("status") != "ok":
                    raise ValueError("Saved invalid output requires explicit investigation")
                validate_output(record["response_text"], by_id[pid]["output_schema"], record.get("finish_reason"))
                seen.add(expected_id)
        done_by_method[method] = seen
    total_per_method = sum(REPETITIONS[row["task_type"]] for row in rows)
    if all(len(done_by_method[method]) == total_per_method for method in methods):
        mark_complete(output_dir, args.model, methods, config_hash, total_per_method)
        print(json_text({"status": "already_complete", "model": args.model,
                         "methods": methods, "rows_per_method": total_per_method}), flush=True)
        return

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    native = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
    context = min(8192, getattr(native, "max_position_embeddings", 8192))
    identity = {"config_sha256": config_hash, "config": config, "versions": versions,
                "context": context, "selected_inputs": len(rows), "planned_rows_per_method": total_per_method}
    atomic_json(output_dir / "identity.json", identity)
    llm = LLM(model=str(path), tokenizer=str(path), trust_remote_code=True, dtype="bfloat16",
              max_model_len=context, gpu_memory_utilization=0.90, max_num_seqs=batch_size,
              enable_prefix_caching=True, generation_config="vllm", seed=BASE_SEED,
              structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True})
    grammars = {row["prompt_id"]: StructuredOutputsParams(json=row["output_schema"], disable_any_whitespace=True)
                for row in rows}

    ledger_by_pid: dict[str, dict[str, Any]] = {}
    if any(method.startswith("ledger_") for method in methods):
        ledger_cache_file = output_dir / "ledger_cache.jsonl"
        extraction_config_hash = digest({"model_fingerprint": fingerprint, "system": LEDGER_SYSTEM,
                                         "schema": LEDGER_SCHEMA, "temperature": 0, "max_tokens": 768,
                                         "seed": BASE_SEED, "renderer": config["baseline_renderer_sha256"],
                                         "runner": config["runner_sha256"], "versions": versions})
        cache: dict[str, dict[str, Any]] = {}
        if ledger_cache_file.exists():
            for record in read_rows(ledger_cache_file):
                if record.get("extraction_config_sha256") != extraction_config_hash:
                    raise ValueError("Ledger cache extraction configuration changed")
                if record["cache_key"] in cache:
                    raise ValueError("Duplicate ledger cache entry")
                cache[record["cache_key"]] = record
        pending: dict[str, tuple[str, str, int]] = {}
        pid_to_key: dict[str, str] = {}
        for row in rows:
            pid = row["prompt_id"]
            try:
                material, _, _ = split_material_question(sanitized_messages(row))
            except ValueError as error:
                ledger_by_pid[pid] = {"ledger": None, "fallback": True, "fallback_reason": str(error),
                                      "cache_key": None, "input_tokens": 0, "completion_tokens": 0}
                continue
            key = digest({"material": material, "extraction_config_sha256": extraction_config_hash})
            pid_to_key[pid] = key
            if key not in cache and key not in pending:
                text = render_prompt(tokenizer, extractor_messages(material), args.model)
                tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if tokens + 768 > context:
                    record = {"cache_key": key, "extraction_config_sha256": extraction_config_hash,
                              "material_sha256": digest(material), "ledger": None, "fallback": True,
                              "fallback_reason": "ledger_context_overflow", "response_text": "",
                              "input_tokens": tokens, "completion_tokens": 0, "finish_reason": None,
                              "seconds": 0.0}
                    append_rows(ledger_cache_file, [record])
                    cache[key] = record
                else:
                    pending[key] = (material, text, tokens)
        pending_items = list(pending.items())
        ledger_grammar = StructuredOutputsParams(json=LEDGER_SCHEMA, disable_any_whitespace=True)
        for start in range(0, len(pending_items), batch_size):
            batch = pending_items[start:start + batch_size]
            parameters = [SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                                         max_tokens=768, seed=BASE_SEED, structured_outputs=ledger_grammar)
                          for _ in batch]
            began = time.time()
            generated = llm.generate([data[1] for _, data in batch], parameters, use_tqdm=False)
            if len(generated) != len(batch):
                raise RuntimeError("Ledger generation count mismatch")
            seconds = time.time() - began
            records = []
            for (key, (material, text, input_tokens)), result in zip(batch, generated):
                if len(result.outputs) != 1:
                    raise RuntimeError("Ledger missing completion")
                completion = result.outputs[0]
                ledger, reason = check_ledger(completion.text, material, completion.finish_reason)
                record = {"cache_key": key, "extraction_config_sha256": extraction_config_hash,
                          "material_sha256": digest(material), "rendered_sha256": digest(text),
                          "ledger": ledger, "fallback": reason is not None, "fallback_reason": reason,
                          "response_text": completion.text, "finish_reason": completion.finish_reason,
                          "input_tokens": input_tokens, "completion_tokens": len(completion.token_ids),
                          "seconds": seconds / len(batch), "seconds_note": "shared batch walltime divided by batch size"}
                records.append(record)
                cache[key] = record
            append_rows(ledger_cache_file, records)
            print(json_text({"model": args.model, "stage": "ledger", "done": start + len(batch),
                             "new_materials": len(pending_items)}), flush=True)
        for pid, key in pid_to_key.items():
            ledger_by_pid[pid] = cache[key]
        # Raw cache entries are paid once per unique material, not once per use/method.
        atomic_json(output_dir / "ledger_cost.json", {
            "extraction_config_sha256": extraction_config_hash,
            "unique_materials": len(set(pid_to_key.values())),
            "input_tokens": sum(cache[key]["input_tokens"] for key in set(pid_to_key.values())),
            "completion_tokens": sum(cache[key]["completion_tokens"] for key in set(pid_to_key.values())),
            "fallback_prompt_count": sum(bool(record["fallback"]) for record in ledger_by_pid.values()),
            "prompt_count": len(rows), "cost_scope": "One extraction shared across both ledger architectures and all repetitions",
        })

    began_all = time.time()
    for method in methods:
        output_file = output_dir / f"{method}.jsonl"
        done = done_by_method[method]
        prepared: dict[str, dict[str, Any]] = {}
        for row in rows:
            pid = row["prompt_id"]
            ledger_record = ledger_by_pid.get(pid) if method.startswith("ledger_") else None
            ledger = ledger_record["ledger"] if ledger_record else None
            messages = method_messages(row, method, ledger)
            text = render_prompt(tokenizer, messages, args.model)
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            fallback_reason = ledger_record.get("fallback_reason") if ledger_record else None
            fallback = bool(ledger_record and ledger_record["fallback"])
            if tokens + 128 > context and method.startswith("ledger_"):
                messages = method_messages(row, "baseline")
                text = render_prompt(tokenizer, messages, args.model)
                tokens = len(tokenizer.encode(text, add_special_tokens=False))
                fallback, fallback_reason = True, "decision_context_overflow_after_ledger"
            if tokens + 128 > context:
                raise ValueError(f"Context overflow before generation: method={method}, prompt={pid}")
            prepared[pid] = {"text": text, "input_tokens": tokens, "messages_sha256": digest(messages),
                             "rendered_sha256": digest(text), "fallback": fallback,
                             "fallback_reason": fallback_reason,
                             "ledger_cache_key": ledger_record["cache_key"] if ledger_record else None}
        prompt_hashes = {pid: {key: value for key, value in data.items() if key != "text"}
                         for pid, data in prepared.items()}
        atomic_json(output_dir / f"{method}.prompts.json", prompt_hashes)
        pending_trials = []
        for row in rows:
            for replicate in range(REPETITIONS[row["task_type"]]):
                trial_id = f"{args.model}|{method}|{row['prompt_id']}|{replicate}"
                if trial_id not in done:
                    pending_trials.append((row, replicate, trial_id))
        for start in range(0, len(pending_trials), batch_size):
            batch = pending_trials[start:start + batch_size]
            seeds = [trial_seed(args.model, method, row["prompt_id"], replicate)
                     for row, replicate, _ in batch]
            parameters = [SamplingParams(temperature=0.7, top_p=0.95, top_k=-1, repetition_penalty=1.0,
                                         max_tokens=128, seed=seed, structured_outputs=grammars[row["prompt_id"]])
                          for (row, _, _), seed in zip(batch, seeds)]
            began = time.time()
            generated = llm.generate([prepared[row["prompt_id"]]["text"] for row, _, _ in batch],
                                     parameters, use_tqdm=False)
            if len(generated) != len(batch):
                raise RuntimeError("Decision generation count mismatch")
            seconds = time.time() - began
            records, invalid = [], []
            for (row, replicate, trial_id), result, seed in zip(batch, generated, seeds):
                if len(result.outputs) != 1:
                    raise RuntimeError("Decision missing completion")
                completion = result.outputs[0]
                data = prepared[row["prompt_id"]]
                record = {"model": args.model, "method": method, "prompt_id": row["prompt_id"],
                          "task_type": row["task_type"], "replicate": replicate, "batch": "main", "trial_id": trial_id,
                          "response_text": completion.text, "status": "ok", "finish_reason": completion.finish_reason,
                          "input_tokens": data["input_tokens"], "completion_tokens": len(completion.token_ids),
                          "sampling_seed": seed, "config_sha256": config_hash,
                          "messages_sha256": data["messages_sha256"], "rendered_sha256": data["rendered_sha256"],
                          "ledger_cache_key": data["ledger_cache_key"], "fallback": data["fallback"],
                          "fallback_reason": data["fallback_reason"], "batch_seconds": seconds,
                          "batch_size_actual": len(batch)}
                try:
                    validate_output(completion.text, row["output_schema"], completion.finish_reason)
                except (ValueError, TypeError, KeyError) as error:
                    record["status"], record["validation_error"] = "invalid_output", str(error)
                    invalid.append(trial_id)
                records.append(record)
            # Preserve invalid completions, then stop. Never silently reroll an answer.
            append_rows(output_file, records)
            if invalid:
                raise RuntimeError(f"Invalid decision output retained; investigate {invalid[:3]}")
            progress = {"model": args.model, "method": method,
                        "done": len(done) + start + len(batch), "total": total_per_method,
                        "seconds": round(time.time() - began_all, 2),
                        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            atomic_json(output_dir / "progress.json", progress)
            if start // batch_size % 20 == 0:
                print(json_text(progress), flush=True)
        atomic_json(output_dir / f"{method}.complete.json", {
            "model": args.model, "method": method, "status": "complete", "rows": total_per_method,
            "config_sha256": config_hash, "responses_sha256": sha_bytes(output_file.read_bytes()),
            "fallback_prompts": sum(data["fallback"] for data in prepared.values()),
        })
    mark_complete(output_dir, args.model, methods, config_hash, total_per_method)
    print(json_text({"model": args.model, "status": "complete", "methods": methods,
                     "rows_per_method": total_per_method}), flush=True)


def self_test() -> None:
    """Meaningful boundary/metadata/schema checks, no model or CUDA imports."""
    goal_material = "【个体设定】\n你是P2。\n【实验规则】\nP1得5点，P2得9点。\n【本次明确目标】\n最大化本人点数。\n\n"
    question = "【本次决策】\n只输出JSON。"
    material, actual_question, _ = split_material_question([{"role": "user", "content": goal_material + question}])
    assert material == goal_material and actual_question == question
    assert "最大化本人点数" in material
    assert "最大化本人点数" in extractor_messages(material)[1]["content"]
    fact_text = "【实验规则】P1自主选择。\n【独立事实检查】\nA：P1自主选择。B：计算机决定。"
    fact_material, fact_question, _ = split_material_question([{"role": "user", "content": fact_text}])
    assert "B：计算机决定" not in fact_material and fact_question.startswith("【独立事实检查】")
    role_material, role_question, _ = split_material_question([
        {"role": "user", "content": "你是Q7。\n【事实理解检查】编号是什么？"}])
    assert role_material == "你是Q7。\n" and role_question.startswith("【事实理解检查】")
    early_material, early_question, _ = split_material_question([
        {"role": "user", "content": "规则。\n【独立事实检查】A或B。\n【本次决策】后置。"}])
    assert early_material == "规则。\n" and early_question.startswith("【独立事实检查】")
    ledger = {"claims": [{"fact": "你是P2。", "evidence": "你是P2。"}], "unknowns": []}
    assert check_ledger(json_text(ledger), goal_material, "stop") == (ledger, None)
    bad_ledger = {"claims": [{"fact": "计算机决定。", "evidence": "计算机决定。"}], "unknowns": []}
    assert check_ledger(json_text(bad_ledger), goal_material, "stop")[0] is None
    assert check_ledger(json_text(ledger), goal_material, "length")[0] is None
    assert check_ledger("not JSON", goal_material, "stop")[0] is None
    schema = {"type": "object", "additionalProperties": False, "required": ["action"],
              "properties": {"action": {"type": "integer", "minimum": 0, "maximum": 10}}}
    validate_output('{"action":3}', schema, "stop")
    for invalid in ('{"action":true}', '{"action":3.0}', '{"action":11}', '{"action":3,"extra":1}'):
        try:
            validate_output(invalid, schema, "stop")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid schema accepted")
    clean_messages = [{"role": "system", "content": "只输出JSON。"},
                      {"role": "user", "content": goal_material + question}]
    row = {"messages": copy.deepcopy(clean_messages), "expected_answer": "SECRET_TRUTH_391",
           "expected_actions": ["SECRET_ACTION_392"], "case_id": "SECRET_CASE_393",
           "variant_id": "SECRET_VARIANT_394", "payoffs": {"SECRET_PAYOFF_395": 99},
           "uses": [{"condition": "SECRET_CONDITION_396"}]}
    for method in METHODS:
        prepared = method_messages(row, method, ledger)
        text = json_text(prepared)
        assert "SECRET_" not in text
        if method == "baseline":
            assert prepared == clean_messages
        if method.startswith("ledger_"):
            assert question in "\n".join(message["content"] for message in prepared)
            assert method_messages(row, method, None) == clean_messages
    assert row["messages"] == clean_messages, "Builder mutated source messages"
    original = method_messages(row, "ledger_original", ledger)[1]["content"]
    only = method_messages(row, "ledger_only", ledger)[1]["content"]
    assert original.startswith(goal_material)
    assert "P1得5点，P2得9点。" not in only
    assert original.endswith(question) and only.endswith(question)
    assert trial_seed("model", "baseline", "pid", 0) == trial_seed("model", "baseline", "pid", 0)
    assert trial_seed("model", "baseline", "pid", 0) != trial_seed("model", "generic", "pid", 0)
    print(json_text({"self_test": "passed", "checks": ["earliest_question_boundary", "goal_kept_in_material",
                      "fact_options_not_in_extractor", "all_three_markers", "evidence_substring",
                      "truncation_fallback", "invalid_json_fallback", "strict_schema_types",
                      "metadata_never_serialized", "baseline_exact_messages", "ledger_shared_builder",
                      "fallback_original_messages", "input_not_mutated", "reproducible_fresh_seeds"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("baseline", "manifest", "model", "out")):
        parser.error("--baseline, --manifest, --model and --out are required unless --self-test")
    try:
        run(args)
    except Exception as error:
        if args.out and args.model:
            directory = args.out / args.model
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "FAILED.json", {"error": f"{type(error).__name__}: {error}",
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise


if __name__ == "__main__":
    main()
