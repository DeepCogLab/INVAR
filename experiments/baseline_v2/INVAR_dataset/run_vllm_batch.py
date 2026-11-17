"""Run INVAR prompts with local vLLM weights; one output file per model.

This runner is intentionally independent of any online API.  It reads a frozen
run plan, batches prompts for one model at a time, and writes one JSONL record
per planned trial.  Existing keys are skipped so an interrupted run can resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


MODEL_PATHS = {
    "qwen3_8b": "/workspace/models/Qwen3-8B",
    "qwen25_7b_instruct": "/workspace/models/Qwen2.5-7B-Instruct",
    "qwen25_7b_base": "/workspace/models/Qwen2.5-7B",
    "qwen25_14b_instruct": "/workspace/models/Qwen2.5-14B-Instruct",
    "glm4_9b": "/workspace/models/GLM-4-9B-0414",
    "llama31_8b_instruct": "/workspace/models/Llama-3.1-8B-Instruct",
    "mistral7b_instruct_v03": "/workspace/models/Mistral-7B-Instruct-v0.3",
    "deepseek_llm7b_chat": "/workspace/models/deepseek-llm-7b-chat",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_text(tokenizer: Any, messages: list[dict[str, str]], model_id: str) -> str:
    """Render a chat prompt, falling back to a neutral plain transcript."""
    if model_id == "mistral7b_instruct_v03":
        messages = [{"role":"user", "content":"\n\n".join(m["content"] for m in messages)}]
    if model_id == "qwen25_7b_base":
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if model_id.startswith("qwen3_"):
            # Qwen3 supports this keyword in its chat template.  If a local
            # template version does not, retry without it below.
            try:
                return tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **kwargs
                )
            except TypeError:
                pass
        return tokenizer.apply_chat_template(messages, **kwargs)
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"


def trial_seed(model_id: str, row: dict[str, Any]) -> int:
    key = f"{model_id}|{row['prompt_id']}|{row['replicate']}|{row.get('batch', 'main')}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--model-id", choices=sorted(MODEL_PATHS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.70)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")

    plan = [r for r in read_jsonl(args.plan) if r.get("model") == args.model_id]
    prompts = {p["prompt_id"]: p for p in read_jsonl(args.prompts)}
    if not plan:
        raise SystemExit(f"no plan rows for {args.model_id}")
    missing = sorted({r["prompt_id"] for r in plan} - prompts.keys())
    if missing:
        raise SystemExit(f"unknown prompt IDs: {missing[:3]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int, str]] = set()
    if args.out.exists():
        for row in read_jsonl(args.out):
            done.add((row.get("prompt_id"), int(row.get("replicate", -1)), row.get("batch", "main")))
    pending = [
        r for r in plan
        if (r["prompt_id"], int(r["replicate"]), r.get("batch", "main")) not in done
    ]
    if not pending:
        print(json.dumps({"model": args.model_id, "status": "already_complete", "rows": len(plan)}))
        return

    # Import after argument parsing so --help works without CUDA/vLLM imports.
    from transformers import AutoTokenizer, AutoConfig
    from vllm import LLM, SamplingParams

    model_path = MODEL_PATHS[args.model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    native = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    context = min(args.max_model_len, getattr(native, "max_position_embeddings", args.max_model_len))
    rendered = {pid: prompt_text(tokenizer, p["messages"], args.model_id) for pid,p in prompts.items()}
    lengths = {pid: len(tokenizer.encode(t, add_special_tokens=False)) for pid,t in rendered.items()}
    if max(lengths.values()) + args.max_output_tokens > context:
        raise ValueError(f"Context overflow: longest={max(lengths.values())}, output={args.max_output_tokens}, native={context}")
    identity = {"model":args.model_id,"resolved_weights":str(Path(model_path).resolve()),
        "config_sha256":hashlib.sha256(Path(model_path,"config.json").read_bytes()).hexdigest(),
        "rendered_prompt_sha256":hashlib.sha256(json.dumps(rendered,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        "max_input_tokens":max(lengths.values()),"effective_context":context,
        "template_policy":"base_plain_transcript" if args.model_id.endswith("_base") else "native_chat_system_merged_for_mistral",
        "versions":{k:__import__("importlib.metadata",fromlist=["version"]).version(k) for k in ("torch","transformers","vllm")}}
    args.out.with_suffix(".identity.json").write_text(json.dumps(identity,indent=2),encoding="utf-8")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=context,
        generation_config="vllm",
        max_num_seqs=max(args.batch_size, 32),
        enable_prefix_caching=True,
        seed=args.seed,
    )
    config = {
        "snapshot": args.model_id,
        "weights_path": model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "max_model_len": context,
        "top_k": -1, "repetition_penalty": 1.0, "seed_policy":"sha256_trial_v1",
        "template_policy":identity["template_policy"],
        "batch_size": args.batch_size,
        "runner": "vllm_offline_batch",
        "qwen3_enable_thinking": False if args.model_id.startswith("qwen3_") else None,
    }
    started = time.time()
    with args.out.open("a", encoding="utf-8") as fout:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            texts = [rendered[r["prompt_id"]] for r in batch]
            params = [
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_output_tokens,
                    top_k=-1, repetition_penalty=1.0,
                    seed=(args.seed + trial_seed(args.model_id, r)) % (2**31 - 1),
                )
                for r in batch
            ]
            try:
                outputs = llm.generate(texts, sampling_params=params, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise RuntimeError("Output count mismatch")
                for row, output in zip(batch, outputs):
                    text = output.outputs[0].text if output.outputs else ""
                    record = {
                        "model": args.model_id,
                        "prompt_id": row["prompt_id"],
                        "replicate": int(row["replicate"]),
                        "batch": row.get("batch", "main"),
                        "status": "ok",
                        "response_text": text,
                        "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                        "completion_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
                        "input_tokens": lengths[row["prompt_id"]],
                        "sampling_seed": (args.seed + trial_seed(args.model_id,row)) % (2**31-1),
                        "generation_config": config,
                        "trial_id": row.get("trial_id"),
                        "order": row.get("order"),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:  # preserve failed batch, then stop infrastructure failure
                error = f"{type(exc).__name__}: {exc}"
                for row in batch:
                    record = {
                        "model": args.model_id,
                        "prompt_id": row["prompt_id"],
                        "replicate": int(row["replicate"]),
                        "batch": row.get("batch", "main"),
                        "status": "service_error",
                        "response_text": "",
                        "error": error[:1000],
                        "generation_config": config,
                        "trial_id": row.get("trial_id"),
                        "order": row.get("order"),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                raise
            fout.flush()
            if (start // args.batch_size) % 10 == 0:
                elapsed = max(time.time() - started, 1.0)
                print(json.dumps({"model": args.model_id, "done": start + len(batch), "total": len(pending), "rows_per_second": round((start + len(batch)) / elapsed, 2)}), flush=True)
    print(json.dumps({"model": args.model_id, "status": "complete", "rows": len(plan), "new_rows": len(pending), "out": str(args.out)}))


if __name__ == "__main__":
    main()
