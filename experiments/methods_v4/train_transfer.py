"""Retrospective INVAR source/model/seed transfer: SFT versus free JS.

Read-only reuse of frozen V2 preparation, candidate likelihood and suffix
alignment. No teacher, KL, cache generation or evaluation/scheduler changes.
--prepare-only imports no torch/transformers/PEFT. --self-test is CPU tensors.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROTOCOL = "INVAR_retrospective_source_model_seed_transfer_20261026_v1"
MODEL_TYPES = {"qwen3_8b": "qwen3", "qwen25_14b_instruct": "qwen2"}
MODEL_PATHS = {"qwen3_8b": "/workspace/models/Qwen3-8B",
               "qwen25_14b_instruct": "/workspace/models/Qwen2.5-14B-Instruct"}
SOURCES = ("HPVBG", "IGVOFO", "PGG", "PIQ572")
SEEDS = (20261026, 20261028, 20261029)
ARMS = {"anchor_sft": 0.0, "anchor_js": 1.0}


def load_v2(directory):
    path = Path(directory).resolve() / "train_condition_anchor.py"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen V2 helper absent: {path}")
    spec = importlib.util.spec_from_file_location("invar_frozen_v2_transfer_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    v2 = load_v2(args.v2_helper_dir)
    v2.require(args.model_id in MODEL_TYPES and args.heldout_source in SOURCES, "Unapproved model/source")
    v2.require(args.seed in SEEDS, "Training seed must be one of the three frozen seeds")
    v2.require(args.max_steps == 150 and args.max_free_candidates == 16,
               "This transfer stage freezes 150 steps and complete free menus up to 16")
    old = v2.prepare(args)
    legacy, sup, sup_edges, free, free_edges, conditions, schedule, old_audit = old
    pool_hash = v2.digest({"supervised_inputs": sup, "supervised_edges": sup_edges,
                           "free_inputs": free, "free_edges": free_edges, "conditions": conditions})
    audit = {key: value for key, value in old_audit.items() if key != "split_content_sha256"}
    audit.update(protocol=PROTOCOL, preparation_origin_protocol=old_audit["protocol"],
                 v2_preparation_sha256=old_audit["split_content_sha256"],
                 model_id=args.model_id, training_seed=args.seed,
                 train_sources=sorted(set(SOURCES) - {args.heldout_source}),
                 source_pool_sha256=pool_hash,
                 v2_helper_sha256=v2.sha_file(Path(args.v2_helper_dir) / "train_condition_anchor.py"),
                 teacher_required=False, teacher_manifest_sha256=None,
                 teacher_pool=None, lambda_free_anchor=0.0,
                 source_claim="all source materials and earlier results have been inspected; retrospective transfer only")
    audit["split_content_sha256"] = v2.digest(audit)
    return v2, (legacy, sup, sup_edges, free, free_edges, conditions, schedule, audit)


def tokenize_rows(v2, legacy, rows, tokenizer, model_id, max_length):
    """Native per-model chat prefix, complete JSON + EOS suffix, no truncation."""
    if tokenizer.pad_token_id is None:
        v2.require(tokenizer.eos_token_id is not None, "No tokenizer pad/EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs = {"enable_thinking": False} if model_id == "qwen3_8b" else {}
    cache = {}
    audit = {"model_id": model_id, "max_sequence_length": 0, "max_prompt_tokens": 0,
             "max_answer_tokens": 0, "max_projected_logits_positions": 0,
             "template_kwargs": model_kwargs, "max_length_budget": max_length}
    for row in rows:
        prefix = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                               add_generation_prompt=True, **model_kwargs)
        for action in row["candidates"]:
            response = json.dumps({row["field"]: action}, ensure_ascii=False, separators=(",", ":"))
            complete = tokenizer.apply_chat_template(
                row["messages"] + [{"role": "assistant", "content": response}],
                tokenize=True, add_generation_prompt=False, **model_kwargs)
            v2.require(0 < len(prefix) < len(complete) and complete[:len(prefix)] == prefix,
                       "Native chat prefix/full-answer mismatch")
            v2.require(len(complete) <= max_length,
                       f"Context overflow {row['prompt_id']}: {len(complete)}>{max_length}; no truncation")
            labels = [-100] * len(prefix) + complete[len(prefix):]
            keep, suffix = legacy.answer_projection_spec(labels)
            cache[(row["prompt_id"], v2.canonical(action))] = (complete, labels)
            audit["max_sequence_length"] = max(audit["max_sequence_length"], len(complete))
            audit["max_prompt_tokens"] = max(audit["max_prompt_tokens"], len(prefix))
            audit["max_answer_tokens"] = max(audit["max_answer_tokens"], len(suffix))
            audit["max_projected_logits_positions"] = max(audit["max_projected_logits_positions"], keep)
    audit.update(status="passed_no_truncation", cached_candidates=len(cache),
                 model_weights_loaded_by_tokenizer_audit=False, cuda_calls_by_tokenizer_audit=False,
                 labels="complete assistant JSON and native end-of-turn suffix only")
    return cache, audit


def js_divergence(torch, logqa, logqb):
    # Identical JS mathematics to V2 js_and_anchor; no dummy teacher or KL work.
    mixture = torch.logsumexp(torch.stack((logqa, logqb)), dim=0) - math.log(2)
    return 0.5 * ((logqa.exp() * (logqa - mixture)).sum() + (logqb.exp() * (logqb - mixture)).sum())


def d0_supervision(torch, complete_correct_logps, suffix_lengths):
    return -torch.logsumexp(complete_correct_logps, dim=0) / (sum(suffix_lengths) / len(suffix_lengths))


def train(args, v2, prepared, output):
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    legacy, sup, sup_edges, free, free_edges, _, schedule, audit = prepared
    v2.require(not (output / "training_started.json").exists(), "Already started; no optimizer/RNG auto-resume")
    model_path = Path(args.model or MODEL_PATHS[args.model_id]).resolve()
    v2.require(model_path.is_dir(), "Existing local model weights required")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    cache, token_audit = tokenize_rows(v2, legacy, sup + free, tokenizer, args.model_id, args.max_length)
    v2.atomic_json(output / "tokenization_audit.json", token_audit)
    v2.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "One BF16 CUDA device required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    v2.require(model.config.model_type == MODEL_TYPES[args.model_id], "Native model architecture differs from requested model_id")
    v2.require(getattr(model.config, "attention_dropout", 0.0) == 0.0, "Candidate scoring requires zero attention dropout")
    v2.require("logits_to_keep" in inspect.signature(model.forward).parameters, "Suffix-only logits required")
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    v2.require({"q_proj", "v_proj"} <= available, "Native q/v projection modules missing")
    model.config.use_cache = False
    weight_metadata = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(model_path.glob("*.safetensors"))]
    identity = {"base_model_path": str(model_path), "model_id": args.model_id, "model_type": model.config.model_type,
                "base_model_config_sha256": v2.sha_file(model_path / "config.json"),
                "base_weight_file_metadata": weight_metadata, "base_weight_metadata_sha256": v2.digest(weight_metadata),
                "weight_identity_note": "configuration byte hash and weight metadata; not full weight-byte hashes",
                "chat_template_sha256": v2.digest(tokenizer.chat_template),
                "template_kwargs": token_audit["template_kwargs"], "native_template": True,
                "torch_version": torch.__version__, "transformers_version": transformers.__version__, "peft_version": peft.__version__,
                "dtype": "bfloat16", "max_length": args.max_length, "runner_sha256": v2.sha_file(Path(__file__)),
                "v2_helper_sha256": audit["v2_helper_sha256"], "legacy_helper_sha256": audit["legacy_helper_sha256"]}
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                            target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    v2.require(trainable and all("lora_" in name for name, _ in trainable), "Non-adapter weights would update")
    model.cuda().train()
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=1e-4, weight_decay=0.01)
    config = {"protocol": PROTOCOL, "mode": args.arm, "arm": args.arm, "model_id": args.model_id,
              "model": str(model_path), "heldout_source": args.heldout_source, "train_sources": audit["train_sources"],
              "seed": args.seed, "training_seed": args.seed, "input_sha256": audit["input_sha256"],
              "split_content_sha256": audit["split_content_sha256"], "source_pool_sha256": audit["source_pool_sha256"],
              "exposure_schedule_sha256": audit["exposure_schedule_sha256"], "identity": identity,
              "base_model_config_sha256": identity["base_model_config_sha256"],
              "rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"], "dropout": 0.0,
              "learning_rate": 1e-4, "weight_decay": 0.01, "gradient_clip": 1.0,
              "gradient_accumulation": 8, "batch_size": 1, "max_steps": 150, "bf16": True,
              "gradient_checkpointing": True, "lambda_free_js": ARMS[args.arm], "lambda_free_anchor": 0.0,
              "teacher_required": False, "teacher_manifest_sha256": None, "supervised_js": False,
              "supervised_loss": audit["supervised_loss"], "proxy": audit["proxy"],
              "same_exposure": "same immutable supervised/free schedule; both arms retain zero/nonzero JS graph",
              "trainable_parameters": sum(p.numel() for _, p in trainable),
              "confirmation_status": "retrospective_transfer_all_sources_previously_inspected",
              "inference_seed_policy": "not specified by trainer; independent evaluation configuration"}
    v2.atomic_json(output / "training_config.json", config)
    v2.atomic_json(output / "training_started.json", {"unix": time.time(), "config_sha256": v2.sha_file(output / "training_config.json")})
    by_id = {row["prompt_id"]: row for row in sup + free}
    se_by_id = {row["edge_id"]: row for row in sup_edges}
    fe_by_id = {row["edge_id"]: row for row in free_edges}
    def supervised_loss(row):
        logps, lengths = v2.candidate_logps(torch, model, cache, legacy, row, row["expected"])
        return d0_supervision(torch, logps, lengths)
    def distribution(row):
        logps, _ = v2.candidate_logps(torch, model, cache, legacy, row)
        return torch.log_softmax(logps, dim=0)
    started, last_checkpoint = time.time(), None
    exposures = collections.Counter()
    optimizer.zero_grad(set_to_none=True)
    v2.atomic_json(output / "progress.json", {"status": "training", "model_id": args.model_id,
        "arm": args.arm, "heldout_source": args.heldout_source, "training_seed": args.seed,
        "completed_optimizer_steps": 0, "planned_optimizer_steps": 150, "resumable": False})
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(1, 151):
            metrics = collections.defaultdict(float)
            for event in schedule[(step - 1) * 8:step * 8]:
                se, fe = se_by_id[event["supervised_edge_id"]], fe_by_id[event["free_edge_id"]]
                sa, sb, fa, fb = [by_id[pid] for pid in (se["left"], se["right"], fe["left"], fe["right"])]
                v2.require(all(row["source_id"] != args.heldout_source for row in (sa, sb, fa, fb)), "Runtime source leakage")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    supervised = (supervised_loss(sa) + supervised_loss(sb)) / 2
                    pa, pb = distribution(fa), distribution(fb)
                    js = js_divergence(torch, pa, pb)
                    total = supervised + ARMS[args.arm] * js + 0.0 * (pa.sum() + pb.sum())
                v2.require(bool(torch.isfinite(total)), "Nonfinite loss")
                (total / 8).backward()
                for key, val in (("D0_supervised", supervised), ("free_js_proxy", js), ("total_loss", total)):
                    metrics[key] += float(val.detach()) / 8
                entropy = -0.5 * ((pa.exp() * pa).sum() + (pb.exp() * pb).sum())
                metrics["free_entropy_proxy"] += float(entropy.detach()) / 8
                for pid in (se["left"], se["right"], fe["left"], fe["right"]):
                    exposures[pid] += 1
            norm = torch.nn.utils.clip_grad_norm_([param for _, param in trainable], 1.0)
            v2.require(bool(torch.isfinite(norm)), "Nonfinite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, **metrics, "gradient_norm": float(norm), "elapsed_seconds": time.time() - started,
                      "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated()}
            stream.write(v2.canonical(record) + "\n")
            stream.flush()
            if step % 25 == 0:
                checkpoint = output / f"adapter_checkpoint_{step:03d}"
                v2.require(not checkpoint.exists(), "Checkpoint already exists")
                model.save_pretrained(checkpoint, safe_serialization=True)
                v2.atomic_json(checkpoint / "checkpoint_metadata.json", {
                    "completed_optimizer_steps": step, "training_config_sha256": v2.sha_file(output / "training_config.json"),
                    "split_content_sha256": audit["split_content_sha256"], "model_id": args.model_id,
                    "heldout_source": args.heldout_source, "training_seed": args.seed,
                    "teacher_required": False, "teacher_manifest_sha256": None,
                    "adapter_files": {p.name: v2.sha_file(p) for p in checkpoint.iterdir() if p.is_file()},
                    "resumable": False, "optimizer_saved": False, "rng_saved": False})
                last_checkpoint = checkpoint.name
            v2.atomic_json(output / "progress.json", {"status": "training", "model_id": args.model_id, "arm": args.arm,
                "heldout_source": args.heldout_source, "training_seed": args.seed, "completed_optimizer_steps": step,
                "planned_optimizer_steps": 150, "last_adapter_checkpoint": last_checkpoint, "resumable": False, "record": record})
            if step == 1 or step % 10 == 0:
                print(v2.canonical(record), flush=True)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "tokenizer")
    v2.atomic_json(output / "actual_exposures.json", {"schedule_sha256": audit["exposure_schedule_sha256"],
        "prompt_exposure_counts": dict(sorted(exposures.items())), "total_prompt_exposures": sum(exposures.values())})
    complete = {"status": "training_complete_not_yet_evaluated", "steps": 150,
        "config_sha256": v2.sha_file(output / "training_config.json"), "split_content_sha256": audit["split_content_sha256"],
        "model_id": args.model_id, "heldout_source": args.heldout_source, "training_seed": args.seed,
        "exposure_schedule_sha256": audit["exposure_schedule_sha256"], "teacher_required": False,
        "teacher_manifest_sha256": None,
        "adapter_files": {p.name: v2.sha_file(p) for p in (output / "adapter").iterdir() if p.is_file()},
        "exposures_sha256": v2.sha_file(output / "actual_exposures.json"),
        "claim": "retrospective transfer training complete; no claim of improved generation or external confirmation"}
    v2.atomic_json(output / "training_complete.json", complete)
    v2.atomic_json(output / "progress.json", {**complete, "completed_optimizer_steps": 150,
                                               "last_adapter_checkpoint": last_checkpoint, "resumable": False})


def self_test(v2_dir):
    import torch
    v2 = load_v2(v2_dir)
    left = torch.tensor([0.1, 1.4, -0.5], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([-0.8, 0.2, 0.6], dtype=torch.float64, requires_grad=True)
    pa, pb = torch.log_softmax(left, 0), torch.log_softmax(right, 0)
    actual = js_divergence(torch, pa, pb)
    expected, _ = v2.js_and_anchor(torch, pa, pb, pa.detach())
    v2.require(torch.allclose(actual, expected, atol=1e-14), "JS differs from frozen V2")
    grads_a = torch.autograd.grad(actual, (left, right), retain_graph=True)
    grads_b = torch.autograd.grad(expected, (left, right))
    v2.require(all(torch.allclose(a, b, atol=1e-14) and torch.isfinite(a).all() for a, b in zip(grads_a, grads_b)),
               "JS gradient differs from frozen V2")
    scores = torch.tensor([-2.5, -1.2], dtype=torch.float64, requires_grad=True)
    d0 = d0_supervision(torch, scores, [5, 6])
    v2.require(torch.allclose(d0, -torch.logsumexp(scores, 0) / 5.5), "D0 changed")
    d0.backward()
    v2.require(torch.isfinite(scores.grad).all() and (scores.grad < 0).all(), "D0 correct-set gradients invalid")
    zero_a = torch.tensor([0.0, 0.2], dtype=torch.float64, requires_grad=True)
    zero_b = torch.tensor([0.3, -0.1], dtype=torch.float64, requires_grad=True)
    zero_js = 0.0 * js_divergence(torch, torch.log_softmax(zero_a, 0), torch.log_softmax(zero_b, 0))
    zero_js.backward()
    v2.require(torch.equal(zero_a.grad, torch.zeros_like(zero_a)) and torch.equal(zero_b.grad, torch.zeros_like(zero_b)),
               "SFT zero-coefficient behavior graph changes gradients")
    v2.require(not torch.cuda.is_initialized(), "CPU self-test initialized CUDA")
    print(v2.canonical({"self_test": "passed", "device": "cpu", "checks": ["V2_JS_value_gradient_identical",
        "D0_unchanged_finite_gradient", "SFT_free_zero_gradient", "no_teacher", "no_CUDA"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE.parent / "baseline_v2")
    parser.add_argument("--legacy-helper-dir", type=Path,
                        default=Path(os.environ.get("INVAR_METHODS_V1", str(HERE.parent / "methods_v1"))))
    parser.add_argument("--v2-helper-dir", type=Path,
                        default=Path(os.environ.get("INVAR_METHODS_V2", str(HERE.parent / "methods_v2"))))
    parser.add_argument("--model-id", choices=sorted(MODEL_TYPES), default="qwen3_8b")
    parser.add_argument("--model", help="Existing local weight path; defaults to fixed server path for model-id")
    parser.add_argument("--heldout-source", choices=SOURCES, default="HPVBG")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=20261026)
    parser.add_argument("--arm", choices=sorted(ARMS), default="anchor_sft")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-free-candidates", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--tokenizer-audit", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.v2_helper_dir)
        return
    output = args.output or HERE / "learning" / args.model_id / args.heldout_source / f"seed{args.seed}" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "RUN.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "unix": time.time(), "model_id": args.model_id,
                                 "heldout_source": args.heldout_source, "training_seed": args.seed, "arm": args.arm}))
    try:
        v2, prepared = prepare(args)
        v2.save_preparation(output, prepared)
        print(v2.canonical({key: prepared[-1][key] for key in ("protocol", "model_id", "heldout_source", "training_seed",
              "supervised_inputs", "free_inputs", "semantic_conditions", "source_pool_sha256", "exposure_schedule_sha256",
              "split_content_sha256", "teacher_required")}), flush=True)
        if args.prepare_only:
            if args.tokenizer_audit:
                from transformers import AutoTokenizer
                path = args.model or MODEL_PATHS[args.model_id]
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
                _, audit = tokenize_rows(v2, prepared[0], prepared[1] + prepared[3], tokenizer, args.model_id, args.max_length)
                v2.atomic_json(output / "tokenization_audit.json", audit)
                print(v2.canonical(audit), flush=True)
        else:
            train(args, v2, prepared, output)
        lock.unlink()
    except Exception as error:
        failure = output / f"FAILED_{time.time_ns()}.json"
        failure.write_text(json.dumps({"error": str(error), "type": type(error).__name__, "lock_retained": True,
                                       "unix": time.time()}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
