"""Small INVAR supervised LoRA pilot; default prepare-only needs no ML imports.

This is a development experiment, not confirmation of improved reliability.
Both modes use exactly the same supervised equivalent-presentation pairs.
The consistency term is JS of normalized COMPLETE legal-answer sequence
likelihoods. It is a proxy, not vLLM grammar-constrained sampling probability.

Examples (run from project root):
  python experiments/methods_v1/train_lora.py --prepare-only
  python experiments/methods_v1/train_lora.py --prepare-only --tokenizer-audit --model /models/Qwen3-8B
  python experiments/methods_v1/train_lora.py --mode sft --model /models/Qwen3-8B
  python experiments/methods_v1/train_lora.py --mode sft_consistency --model /models/Qwen3-8B
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import math
import random
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXCLUDED_VARIANTS = frozenset({"V4_rules_table", "V6_relation"})
SUPERVISED_TYPES = frozenset({"changed_fact", "stable_fact", "goal"})
VERSION = "invar_lora_pilot_0.1"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def answer_spec(row):
    props = row["output_schema"]["properties"]
    require(len(props) == 1, "Exactly one output field required")
    field, spec = next(iter(props.items()))
    if "enum" in spec:
        candidates = list(spec["enum"])
    elif spec.get("type") == "integer":
        require("minimum" in spec and "maximum" in spec, "Bounded integer menu required")
        candidates = list(range(spec["minimum"], spec["maximum"] + 1))
    else:
        raise ValueError("Unsupported legal output menu")
    require(1 < len(candidates) <= 256, "Unexpected legal menu size")
    expected = row["expected_actions"] if row["task_type"] == "goal" else [row["expected_answer"]]
    require(expected and all(a in candidates for a in expected), "Invalid supervised answer set")
    require(len(expected) == len(set(expected)), "Duplicate expected answers")
    return field, candidates, expected


def prepare(baseline, heldout_source):
    """Resolve provenance using ALL quartet references before selecting examples."""
    baseline = Path(baseline).resolve()
    paths = {
        "inputs": baseline / "inputs.jsonl",
        "frozen_manifest": baseline / "frozen_manifest.json",
        "pairs": baseline / "INVAR_dataset/data/pairs.jsonl",
        "fact_quartets": baseline / "fact_quartets.jsonl",
        "goal_quartets": baseline / "goal_quartets.jsonl",
        "behavior_quartets": baseline / "INVAR_dataset/data/quartets.jsonl",
    }
    for name, path in paths.items():
        require(path.is_file(), f"Missing {name}: {path}")
    frozen = json.loads(paths["frozen_manifest"].read_text(encoding="utf-8-sig"))
    require(sha256(paths["inputs"]) == frozen["input_sha256"], "Baseline input hash mismatch")
    rows = read_jsonl(paths["inputs"])
    inputs = {r["prompt_id"]: r for r in rows}
    require(len(inputs) == len(rows), "Duplicate prompt ID")
    pairs = {r["pair_id"]: r for r in read_jsonl(paths["pairs"])}
    all_sources = {r["source_id"] for r in pairs.values()}
    require(heldout_source in all_sources, "Held-out source does not exist")
    references = collections.defaultdict(set)

    def add_reference(pid, pair_id):
        require(pid in inputs, f"Unknown referenced input {pid}")
        require(pair_id in pairs, f"Unknown referenced condition pair {pair_id}")
        references[pid].add(pair_id)

    all_quartets = {}
    for kind in ("fact", "goal", "behavior"):
        qs = read_jsonl(paths[kind + "_quartets"])
        all_quartets[kind] = qs
        for q in qs:
            for side in "ABCD":
                add_reference(q[side], q["pair_id"])
    for row in rows:
        pid = row["prompt_id"]
        if row.get("pair_id"):
            add_reference(pid, row["pair_id"])
        for use in row.get("uses", []):
            add_reference(pid, use["pair_id"])

    accepted, rejected = {}, collections.Counter()
    source_closure = {}
    for row in rows:
        if row.get("task_type") not in SUPERVISED_TYPES:
            rejected["not_supervised_truth"] += 1
            continue
        pid = row["prompt_id"]
        require(references[pid], f"No provenance closure for supervised prompt {pid}")
        sources = sorted({pairs[p]["source_id"] for p in references[pid]})
        source_closure[pid] = sources
        if len(sources) != 1:
            rejected["mixed_source_prompt"] += 1
            continue
        if sources[0] == heldout_source:
            rejected["heldout_source"] += 1
            continue
        if row.get("variant_id") in EXCLUDED_VARIANTS:
            rejected["heldout_variant"] += 1
            continue
        require(isinstance(row["messages"], list) and len(row["messages"]) == 2,
                f"Unexpected messages in {pid}")
        require([m["role"] for m in row["messages"]] == ["system", "user"], "Input role mismatch")
        field, menu, expected = answer_spec(row)
        semantic_menu = list(row.get("option_semantic_order") or menu)
        require(len(semantic_menu) == len(menu) and len(set(semantic_menu)) == len(menu),
                "Invalid candidate semantic mapping")
        if row["task_type"] != "goal":
            require(semantic_menu[menu.index(expected[0])] == row["semantic_answer"],
                    "Fact reference answer and semantic option mapping disagree")
        accepted[pid] = {
            "prompt_id": pid, "task_type": row["task_type"], "task": row["task"],
            "source_id": sources[0], "pair_references": sorted(references[pid]),
            "variant_id": row.get("variant_id", "base"), "messages": row["messages"],
            "field": field, "candidates": menu, "expected": expected,
            "candidate_semantics": semantic_menu,
            "option_semantic_order": row.get("option_semantic_order"),
            "messages_sha256": digest(row["messages"]),
        }

    # Compare A with B or C with D only. A with C is a real intervention.
    edges = {}
    edge_rejections = collections.Counter()
    for kind in ("fact", "goal"):
        for q in all_quartets[kind]:
            if q["variant_id"] in EXCLUDED_VARIANTS:
                edge_rejections["heldout_variant_quartet"] += 1
                continue
            for left_side, right_side in (("A", "B"), ("C", "D")):
                left, right = q[left_side], q[right_side]
                if left not in accepted or right not in accepted:
                    edge_rejections["unavailable_endpoint"] += 1
                    continue
                if left == right:
                    edge_rejections["identical_prompt"] += 1
                    continue
                a, b = accepted[left], accepted[right]
                require(a["source_id"] == b["source_id"] != heldout_source,
                        "Test or mixed-source prompt in selected edge")
                require(a["task_type"] == b["task_type"], "Inconsistent task in equivalence edge")
                require(a["field"] == b["field"] and a["candidates"] == b["candidates"],
                        "Legal action mismatch in equivalence edge")
                semantic_correct_a = {a["candidate_semantics"][a["candidates"].index(t)] for t in a["expected"]}
                semantic_correct_b = {b["candidate_semantics"][b["candidates"].index(t)] for t in b["expected"]}
                require(semantic_correct_a == semantic_correct_b, "Semantic answers differ on equivalent edge")
                require(set(a["candidate_semantics"]) == set(b["candidate_semantics"]),
                        "Incompatible semantic action menus")
                key = tuple(sorted((left, right)))
                if key not in edges:
                    edges[key] = {
                        "edge_id": "eq_" + digest(key)[:20], "left": key[0], "right": key[1],
                        "task_type": a["task_type"], "source_id": a["source_id"],
                        "references": [],
                    }
                edges[key]["references"].append({
                    "quartet_id": q["quartet_id"], "pair_id": q["pair_id"],
                    "sides": left_side + right_side, "variant_id": q["variant_id"],
                })

    require(edges, "No eligible equivalent supervised pairs")
    used_ids = {e[k] for e in edges.values() for k in ("left", "right")}
    examples = [accepted[p] for p in sorted(used_ids)]
    edge_rows = sorted(edges.values(), key=lambda e: e["edge_id"])
    require({r["task_type"] for r in examples} == SUPERVISED_TYPES,
            "All three supervised tasks must be represented")
    # Repeat non-negotiable checks on the FINAL training material.
    for row in examples:
        require(source_closure[row["prompt_id"]] == [row["source_id"]], "Mixed-source training row")
        require(row["source_id"] != heldout_source, "Held-out source in training")
        require(row["variant_id"] not in EXCLUDED_VARIANTS, "Held-out variant in training")
        require(all(pairs[p]["source_id"] != heldout_source for p in row["pair_references"]),
                "Held-out source in referenced closure")
    by_type = dict(collections.Counter(r["task_type"] for r in examples))
    audit = {
        "version": VERSION, "status": "cpu_data_audit_passed", "heldout_source": heldout_source,
        "train_sources": sorted(all_sources - {heldout_source}),
        "heldout_variants": sorted(EXCLUDED_VARIANTS), "input_sha256": sha256(paths["inputs"]),
        "source_file_hashes": {name: sha256(path) for name, path in paths.items()},
        "unique_supervised_training_inputs": len(examples), "unique_equivalence_edges": len(edge_rows),
        "training_inputs_by_type": by_type,
        "training_inputs_by_source": dict(collections.Counter(r["source_id"] for r in examples)),
        "edges_by_type": dict(collections.Counter(e["task_type"] for e in edge_rows)),
        "rejected_input_counts": dict(rejected), "rejected_edge_counts": dict(edge_rejections),
        "heldout_source_rows_in_training": 0, "mixed_source_rows_in_training": 0,
        "heldout_variant_rows_in_training": 0, "behavior_labels_used": 0,
        "training_examples_content_sha256": digest(examples), "training_edges_content_sha256": digest(edge_rows),
        "split_claim": "retrospective source holdout; all V2 evidence was previously inspected",
        "confirmation_status": "not_evaluated_on_new_confirmation_data",
        "consistency_scope": "A-B and C-D only; never A-C or B-D",
        "proxy": "JS of normalized complete legal-answer sequence likelihoods, not grammar-decoder probabilities",
        "task_weighting": "equal probability across changed_fact, stable_fact, goal; uniform edges within task",
        "correct_sets": "probability mass of all legal correct answers; no arbitrary tie-breaking",
    }
    audit["split_content_sha256"] = digest({
        "heldout_source": heldout_source, "variants": sorted(EXCLUDED_VARIANTS),
        "examples": audit["training_examples_content_sha256"],
        "edges": audit["training_edges_content_sha256"],
    })
    return examples, edge_rows, audit


def save_preparation(out, examples, edges, audit):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Existing artifacts are immutable: accidental changed inputs require a new directory.
    previous = out / "data_audit.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        require(old["split_content_sha256"] == audit["split_content_sha256"],
                "Preparation changed; use a new output directory")
    write_jsonl(out / "training_examples.jsonl", examples)
    write_jsonl(out / "training_edges.jsonl", edges)
    audit["prepared_file_hashes"] = {
        "training_examples.jsonl": sha256(out / "training_examples.jsonl"),
        "training_edges.jsonl": sha256(out / "training_edges.jsonl"),
    }
    write_json(out / "data_audit.json", audit)


def answer_projection_spec(labels):
    """Pure-CPU alignment for a batch of ONE: keep predicting the entire suffix.

    For prompt length P and answer length K, full logits positions P-1..P+K-2
    predict answer tokens P..P+K-1. Requesting the final K+1 logits includes
    these K positions plus the unused last next-token prediction.
    """
    active = [i for i, label in enumerate(labels) if label != -100]
    require(active and active[0] > 0, "Answer suffix requires a nonempty prompt")
    require(active == list(range(active[0], len(labels))), "Labels must be a contiguous answer suffix")
    return len(active) + 1, labels[active[0]:]


def tokenize_examples(examples, tokenizer, max_length):
    """Token audit without loading model weights or initializing CUDA."""
    if tokenizer.pad_token_id is None:
        require(tokenizer.eos_token_id is not None, "Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    cache = {}
    audit = {"max_sequence_length": 0, "max_prompt_tokens": 0, "max_answer_tokens": 0,
             "max_projected_logits_positions": 0, "max_length_budget": max_length}
    for row in examples:
        prefix = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
        for action in row["candidates"]:
            response = json.dumps({row["field"]: action}, ensure_ascii=False, separators=(",", ":"))
            full = tokenizer.apply_chat_template(
                row["messages"] + [{"role": "assistant", "content": response}],
                tokenize=True, add_generation_prompt=False, enable_thinking=False,
            )
            require(full[:len(prefix)] == prefix, "Chat-template prefix/assistant suffix mismatch")
            require(0 < len(prefix) < len(full), "Missing assistant answer suffix")
            require(len(full) <= max_length,
                    f"Input {row['prompt_id']} has {len(full)} tokens > {max_length}; truncation forbidden")
            labels = [-100] * len(prefix) + full[len(prefix):]
            keep, suffix = answer_projection_spec(labels)
            cache[(row["prompt_id"], canonical(action))] = (full, labels)
            audit["max_sequence_length"] = max(audit["max_sequence_length"], len(full))
            audit["max_prompt_tokens"] = max(audit["max_prompt_tokens"], len(prefix))
            audit["max_answer_tokens"] = max(audit["max_answer_tokens"], len(suffix))
            audit["max_projected_logits_positions"] = max(audit["max_projected_logits_positions"], keep)
    audit.update(status="passed_no_truncation", cached_candidates=len(cache), enable_thinking=False,
                 labels="assistant JSON and end-of-turn suffix only",
                 model_weights_loaded_by_tokenizer_audit=False, cuda_calls_by_tokenizer_audit=False)
    return cache, audit


def train(args, examples, edges, audit, output):
    # Deliberately lazy: --prepare-only works without a GPU or ML dependencies.
    import torch
    import torch.nn.functional as F
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "Training requires one CUDA device; preparation has already passed")
    require(torch.cuda.is_bf16_supported(), "BF16 support required for this fixed pilot")
    require(args.model is not None, "--model is required for actual training")
    require(args.max_steps <= 150 and args.max_steps > 0, "Pilot budget is 1-150 optimizer steps")
    output = Path(output)
    require(not (output / "training_complete.json").exists(), "Training output already complete")
    require(not (output / "training_started.json").exists(),
            "Previous training already started; use a new directory rather than silently restart")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    example_map = {r["prompt_id"]: r for r in examples}
    # Full CPU tokenizer audit of ALL legal answers before any model enters GPU memory.
    token_cache, token_audit = tokenize_examples(examples, tokenizer, args.max_length)
    write_json(output / "tokenization_audit.json", token_audit)
    # Batch size one is fixed. This explicit padding collator also masks future variable lengths.
    def collate(items):
        width = max(len(item[0]) for item in items)
        ids = torch.full((len(items), width), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(items), width), -100, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, (tokens, targets) in enumerate(items):
            ids[i, :len(tokens)] = torch.tensor(tokens)
            labels[i, :len(targets)] = torch.tensor(targets)
            mask[i, :len(tokens)] = 1
        return {"input_ids": ids.cuda(), "attention_mask": mask.cuda(), "labels": labels.cuda()}

    config = {
        "mode": args.mode, "model": str(Path(args.model).resolve()), "rank": 8,
        "alpha": 16, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.0,
        "learning_rate": 1e-4, "batch_size": 1, "gradient_accumulation": 8,
        "max_optimizer_steps": args.max_steps, "seed": args.seed, "bf16": True,
        "gradient_checkpointing": True, "max_length": args.max_length,
        "consistency_weight": args.consistency_weight if args.mode == "sft_consistency" else 0,
        "split_content_sha256": audit["split_content_sha256"], "input_sha256": audit["input_sha256"],
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "peft_version": peft.__version__, "proxy": audit["proxy"],
        "supervised_loss": "mean over two prompts: negative log complete correct-set probability / suffix length",
        "free_behavior_supervision": False, "confirmation_status": audit["confirmation_status"],
        "lm_head_projection": "only answer_len+1 suffix positions via logits_to_keep",
        "checkpoint_interval_optimizer_steps": 25,
        "resume_policy": "adapter checkpoints are inspection/evaluation only; optimizer/RNG not saved; no auto-resume",
    }
    model_config_path = Path(args.model) / "config.json"
    if model_config_path.is_file():
        config["base_model_config_sha256"] = sha256(model_config_path)
    config["chat_template_sha256"] = digest(tokenizer.chat_template)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    require("logits_to_keep" in inspect.signature(model.forward).parameters,
            "This pilot requires native logits_to_keep support; full-prompt LM-head fallback is forbidden")
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    require({"q_proj", "v_proj"} <= available, "Expected Qwen3 attention projections missing")
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(trainable and all("lora_" in name for name, _ in trainable), "Non-adapter parameters are trainable")
    config["trainable_parameters"] = sum(p.numel() for _, p in trainable)
    config["trainable_module_names"] = [name for name, _ in trainable]
    model.cuda().train()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=0.01)
    write_json(output / "training_started.json", {"started_unix": time.time(), **config})
    write_json(output / "training_config.json", config)
    rng = random.Random(args.seed)
    groups = {kind: [edge for edge in edges if edge["task_type"] == kind]
              for kind in sorted(SUPERVISED_TYPES)}
    require(all(groups.values()), "Empty supervised task group")

    def log_probability(row, action):
        tokens, labels = token_cache[(row["prompt_id"], canonical(action))]
        keep, suffix = answer_projection_spec(labels)
        batch = collate([(tokens, labels)])
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, logits_to_keep=keep)
        require(outputs.logits.shape[1] == keep, "Model ignored suffix-only projection")
        # The final returned position predicts AFTER the answer, so omit it.
        scores = outputs.logits[0, :-1, :].float()
        targets = torch.tensor(suffix, dtype=torch.long, device=scores.device)
        require(scores.shape[0] == targets.shape[0], "Suffix target/logit alignment mismatch")
        logp = -F.cross_entropy(scores, targets, reduction="sum")
        return logp, len(suffix)

    def distribution_and_supervision(row, need_distribution):
        targets = set(row["expected"])
        candidates = row["candidates"] if need_distribution else row["expected"]
        logps, counts = zip(*(log_probability(row, action) for action in candidates))
        stack = torch.stack(logps)
        correct_idx = [i for i, action in enumerate(candidates) if action in targets]
        # Sum over ALL optimal actions, not arbitrary tie-breaking or a uniform preference within ties.
        loss = -torch.logsumexp(stack[correct_idx], dim=0) / (sum(counts[i] for i in correct_idx) / len(correct_idx))
        if need_distribution:
            # A/B display positions may swap. Match semantic answers, never label letters.
            order = sorted(range(len(candidates)), key=lambda i: canonical(row["candidate_semantics"][i]))
            distribution = F.log_softmax(stack[order], dim=0)
        else:
            distribution = None
        return distribution, loss

    # A small custom Dataset supplies immutable equivalence edges; sampling is task-balanced.
    class EdgeDataset(torch.utils.data.Dataset):
        def __init__(self, values):
            self.values = values
        def __len__(self):
            return len(self.values)
        def __getitem__(self, index):
            return self.values[index]

    dataset = EdgeDataset(edges)
    require(len(dataset) == len(edges), "Dataset mismatch")
    need_distribution = args.mode == "sft_consistency"
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    last_checkpoint = None

    def save_adapter_checkpoint(step, record):
        target = output / f"adapter_checkpoint_{step:03d}"
        temporary = target.with_name(target.name + ".tmp")
        require(not target.exists() and not temporary.exists(), "Checkpoint directory already exists")
        model.save_pretrained(temporary, safe_serialization=True)
        files = {p.name: sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        write_json(temporary / "checkpoint_metadata.json", {
            "completed_optimizer_steps": step, "training_config_sha256": sha256(output / "training_config.json"),
            "split_content_sha256": audit["split_content_sha256"], "adapter_files": files,
            "resumable": False, "optimizer_saved": False, "rng_saved": False,
            "purpose": "inspection and evaluation only; restarting requires a new run directory",
            "record": record,
        })
        temporary.replace(target)
        return target.name

    write_json(output / "progress.json", {
        "status": "training", "completed_optimizer_steps": 0, "planned_optimizer_steps": args.max_steps,
        "last_adapter_checkpoint": None, "resumable": False,
    })
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.max_steps + 1):
            metrics = collections.defaultdict(float)
            for micro in range(8):
                task = rng.choice(sorted(groups))
                edge = rng.choice(groups[task])
                a, b = example_map[edge["left"]], example_map[edge["right"]]
                require(a["source_id"] == b["source_id"] != args.heldout_source, "Runtime source violation")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pa, la = distribution_and_supervision(a, need_distribution)
                    pb, lb = distribution_and_supervision(b, need_distribution)
                    supervised = (la + lb) / 2
                    js = torch.zeros((), device="cuda")
                    if need_distribution:
                        # logsumexp is stable even when some complete answer likelihoods are tiny.
                        mixture = torch.logsumexp(torch.stack([pa, pb]), dim=0) - math.log(2)
                        js = 0.5 * ((pa.exp() * (pa - mixture)).sum() + (pb.exp() * (pb - mixture)).sum())
                    loss = supervised + args.consistency_weight * js
                require(bool(torch.isfinite(loss).item()), "Nonfinite training loss")
                (loss / 8).backward()
                metrics["supervised_loss"] += float(supervised.detach()) / 8
                metrics["js_proxy"] += float(js.detach()) / 8
                metrics["total_loss"] += float(loss.detach()) / 8
            grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            require(bool(torch.isfinite(grad_norm).item()), "Nonfinite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"step": step, **metrics, "grad_norm": float(grad_norm),
                      "elapsed_seconds": time.time() - started,
                      "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated()}
            log.write(canonical(record) + "\n")
            log.flush()
            if step % 25 == 0 or step == args.max_steps:
                last_checkpoint = save_adapter_checkpoint(step, record)
            write_json(output / "progress.json", {
                "status": "training", "completed_optimizer_steps": step,
                "planned_optimizer_steps": args.max_steps, "last_adapter_checkpoint": last_checkpoint,
                "resumable": False, "record": record,
            })
            if step == 1 or step % 10 == 0:
                print(canonical(record), flush=True)
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(output / "tokenizer")
    write_json(output / "training_complete.json", {
        "status": "pilot_training_complete_not_yet_evaluated", "steps": args.max_steps,
        "elapsed_seconds": time.time() - started, "config_sha256": sha256(output / "training_config.json"),
        "split_content_sha256": audit["split_content_sha256"],
        "adapter_files": {p.name: sha256(p) for p in sorted(adapter.iterdir()) if p.is_file()},
        "claim": "No claim of reliability improvement or new confirmation-test success",
    })
    write_json(output / "progress.json", {
        "status": "pilot_training_complete_not_yet_evaluated", "completed_optimizer_steps": args.max_steps,
        "planned_optimizer_steps": args.max_steps, "last_adapter_checkpoint": last_checkpoint,
        "final_adapter": "adapter", "resumable": False, "elapsed_seconds": time.time() - started,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, default=HERE.parent / "baseline_v2")
    ap.add_argument("--heldout-source", default="HPVBG")
    ap.add_argument("--mode", choices=("sft", "sft_consistency"), default="sft")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--model", help="Existing local Qwen3-8B model directory; never downloads weights")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--tokenizer-audit", action="store_true",
                    help="With --prepare-only and --model, audit actual token lengths on CPU; no model weights/CUDA")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--consistency-weight", type=float, default=1.0)
    args = ap.parse_args()
    require(args.consistency_weight >= 0, "Consistency weight must be nonnegative")
    output = args.output or HERE / "learning_pilot" / f"{args.heldout_source}_{args.mode}_seed{args.seed}"
    examples, edges, audit = prepare(args.baseline, args.heldout_source)
    save_preparation(output, examples, edges, audit)
    print(canonical({k: audit[k] for k in (
        "status", "heldout_source", "unique_supervised_training_inputs", "unique_equivalence_edges",
        "training_inputs_by_type", "training_inputs_by_source", "split_content_sha256",
    )}), flush=True)
    if args.prepare_only and args.tokenizer_audit:
        require(args.model is not None, "Tokenizer audit requires --model")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        _, token_audit = tokenize_examples(examples, tokenizer, args.max_length)
        write_json(output / "tokenization_audit.json", token_audit)
        print(canonical(token_audit), flush=True)
    elif not args.prepare_only:
        try:
            train(args, examples, edges, audit, output)
        except Exception as error:
            progress_path = Path(output) / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
            progress.update(status="failed_no_automatic_resume", error_type=type(error).__name__,
                            error=str(error), resumable=False)
            write_json(progress_path, progress)
            raise


if __name__ == "__main__":
    main()
