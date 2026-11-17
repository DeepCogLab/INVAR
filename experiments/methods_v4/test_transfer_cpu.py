"""CPU-only transfer runner tests; never import a model runtime or start CUDA."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transfer_runner_under_test", HERE / "run_transfer_evaluation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TransferTests(unittest.TestCase):
    def test_full_embedded_boundary_resume_and_budget_suite(self):
        runner.self_test()

    def test_independent_training_and_inference_seeds(self):
        baseline = runner.seed_for("qwen3_8b", "baseline", "unadapted", "p", 0, 20261030)
        streams = {baseline,
                   runner.seed_for("qwen3_8b", "anchor_js", "train-seed26-adapter", "p", 0, 20261030),
                   runner.seed_for("qwen3_8b", "anchor_js", "train-seed28-adapter", "p", 0, 20261030),
                   runner.seed_for("qwen3_8b", "anchor_sft", "train-seed26-adapter", "p", 0, 20261030),
                   runner.seed_for("qwen25_14b_instruct", "baseline", "unadapted", "p", 0, 20261030),
                   runner.seed_for("qwen3_8b", "baseline", "unadapted", "p", 0, 20261031)}
        self.assertEqual(len(streams), 6)

    def test_full_8666_input_subset_and_budget(self):
        baseline = Path(os.environ.get("INVAR_BASELINE", HERE.parent / "baseline_v2")).resolve()
        if not (baseline / "inputs.jsonl").is_file():
            self.skipTest("Set INVAR_BASELINE to run actual-corpus CPU validation")
        rows = runner.read_rows(baseline / "inputs.jsonl")
        reps = {"behavior": 256, "goal": 256, "changed_fact": 64, "stable_fact": 64, "role": 64}
        manifest = {"baseline_inputs_sha256": runner.sha(baseline / "inputs.jsonl"),
                    "selected_prompt_ids": [row["prompt_id"] for row in rows], "repetitions": reps}
        with tempfile.TemporaryDirectory(prefix="invar_transfer_manifest_cpu_") as temp:
            path = Path(temp) / "manifest.json"
            runner.atomic_json(path, manifest)
            selected, _, plan = runner.load_plan(baseline, path)
        self.assertEqual(len(selected), len(rows))
        self.assertEqual(plan["planned_rows"], sum(reps[row["task_type"]] for row in rows))
        self.assertEqual(len(plan["provenance_sources"]), len(rows))
        sentinel = "PRIVATE_METADATA_MUST_NOT_BE_IN_MODEL_847"
        for row in rows:
            poisoned = dict(row, expected_answer=sentinel, expected_actions=[sentinel],
                            pair_id=sentinel, variant_id=sentinel, prompt_id=sentinel)
            self.assertEqual(runner.current_messages(row), runner.current_messages(poisoned))
            self.assertNotIn(sentinel, json.dumps(runner.current_messages(poisoned), ensure_ascii=False))

    def test_no_model_imports(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "transformers", "vllm")))


if __name__ == "__main__":
    unittest.main()
