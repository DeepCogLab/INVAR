"""Standard-library preparation/cache tests; no torch/model/GPU imports."""
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("condition_anchor", ROOT / "train_condition_anchor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConditionAnchorCPU(unittest.TestCase):
    def test_actual_preparation_has_no_behavior_truth_or_source_overlap(self):
        args = SimpleNamespace(
            baseline=Path(os.environ.get("INVAR_BASELINE", ROOT.parent / "baseline_v2")),
            legacy_helper_dir=Path(os.environ.get("INVAR_METHODS_V1", ROOT.parent / "methods_v1")),
            heldout_source="HPVBG", max_free_candidates=16, max_steps=150, seed=20261026,
        )
        prepared = module.prepare(args)
        _, supervised, _, free, edges, conditions, schedule, audit = prepared
        self.assertEqual((len(supervised), len(free), len(edges), len(conditions)), (5010, 774, 659, 115))
        self.assertTrue(all(r["source_id"] != "HPVBG" and len(r["candidates"]) <= 16 for r in free))
        self.assertTrue(all("expected" not in r and "expected_actions" not in r for r in free))
        self.assertTrue(all(ref["sides"] in ("AB", "CD") and ref["variant_id"] not in module.EXCLUDED_VARIANTS
                            for edge in edges for ref in edge["references"]))
        self.assertEqual(len(schedule), 150 * 8)
        self.assertEqual(module.digest(schedule), audit["exposure_schedule_sha256"])
        rebuilt = module.build_schedule(prepared[2], edges, conditions, 150, 20261026)
        self.assertEqual(schedule, rebuilt)

    def test_cache_requires_complete_manifest_and_checks_arithmetic_teacher(self):
        with tempfile.TemporaryDirectory(prefix="invar_anchor_cpu_") as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            module.atomic_json(model / "config.json", {"model_type": "qwen3"})
            args = SimpleNamespace(teacher_cache=root / "teacher", model=model, heldout_source="HPVBG")
            args.teacher_cache.mkdir()
            free = [{"prompt_id": pid, "source_id": "train", "candidates": [0, 1],
                     "candidate_semantics": [0, 1], "messages_sha256": "message_" + pid} for pid in ("A", "B")]
            condition = {"condition_id": "C", "source_id": "train", "prompt_ids": ["A", "B"], "candidates": [0, 1]}
            audit = {"input_sha256": "inputs", "split_content_sha256": "split"}
            prepared = (None, [], [], free, [], [condition], [], audit)
            with self.assertRaises(ValueError):
                module.load_teacher(args, prepared)
            prompt_rows = []
            for row, probs in zip(free, ((0.2, 0.8), (0.6, 0.4))):
                logs = [math.log(p) for p in probs]
                prompt_rows.append({**row, "complete_candidate_logps": logs,
                                    "suffix_token_lengths": [5, 5], "logq": logs})
            condition_rows = [{**condition, "teacher_logq": [math.log(0.4), math.log(0.6)]}]
            prompt_path = args.teacher_cache / "teacher_prompt_distributions.jsonl"
            condition_path = args.teacher_cache / "teacher_condition_distributions.jsonl"
            module.write_rows(prompt_path, prompt_rows)
            module.write_rows(condition_path, condition_rows)
            config = {"identity": {"base_model_path": str(model.resolve()),
                                  "base_model_config_sha256": module.sha_file(model / "config.json")}}
            manifest = {"status": "complete", "config": config, "config_sha256": module.digest(config),
                        **audit, "prompt_rows": 2,
                        "files_sha256": {p.name: module.sha_file(p) for p in (prompt_path, condition_path)}}
            manifest_path = args.teacher_cache / "teacher_manifest.json"
            module.atomic_json(manifest_path, manifest)
            teacher, _, _ = module.load_teacher(args, prepared)
            self.assertEqual(set(teacher), {"C"})
            lock = args.teacher_cache / "BUILD.lock"
            lock.write_text("incomplete", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.load_teacher(args, prepared)
            lock.unlink()
            # Even refreshing file hashes cannot hide a mathematically wrong pool.
            condition_rows[0]["teacher_logq"] = [math.log(0.5), math.log(0.5)]
            module.write_rows(condition_path, condition_rows)
            manifest["files_sha256"][condition_path.name] = module.sha_file(condition_path)
            module.atomic_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "arithmetic"):
                module.load_teacher(args, prepared)


if __name__ == "__main__":
    unittest.main()
