"""CPU-only transfer preparation/template checks; no ML dependencies."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transfer_training", ROOT / "train_transfer.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def arguments(source="HPVBG", seed=20261026, model="qwen3_8b", arm="anchor_sft"):
    return SimpleNamespace(
        baseline=Path(os.environ.get("INVAR_BASELINE", ROOT.parent / "baseline_v2")),
        legacy_helper_dir=Path(os.environ.get("INVAR_METHODS_V1", ROOT.parent / "methods_v1")),
        v2_helper_dir=Path(os.environ.get("INVAR_METHODS_V2", ROOT.parent / "methods_v2")),
        model_id=model, heldout_source=source, seed=seed, arm=arm, max_steps=150, max_free_candidates=16,
    )


class TransferCPU(unittest.TestCase):
    def test_four_sources_complete_menus_and_no_teacher(self):
        for source in training.SOURCES:
            with self.subTest(source=source):
                v2, prepared = training.prepare(arguments(source=source))
                _, sup, _, free, edges, conditions, schedule, audit = prepared
                self.assertTrue(sup and free and conditions)
                self.assertTrue(all(row["source_id"] != source for row in sup + free))
                self.assertTrue(all(len(row["candidates"]) in (2, 7, 11) for row in free))
                self.assertTrue(all("expected" not in row for row in free))
                self.assertTrue(all(ref["sides"] in ("AB", "CD") and ref["variant_id"] not in v2.EXCLUDED_VARIANTS
                                    for edge in edges for ref in edge["references"]))
                self.assertEqual(len(schedule), 1200)
                self.assertFalse(audit["teacher_required"])
                self.assertIsNone(audit["teacher_manifest_sha256"])

    def test_same_exposure_across_model_and_arm_new_seed_changes_only_schedule(self):
        _, first = training.prepare(arguments())
        _, second = training.prepare(arguments(model="qwen25_14b_instruct", arm="anchor_js"))
        self.assertEqual(first[6], second[6])
        self.assertEqual(first[-1]["source_pool_sha256"], second[-1]["source_pool_sha256"])
        for seed in (20261028, 20261029):
            _, changed = training.prepare(arguments(seed=seed))
            self.assertEqual(first[-1]["source_pool_sha256"], changed[-1]["source_pool_sha256"])
            self.assertNotEqual(first[-1]["exposure_schedule_sha256"], changed[-1]["exposure_schedule_sha256"])

    def test_native_template_flags_and_suffix_mask(self):
        v2 = training.load_v2(arguments().v2_helper_dir)
        legacy = v2.load_legacy(arguments().legacy_helper_dir)
        row = {"prompt_id": "p", "messages": [{"role": "user", "content": "原始题目"}],
               "field": "action", "candidates": [0, 10]}
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1
            def __init__(self, expected):
                self.expected = expected
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
                if kwargs != self.expected:
                    raise AssertionError("Wrong native model template kwargs")
                prefix = [2, 3, 4]
                return prefix if add_generation_prompt else prefix + [ord(c) + 10 for c in messages[-1]["content"]] + [1]
        for model, kwargs in (("qwen3_8b", {"enable_thinking": False}), ("qwen25_14b_instruct", {})):
            cache, audit = training.tokenize_rows(v2, legacy, [row], Tokenizer(kwargs), model, 4096)
            self.assertEqual(len(cache), 2)
            self.assertEqual(audit["template_kwargs"], kwargs)
            self.assertTrue(all(targets[:3] == [-100] * 3 for _, targets in cache.values()))
            with self.assertRaisesRegex(ValueError, "no truncation"):
                training.tokenize_rows(v2, legacy, [row], Tokenizer(kwargs), model, 3)
        self.assertFalse(any(name in sys.modules for name in ("torch", "transformers", "peft")))


if __name__ == "__main__":
    unittest.main()
