"""CPU regression checks for a final marker arriving during scoring."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from watch_scores import completion_status


class ScoringCompletion(unittest.TestCase):
    def signatures(self):
        pairs = [(m, a) for m in ['qwen3_8b', 'qwen25_14b_instruct', 'glm4_9b']
                 for a in ['baseline', 'canonical_modules', 'canonical_profile']]
        pairs += [('qwen3_8b', a) for a in ['anchor_sft', 'anchor_js', 'anchor_kl', 'anchor_js_kl']]
        return [(f'/screen/{m}/{a}.complete.json', 1) for m, a in pairs]

    def test_late_marker_requires_another_scoring_pass(self):
        latest = self.signatures()
        self.assertIsNone(completion_status({'failures': []}, True, latest, latest[:-1]))

    def test_all_planned_methods_must_have_been_scored(self):
        latest = self.signatures()
        self.assertEqual(completion_status({'failures': []}, True, latest, latest),
                         {'status': 'completed', 'completed_methods': 13})
        with self.assertRaises(AssertionError):
            completion_status({'failures': []}, True, latest[:-1], latest[:-1])

    def test_failed_campaign_is_not_success(self):
        result = completion_status({'failures': ['gpu1_FAILED.json']}, True, [], [])
        self.assertEqual(result['status'], 'partial_failure')
        self.assertIsNone(completion_status(None, True, [], []))


if __name__ == '__main__':
    unittest.main()
