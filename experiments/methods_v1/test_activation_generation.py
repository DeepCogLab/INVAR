"""CPU simulations of candidate trie, prefill position, paired seeds, and real-row scoring."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import activation_probe as proxy
import run_activation_generation as gen


class GenerationTests(unittest.TestCase):
    def test_multi_token_candidates_and_eos_only_at_complete_leaf(self):
        trie = gen.CandidateTrie([[10, 11, 12], [20, 21]], 99, 0)
        self.assertEqual(trie.allowed([]), [10, 20])
        self.assertEqual(trie.allowed([10]), [11])
        self.assertEqual(trie.allowed([10, 11]), [12])
        self.assertEqual(trie.allowed([10, 11, 12]), [99])
        self.assertEqual(trie.allowed([20, 21, 99, 0, 0]), [99])
        self.assertTrue(trie.validate([10, 11, 12, 99])[0])
        self.assertTrue(trie.validate([20, 21, 99, 0])[0])
        self.assertFalse(trie.validate([10, 99])[0])
        self.assertFalse(trie.validate([10, 11, 12])[0])
        self.assertFalse(trie.validate([20, 21, 99, 777])[0])
        with self.assertRaises(ValueError):
            trie.allowed([10, 99])

    def test_internal_complete_candidate_can_stop_or_continue(self):
        trie = gen.CandidateTrie([[4, 5], [4, 5, 6]], 99)
        self.assertEqual(trie.allowed([4, 5]), [6, 99])

    def test_no_illegal_suffix_can_escape_trie(self):
        trie = gen.CandidateTrie([[1, 2, 3], [1, 4, 5]], 99)
        frontier = [[]]
        completed = []
        for _ in range(4):
            future = []
            for prefix in frontier:
                for token in trie.allowed(prefix):
                    seq = prefix + [token]
                    if token == 99:
                        self.assertTrue(trie.validate(seq)[0])
                        completed.append(seq)
                    else:
                        future.append(seq)
            frontier = future
        self.assertEqual(sorted(completed), [[1, 2, 3, 99], [1, 4, 5, 99]])
        with self.assertRaises(ValueError):
            trie.allowed([1, 8])

    def test_prefill_exactly_once_decode_never_edited(self):
        self.assertTrue(gen.should_edit_prefill(False, 200, 200))
        self.assertFalse(gen.should_edit_prefill(True, 1, 200))
        self.assertFalse(gen.should_edit_prefill(True, 1, 200))
        with self.assertRaises(ValueError):
            gen.should_edit_prefill(False, 1, 200)
        with self.assertRaises(ValueError):
            gen.should_edit_prefill(True, 200, 200)

    def test_chunk_seeds_paired_but_trials_distinct_by_method(self):
        a = gen.chunk_seed('qwen3_8b', 'fold', 'pid', 0)
        self.assertEqual(a, gen.chunk_seed('qwen3_8b', 'fold', 'pid', 0))
        self.assertNotEqual(a, gen.chunk_seed('qwen3_8b', 'fold', 'pid', 4))
        self.assertNotEqual(gen.generation_trial_id('m', 'f', 'p', 'baseline', 0),
                            gen.generation_trial_id('m', 'f', 'p', 'paired_svd', 0))

    def test_real_sample_frequency_scoring_not_sequence_scores(self):
        ps = {pid: dict(prompt_id=pid, task_type='behavior',
                       output_schema=dict(properties={'action': dict(type='string', enum=['L', 'R'])}),
                       outcome=dict(kind='probability', target_action='L', denominator=1)) for pid in 'ABCD'}
        q = dict(A='A', B='B', C='C', D='D', source_id='S', pair_id='p', quartet_id='q',
                 kind='behavior', variant_id='V', task='task')
        fold = dict(fold='f', heldout_source='S', test_prompt_ids=list('ABCD'))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paths = {}
            for method in gen.METHODS:
                rows = []
                for pid in 'ABCD':
                    for rep in range(gen.REPLICATES):
                        choice = 0 if pid in 'AD' else 1
                        rows.append(dict(prompt_id=pid, replicate=rep, candidate_index=choice,
                                         status='ok', finish_reason='stop'))
                path = root / (method + '.jsonl')
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                paths[('f', method)] = path
            result = gen.score_actual_generations(paths, ps, [q], [fold], root)
            self.assertEqual(len(result), 3)
            for row in result:
                self.assertEqual(row['E'], 1.)
                self.assertEqual(row['G'], 2.)
                self.assertEqual(row['sampled_trials_per_prompt'], 32)
                self.assertIn('actual', row['interface'])
            summary = json.loads((root / 'generation_summary.json').read_text())
            self.assertTrue(all('true HF' in row['interpretation'] for row in summary))


if __name__ == '__main__':
    unittest.main()
