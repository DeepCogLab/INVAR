"""CPU analytic checks; importing the runner does not import torch or load a model."""
import json
import unittest

import numpy as np

import activation_probe as m


def item(pid, typ='behavior', target='L', fmt='list'):
    body = '- a' if fmt == 'list' else '| a | b |'
    return dict(prompt_id=pid, task_type=typ,
                messages=[dict(role='user', content='【个体设定】\n' + body + '\n【实验规则】\n- P1和P2参与\n【当前情况】\n已有规则')],
                output_schema=dict(properties={'action': dict(type='string', enum=['L', 'R'])}),
                outcome=dict(kind='probability', target_action=target, denominator=1),
                expected_actions=['L'], expected_answer='L')


class TestActivationProxy(unittest.TestCase):
    def test_candidates_use_only_legal_schema(self):
        p = item('a')
        p['expected_actions'] = ['SECRET']
        self.assertEqual(m.two_candidates(p), ('action', ['L', 'R']))
        p['output_schema']['properties']['action']['enum'] = list(range(12000))
        self.assertIsNone(m.two_candidates(p))
        p['output_schema']['properties']['action'] = dict(type='integer', minimum=2, maximum=3)
        self.assertEqual(m.two_candidates(p)[1], [2, 3])

    def test_complete_token_lcp(self):
        self.assertEqual(m.longest_common_prefix([[1, 2, 3, 4], [1, 2, 5, 6]]), 2)

    def test_style_is_message_observable_and_ignores_truth_metadata(self):
        a, b = item('a'), item('b', fmt='table')
        self.assertNotEqual(m.observed_style(a['messages']), m.observed_style(b['messages']))
        before = m.observed_style(a['messages'])
        a.update(variant_id='fake', pair_id='heldout', expected_answer='SECRET')
        self.assertEqual(before, m.observed_style(a['messages']))

    def test_shared_prompt_source_closure(self):
        pairs = {'a': {'source_id': 'S1'}, 'b': {'source_id': 'S2'}, 'c': {'source_id': 'S3'}}
        qs = [dict(pair_id='a', A='shared', B='a1', C='a2', D='a3'),
              dict(pair_id='b', A='shared', B='b1', C='b2', D='b3'),
              dict(pair_id='c', A='c0', B='c1', C='c2', D='c3')]
        sources = {pid: set() for q in qs for pid in (q[t] for t in 'ABCD')}
        for q in qs:
            for tag in 'ABCD':
                sources[q[tag]].add(pairs[q['pair_id']]['source_id'])
        manifest = m.build_folds(pairs, qs, sources, sources)
        s1 = next(f for f in manifest['folds'] if f['heldout_source'] == 'S1')
        self.assertEqual(s1['train_pair_ids'], ['c'])
        self.assertFalse(set(s1['train_prompt_ids']) & set(s1['test_prompt_ids']))

    def test_uncentered_svd_identifies_known_nuisance(self):
        x = np.array([[4., 0, 0], [3., 0, 0], [-2., 0, 0]], dtype=np.float32)
        b = m.right_subspace(x, 4)
        self.assertEqual(b.shape[1], 1)
        np.testing.assert_allclose(b @ b.T, np.diag([1., 0, 0]), atol=1e-5)

    def test_condition_protection_removes_overlap_before_svd(self):
        # Format difference=(1,1), true condition difference=(0,1).
        features = np.array([[0, 0], [1, 1], [0, 1], [1, 2]], dtype=np.float32)
        index = {pid: i for i, pid in enumerate('ABCD')}
        qs = [dict(A='A', B='B', C='C', D='D')]
        _, basis, protected, _ = m.fit_basis(list('ABCD'), qs, features, index, True)
        np.testing.assert_allclose(protected @ protected.T, np.diag([0., 1.]), atol=1e-5)
        np.testing.assert_allclose(basis @ basis.T, np.diag([1., 0.]), atol=1e-5)
        np.testing.assert_allclose(protected.T @ basis, 0., atol=1e-5)

    def test_shared_template_calibration_preserves_real_condition_shift(self):
        ps = {pid: item(pid, fmt='table' if pid in 'BD' else 'list') for pid in 'ABCD'}
        # Scores: true condition adds +2 log odds; format adds +1 log odds.
        scores = {'A': np.array([0., 0.]), 'B': np.array([1., 0.]),
                  'C': np.array([2., 0.]), 'D': np.array([3., 0.])}
        qs = [dict(A='A', B='B', C='C', D='D', variant_id='unused')]
        offsets = m.fit_calibration(list('ABCD'), qs, ps, scores)
        odds = {}
        for pid in 'ABCD':
            bias = offsets[(m.calibration_key(ps[pid]), m.observed_style(ps[pid]['messages']))]
            adjusted = scores[pid] - bias  # analytic full-strength check
            odds[pid] = adjusted[0] - adjusted[1]
        self.assertAlmostEqual(odds['A'], odds['B'])
        self.assertAlmostEqual(odds['C'], odds['D'])
        self.assertAlmostEqual(odds['C'] - odds['A'], 2.)

    def test_reversed_fact_option_edges_are_not_fitted(self):
        ps = {pid: item(pid, 'changed_fact', fmt='table' if pid in 'BD' else 'list') for pid in 'ABCD'}
        for pid in 'AC':
            ps[pid]['option_semantic_order'] = [0, 1]
        for pid in 'BD':
            ps[pid]['option_semantic_order'] = [1, 0]
        qs = [dict(A='A', B='B', C='C', D='D', variant_id='metadata_not_used_in_inference')]
        scores = {pid: np.array([i, -i], dtype=float) for i, pid in enumerate('ABCD')}
        biases, audit = m.fit_calibration(list('ABCD'), qs, ps, scores, return_audit=True)
        self.assertEqual(biases, {})
        self.assertEqual(audit['excluded_fact_option_order_equivalence_edges'], 2)
        features = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 2.]], dtype=np.float32)
        _, nuisance, protected, fit = m.fit_basis(list('ABCD'), qs, features,
                         {pid: i for i, pid in enumerate('ABCD')}, True, inputs=ps)
        self.assertEqual(nuisance.shape[1], 0)
        self.assertEqual(fit['n_equivalence_edges'], 0)
        self.assertEqual(fit['excluded_fact_option_order_equivalence_edges'], 2)
        self.assertEqual(fit['n_condition_edges'], 2)  # A/C and B/D still have matched option orders.

    def test_same_order_facts_fit_but_changed_order_conditions_are_excluded(self):
        ps = {pid: item(pid, 'changed_fact', fmt='table' if pid in 'BD' else 'list') for pid in 'ABCD'}
        for pid in 'AB':
            ps[pid]['option_semantic_order'] = [0, 1]
        for pid in 'CD':
            ps[pid]['option_semantic_order'] = [1, 0]
        qs = [dict(A='A', B='B', C='C', D='D', variant_id='unused')]
        scores = {pid: np.array([i, -i], dtype=float) for i, pid in enumerate('ABCD')}
        biases, audit = m.fit_calibration(list('ABCD'), qs, ps, scores, return_audit=True)
        self.assertEqual(len(biases), 2)
        self.assertEqual(audit['excluded_fact_option_order_equivalence_edges'], 0)
        features = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 2.]], dtype=np.float32)
        _, nuisance, protected, fit = m.fit_basis(list('ABCD'), qs, features,
                         {pid: i for i, pid in enumerate('ABCD')}, True, inputs=ps)
        self.assertEqual(nuisance.shape[1], 1)
        self.assertEqual(protected.shape[1], 0)
        self.assertEqual(fit['n_equivalence_edges'], 2)
        self.assertEqual(fit['excluded_fact_option_order_condition_edges'], 2)

    def test_fact_edge_filter_never_uses_expected_truth(self):
        a, b = item('a', 'changed_fact'), item('b', 'changed_fact')
        a['option_semantic_order'] = b['option_semantic_order'] = [0, 1]
        a['expected_answer'], b['expected_answer'] = 'L', 'SECRET'
        self.assertTrue(m.training_edge_allowed('a', 'b', {'a': a, 'b': b}))

    def test_proxy_metrics_known_distributions(self):
        ps = {pid: item(pid) for pid in 'ABCD'}
        q = dict(A='A', B='B', C='C', D='D', source_id='S', pair_id='p', quartet_id='q',
                 kind='behavior', variant_id='V', task='task')
        probs = {'A': np.array([1., 0.]), 'B': np.array([0., 1.]),
                 'C': np.array([0., 1.]), 'D': np.array([1., 0.])}
        rows, skipped = m.proxy_quartets([q], probs, ps, 'baseline', 'f', 'S')
        self.assertEqual(skipped, {})
        self.assertEqual(rows[0]['E'], 1.)
        self.assertEqual(rows[0]['R'], 1.)
        self.assertEqual(rows[0]['G'], 2.)
        self.assertEqual(rows[0]['sampled_trials'], 0)

    def test_accuracy_is_probability_mass_not_draws(self):
        ps = {pid: item(pid, 'goal') for pid in 'ABCD'}
        q = dict(A='A', B='B', C='C', D='D', source_id='S', pair_id='p', quartet_id='q',
                 kind='goal', variant_id='V', task='task')
        probs = {pid: np.array([.8, .2]) for pid in 'ABCD'}
        rows, _ = m.proxy_quartets([q], probs, ps, 'baseline', 'f', 'S')
        self.assertAlmostEqual(rows[0]['Q'], .8)
        self.assertEqual(rows[0]['argmax_accuracy'], 1.)


if __name__ == '__main__':
    unittest.main()
