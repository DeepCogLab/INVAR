"""Check missing-response bounds against explicit completed empirical samples."""
import itertools
import random
import unittest
from score_joint_reasoning_v2 import metrics,w1,w1_check


class MissingBoundsTest(unittest.TestCase):
    def test_wasserstein_independent_quantile_and_cdf(self):
        rng=random.Random(20261111)
        for _ in range(100):
            a=[rng.randrange(5)/4 for i in range(rng.randrange(1,20))]
            b=[rng.randrange(5)/4 for i in range(rng.randrange(1,20))]
            self.assertAlmostEqual(w1(a,b),w1_check(a,b),places=11)

    def test_all_missing_completion_values_inside_bounds(self):
        q=dict(A='a',B='b',C='c',D='d')
        partial=dict(a=[0.],b=[.5],c=[1.],d=[0.])
        bounds=metrics(q,partial,2)
        for completion in itertools.product([0.,.25,.5,.75,1.],repeat=4):
            full={pid:partial[pid]+[x] for pid,x in zip('abcd',completion)}
            actual=metrics(q,full,2)
            for label in ['E','G','R']:
                self.assertLessEqual(bounds[label+'_low'],actual[label+'_conditional']+1e-11)
                self.assertGreaterEqual(bounds[label+'_high']+1e-11,actual[label+'_conditional'])

    def test_shared_reference_cancels_missing_mean(self):
        q=dict(A='a',B='b',C='a',D='b')
        r=metrics(q,dict(a=[],b=[]),2)
        self.assertEqual((r['G_low'],r['G_high']),(0,0))
        self.assertEqual((r['R_low'],r['R_high']),(0,0))


if __name__=='__main__':unittest.main()
