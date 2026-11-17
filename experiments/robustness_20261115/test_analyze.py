import unittest
import numpy as np
from analyze import boot_metrics,audit_vectorization


class VectorAuditTests(unittest.TestCase):
    def test_shared_ids_and_missing_values_agree_with_independent_scalar(self):
        rng=np.random.default_rng(20261115)
        values={str(i):rng.choice([0.,.2,.7,1.,np.nan],size=(3,8)) for i in range(4)}
        values['missing']=np.full((3,8),np.nan)
        qs=[dict(A='0',B='1',C='2',D='3'),dict(A='0',B='0',C='1',D='2'),
            dict(A='missing',B='missing',C='2',D='3'),dict(A='missing',B='missing',C='missing',D='missing')]
        boots=[boot_metrics(q,values) for q in qs]
        audit_vectorization(qs,values,boots)


if __name__=='__main__':unittest.main()
