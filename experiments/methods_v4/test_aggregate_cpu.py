"""Meaningful checks of reporting weights, no-goal handling and baseline reuse."""
import importlib.util
import unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('aggregate_under_test',Path(__file__).with_name('aggregate_transfer.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Aggregate(unittest.TestCase):
    def folds(self):
        result={}
        for source,n,value in [('HPVBG',8,.1),('IGVOFO',8,.2),('PGG',22,.3),('PIQ572',54,.4)]:
            rows=[]
            for method in m.METHODS:
                rows.append(dict(source=source,method=method,kind='behavior',metric='E',estimate=value,weight=n))
                if source!='PGG':
                    rows.append(dict(source=source,method=method,kind='goal',metric='accuracy',estimate=value,weight=2))
            result[source]=dict(values=rows,verified=['anchor_sft','anchor_js'])
        return result

    def test_unequal_pair_weights_and_macro_and_missing_Q(self):
        rows=m.summarize_seed(self.folds(),'model',1)
        lookup={(r['method'],r['kind'],r['aggregation']):r for r in rows}
        self.assertAlmostEqual(lookup[('anchor_js','behavior','pair_weighted')]['estimate'],(8*.1+8*.2+22*.3+54*.4)/92)
        self.assertAlmostEqual(lookup[('anchor_js','behavior','source_macro')]['estimate'],.25)
        goal=lookup[('anchor_js','goal','source_macro')]
        self.assertEqual(goal['eligible_sources'],3)
        self.assertAlmostEqual(goal['estimate'],(.1+.2+.4)/3)

    def test_incomplete_source_or_arm_rejects_grand_mean(self):
        folds=self.folds();folds.pop('PGG');self.assertEqual(m.summarize_seed(folds,'model',1),[])
        folds=self.folds();folds['PGG']['verified']=['anchor_sft']
        self.assertEqual(m.summarize_seed(folds,'model',1),[])

    def test_baseline_is_one_reference_and_seed_SD_is_descriptive(self):
        rows=[]
        for seed in [1,2,3]:
            current=m.summarize_seed(self.folds(),'model',seed)
            for row in current:
                if row['method']=='anchor_js':row['estimate']+=seed*.01
            rows.extend(current)
        stats=m.seed_statistics(rows,[1,2,3]);base=[r for r in stats if r['method']=='baseline']
        self.assertTrue(all(r['n_training_seeds']==0 and r['distinct_fitted_models']==1 and r['sd'] is None for r in base))
        js=[r for r in stats if r['method']=='anchor_js']
        self.assertTrue(all(r['n_training_seeds']==3 and r['distinct_fitted_models']==12 and abs(r['sd']-.01)<1e-12 for r in js))
        delta=m.differences(rows)
        self.assertEqual(len(delta),3*2*2*2)
        next(r for r in rows if r['method']=='baseline')['estimate']+=.1
        with self.assertRaises(AssertionError):m.seed_statistics(rows,[1,2,3])


if __name__=='__main__':unittest.main()
