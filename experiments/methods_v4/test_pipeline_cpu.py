"""Scientific provenance checks for source projection and the full frozen plan."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v4_projection',R/'score_transfer.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Pipeline(unittest.TestCase):
    def test_matrix_and_source_partition(self):
        p=json.loads((R/'stage_manifest.json').read_text(encoding='utf-8'))
        train=[j for j in p['jobs'] if j['kind']=='train_and_evaluate']
        got={(j['model'],j['scope'],j['training_seed'],j['method']) for j in train}
        want={(model,source,seed,arm) for model in p['models'] for source in ['HPVBG','IGVOFO','PGG','PIQ572']
              for seed in p['training_seeds'] for arm in p['arms']}
        self.assertEqual(got,want);self.assertEqual(len(train),48)
        self.assertEqual(sum(j['calls'] for j in p['jobs']),17014784)
        full=json.loads((R/'manifests/all92.json').read_text(encoding='utf-8'))
        seen=set()
        for source in ['HPVBG','IGVOFO','PGG','PIQ572']:
            child=json.loads((R/'manifests'/f'{source}.json').read_text(encoding='utf-8'))
            ids=set(child['selected_prompt_ids'])
            self.assertFalse(seen & ids);seen |= ids
            self.assertTrue(all(v==[source] for v in child['provenance_sources'].values()))
        self.assertEqual(seen,set(full['selected_prompt_ids']))

    def test_exact_projection_is_not_new_generation(self):
        with tempfile.TemporaryDirectory(prefix='invar_v4_projection_test_') as tmp:
            root=Path(tmp);original=m.R;m.R=root
            try:
                parent=root/'parent'/'model';parent.mkdir(parents=True)
                rows=[dict(prompt_id=p,replicate=i,config_sha256='cfg') for p in ['a','b'] for i in range(2)]
                raw=parent/'baseline.jsonl'
                raw.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                m.atomic(raw.with_suffix('.complete.json'),dict(responses_sha256=m.sha(raw),config_sha256='cfg',rows=4))
                m.atomic(raw.with_suffix('.run_identity.json'),dict(config={'generation':'parent'},plan={'selected_prompt_ids':['a','b']}))
                manifest=dict(selected_prompt_ids=['b'],scope='test',calls_per_model_method=2)
                dest=root/'fold1'
                m.baseline_view(root/'parent',dest,manifest,'model')
                got=[json.loads(s) for s in (dest/'model/baseline.jsonl').read_text().splitlines()]
                self.assertEqual(got,rows[2:])
                record=json.loads((dest/'model/baseline.projection.json').read_text())
                self.assertEqual(record['parent_raw_sha256'],m.sha(raw))
                self.assertIn('not_new_generation',record['kind'])
                plan=json.loads((dest/'model/baseline.run_identity.json').read_text())['plan']
                self.assertEqual(plan['unique_inputs'],1)
                stored=plan.pop('content_sha256')
                import hashlib
                self.assertEqual(stored,hashlib.sha256(json.dumps(plan,ensure_ascii=False,sort_keys=True,
                                                               separators=(',',':')).encode()).hexdigest())
                m.baseline_view(root/'parent',root/'fold2',manifest,'model')
                self.assertEqual(m.sha(root/'fold2/model/baseline.jsonl'),m.sha(dest/'model/baseline.jsonl'))
                raw.write_text(raw.read_text()+'{}\n')
                with self.assertRaises(AssertionError):m.baseline_view(root/'parent',root/'fold3',manifest,'model')
            finally:
                m.R=original


if __name__=='__main__':unittest.main()
