"""Scientific and integration checks; synthetic outputs are calibration only."""
import json,random,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate import parse_response,w1,quartet_metrics,evaluate,read_jsonl
from build_dataset import render,IDMAP

class Metrics(unittest.TestCase):
    def test_distribution_change_with_same_mean(self):
        self.assertAlmostEqual(w1([0,1],[.5,.5]),.5)
    def test_unequal_samples(self):
        self.assertAlmostEqual(w1([0,0,1],[0,1]),1/6)
    def test_constant_agent_not_response_proof(self):
        m=quartet_metrics([.3]*8,[.3]*8,[.3]*8,[.3]*8)
        self.assertEqual(m,{'E':0,'delta_original':0,'delta_variant':0,'G':0})
    def test_stable_nonzero_response(self):
        m=quartet_metrics([0]*8,[0]*8,[1]*8,[1]*8)
        self.assertEqual(m['E'],0);self.assertEqual(m['G'],0)
        self.assertEqual(m['delta_original'],1)
    def test_effect_bound(self):
        rng=random.Random(7)
        for _ in range(40):
            arrays=[[rng.random() for _ in range(rng.randint(3,20))] for i in range(4)]
            m=quartet_metrics(*arrays)
            self.assertLessEqual(m['G'],2*m['E']+1e-12)
    def test_strict_parser(self):
        a={'type':'integer','min':0,'max':10}
        for text in ('{"action":true}','{"action":"3"}','{"action":3.0}','{"action":11}','{"action":3,"reason":"x"}','```json\n{"action":3}\n```'):
            self.assertIsNotNone(parse_response(text,a)[1])
        self.assertEqual(parse_response('{"action":3}',a),(3,None))
    def test_menu_gap(self):
        self.assertIsNotNone(parse_response('{"action":1200}',{'values':[0,1000,2000]})[1])

class BuiltData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=read_jsonl(ROOT/'data'/'cases.jsonl')
        cls.prompts=read_jsonl(ROOT/'data'/'prompts.jsonl')
        cls.quartets=read_jsonl(ROOT/'data'/'quartets.jsonl')
    def test_noop_and_ref_integrity(self):
        ids={p['prompt_id'] for p in self.prompts}
        for q in self.quartets:
            self.assertTrue(all(q[k] in ids for k in 'ABCD'))
            self.assertNotEqual(q['A'],q['B']);self.assertNotEqual(q['C'],q['D'])
    def test_numbering_bijection(self):
        for c in self.cases:
            changed=render(c,'V5_ids')[1]['content']
            for old,new in IDMAP.items():changed=changed.replace(new,old)
            self.assertEqual(changed,render(c)[1]['content'])
    def test_piq_ug_reject_not_zero(self):
        cs=[c for c in self.cases if c['source_id']=='PIQ572' and c.get('semantic_action_map',{}).get('NORTH')=='reject']
        self.assertTrue(cs)
        for c in cs:
            self.assertEqual(c['payoffs']['NORTH']['self'],10)
    def test_pgg_never_invents_exact_peer_sum(self):
        for c in self.cases:
            if c['source_id']!='PGG':continue
            latent=c['payoffs']['latent'];t=c['source_condition']['treatment']
            self.assertTrue(all(x is None for x in latent['actual_contributions'].values()))
            if t in (2,3):self.assertNotIn('admissible_peer_sums_if_this_column_applies',latent)
            if t==5:self.assertTrue(latent['signal_independent_of_actual_contributions'])
    def test_group_punishment_is_per_member(self):
        case=next(c for c in self.cases if c['pair_id']=='HPVBG_TPP_OFFER2_INDIV_GROUP' and c['condition']==1)
        punished=case['payoffs']['punish']
        self.assertEqual(punished['self'],4)
        self.assertEqual([punished[p] for p in ('P1','P4','P5')],[6,6,6])
        self.assertEqual(punished['P2'],2)
    def test_econ_two_multipliers_and_threshold(self):
        fixed=next(c for c in self.cases if c['pair_id']=='IGVOFO_TG_SENDER_FIXED_RETURN' and c['condition']==1)
        self.assertEqual(fixed['payoffs']['by_action']['6000'],{'P1':12000,'P2':6000})
        threshold=next(c for c in self.cases if c['pair_id']=='IGVOFO_DG_THRESHOLD' and c['condition']==1)
        self.assertEqual(threshold['payoffs']['by_action']['1000'],{'P1':0,'P2':0})
        self.assertEqual(threshold['payoffs']['by_action']['2000'],{'P1':4000,'P2':2000})
    def test_end_to_end_invalid_not_dropped(self):
        p=self.prompts[0];sp=p['action_space'];v=sp['values'][0] if 'values' in sp else sp['min']
        rows=[{'model':'synthetic_calibration_only','prompt_id':p['prompt_id'],'replicate':0,'response_text':json.dumps({'action':v})},
              {'model':'synthetic_calibration_only','prompt_id':p['prompt_id'],'replicate':1,'response_text':'not json'}]
        with tempfile.TemporaryDirectory(dir=ROOT/'audit') as temp:
            folder=Path(temp);f=folder/'synthetic.jsonl';f.write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
            report=evaluate(ROOT/'data',f,folder/'out',0)
            self.assertEqual(report['response_rows'],2)
            import csv
            with (folder/'out'/'output_validity.csv').open(encoding='utf-8-sig') as handle:
                valid=list(csv.DictReader(handle))
            row=next(r for r in valid if r['prompt_id']==p['prompt_id'])
            self.assertEqual(row['attempted'],'2');self.assertEqual(row['valid_rate'],'0.5')

if __name__=='__main__':unittest.main(verbosity=2)
