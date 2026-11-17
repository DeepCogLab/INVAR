import json
import shutil
import tempfile
import unittest
from pathlib import Path
import supp_common as c
import supp_score as scorer


class SupplementTests(unittest.TestCase):
    def test_full_plans_and_factorial_support(self):
        ps=c.jl(c.S/'frozen/model_inputs.jsonl');total=0
        for job,spec in c.JOBS.items():
            plan=c.make_plan(job,ps);total+=len(plan)
            self.assertEqual(len(plan),spec['expected_rows'])
            self.assertEqual(len({(t['prompt_id'],c.cell(t),t['replicate']) for t in plan}),len(plan))
        self.assertEqual(total,21888)

    def synthetic(self,job,all_truncated=False):
        original=scorer.S;guard=scorer.verify_freeze
        root=c.S/'engineering_tests';root.mkdir(exist_ok=True)
        tmp=tempfile.TemporaryDirectory(prefix='synthetic_',dir=root)
        base=Path(tmp.name).resolve();self.assertTrue(base.is_relative_to(root.resolve()))
        try:
            shutil.copytree(c.S/'frozen',base/'frozen');shutil.copytree(c.S/'upstream',base/'upstream');(base/'plans').mkdir()
            c.save(base/'freeze.json',{})
            ps={p['prompt_id']:p for p in c.jl(base/'frozen/model_inputs.jsonl')};labels={p['prompt_id']:p for p in c.jl(base/'frozen/evaluation.jsonl')}
            plan=c.make_plan(job,list(ps.values()))
            (base/'plans'/(job+'.jsonl')).write_text(''.join(json.dumps(t)+'\n' for t in plan),encoding='utf-8')
            d=base/'runs'/job;d.mkdir(parents=True)
            identity=dict(job=job,model=c.JOBS[job]['model'],freeze_sha256=c.sha(base/'freeze.json'),trial_plan_sha256=c.digest(plan),expected_rows=len(plan),
                rendered_sha256={},prompt_token_ids_sha256={},messages_sha256={},prefilled_think={})
            for t in plan:
                k=c.input_key(t);identity['rendered_sha256'][k]=c.digest(['synthetic',k]);identity['prompt_token_ids_sha256'][k]=c.digest([1]);identity['messages_sha256'][k]=c.digest(c.messages(ps[t['prompt_id']],t,job));identity['prefilled_think'][k]=False
            c.save(d/'identity.json',identity);ih=c.digest(identity)
            with (d/'responses.jsonl').open('w',encoding='utf-8') as f:
                for i,t in enumerate(plan):
                    p=ps[t['prompt_id']];lab=labels[t['prompt_id']];key=c.input_key(t)
                    action=lab['expected_actions'][0] if p['task_type']=='goal' else p['output_schema']['properties']['action'].get('enum',[p['output_schema']['properties']['action'].get('minimum',0)])[0]
                    text=json.dumps(dict(action=action))
                    row=dict(**t,job=job,model=c.JOBS[job]['model'],stage=p['stage'],task_type=p['task_type'],identity_sha256=ih,
                        rendered_sha256=identity['rendered_sha256'][key],prompt_token_ids_sha256=c.digest([1]),messages_sha256=identity['messages_sha256'][key],
                        completion_tokens=1,token_ids=[1],response_text=text,response_with_special_tokens=text,
                        finish_reason='length' if all_truncated or i==0 else 'stop')
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
            c.save(d/'COMPLETE.json',dict(rows=len(plan),raw_sha256=c.sha(d/'responses.jsonl'),identity_sha256=ih))
            scorer.S=base;scorer.verify_freeze=lambda:{}
            result=scorer.score(job)
            attempts=c.jl(d/'responses.jsonl')
            if all_truncated:
                for summary in result['summaries']:
                    self.assertNotIn('core_Q_pair_equal',summary)
                    self.assertIsNone(summary['behavior_pair_E_conditional'])
                    self.assertEqual(summary['core_behavior_delivery'],0)
                    self.assertGreater(summary['behavior_pair_E_high'],0)
            else:
                self.assertEqual(sum(s['core_goal_attempts']+s['parameter_goal_attempts'] for s in result['summaries']),len(plan))
                self.assertTrue(any(s['core_Q_pair_equal']<1 or s['parameter_Q_pair_equal']<1 for s in result['summaries']))
                self.assertTrue(all('behavior_pair_E_conditional' not in s for s in result['summaries']))
        finally:
            scorer.S=original;scorer.verify_freeze=guard;tmp.cleanup()

    def test_pure_goal_retains_failure_and_complete_factorial(self):self.synthetic('language_glm')
    def test_pure_behavior_is_not_perfect_stability_or_fake_Q(self):self.synthetic('length_behavior_qwen',True)


if __name__=='__main__':unittest.main()
