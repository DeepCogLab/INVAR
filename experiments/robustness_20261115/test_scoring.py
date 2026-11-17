"""Synthetic end-to-end audit: denominators, reference reuse and missingness."""
import collections
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import score as scorer
from campaign_common import R, jl, save, sha, digest, messages_for, MODELS
from science_worker import plan


class ScoringTests(unittest.TestCase):
    def test_complete_fixed_answer_and_all_missing_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copytree(R/'frozen',root/'frozen')
            protocol=dict(repeats=dict(behavior=2,goal=2,changed_fact=2,stable_fact=2,parameter=2))
            save(root/'freeze.json',dict(protocol=protocol,files={}))
            job='qwen3_8b';ps=jl(root/'frozen/model_inputs.jsonl');labels={p['prompt_id']:p for p in jl(root/'frozen/evaluation.jsonl')}
            trials=plan(job,ps,protocol);pmap={p['prompt_id']:p for p in ps};d=root/'runs'/job;d.mkdir(parents=True)
            identity=dict(freeze_sha256=sha(root/'freeze.json'),job=job,trial_plan_sha256=digest(trials),expected_rows=len(trials),
                rendered_sha256={p['prompt_id']+'__'+arm:'synthetic' for p in ps for arm in ['native_off','native_on']},
                prefilled_think={p['prompt_id']+'__'+arm:False for p in ps for arm in ['native_off','native_on']},
                prompt_token_ids_sha256={p['prompt_id']+'__'+arm:'synthetic_ids' for p in ps for arm in ['native_off','native_on']});save(d/'identity.json',identity)
            identity_hash=digest(identity)
            with (d/'responses.jsonl').open('w',encoding='utf-8') as f:
                for t in trials:
                    p=pmap[t['prompt_id']];label=labels[t['prompt_id']]
                    if p['task_type']=='behavior':text='unfinished';finish='length'
                    else:
                        obj=dict(action=label['expected_actions'][0]) if p['task_type']=='goal' else dict(answer=label['expected_answer'])
                        text=('</think>\n' if t['arm']=='native_on' else '')+json.dumps(obj);finish='stop'
                    row=dict(**t,model=job,identity_sha256=identity_hash,
                        messages_sha256=digest(messages_for(p,MODELS[job]['kind'],t['arm'])),
                        rendered_sha256='synthetic',prompt_token_ids_sha256='synthetic_ids',completion_tokens=1,token_ids=[0],response_text=text,response_with_special_tokens=text,finish_reason=finish)
                    f.write(json.dumps(row)+'\n')
            save(d/'COMPLETE.json',dict(raw_sha256=sha(d/'responses.jsonl'),rows=len(trials)))
            old=scorer.R;scorer.R=root
            try:result=scorer.score(job)
            finally:scorer.R=old
            self.assertEqual(result['rows'],len(trials))
            for s in result['summaries']:
                self.assertIsNone(s['behavior_pair_equal']['E_conditional'])
                self.assertEqual(s['behavior_pair_equal']['E_low'],0)
                self.assertEqual(s['behavior_pair_equal']['E_high'],1)
                self.assertEqual(s['goal_pair_equal_all_attempt_Q'],1)
                self.assertEqual(s['changed_fact_pair_equal_all_attempt_U'],1)
                self.assertEqual(s['stable_fact_pair_equal_all_attempt_U'],1)
                self.assertEqual(s['parameter_disjoint_joint_all_four_correct'],1)


if __name__=='__main__':unittest.main()
