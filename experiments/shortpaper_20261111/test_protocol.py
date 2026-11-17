"""Tests cover parsing and prohibited label exposure, not numerical paper claims."""
import json
import unittest
from run_api import R,jl,load,tasks
from score_api import native_final


class ProtocolTest(unittest.TestCase):
    def test_final_after_reasoning(self):
        schema={'properties':{'answer':{'type':'string','enum':['A','B']}}}
        for text in ['<think>分析中出现 {"answer":"A"}</think>\n{"answer":"B"}',
                     '<think>分析</think>\n```json\n{"answer":"B"}\n```',
                     '<think>分析</think><Answer>{"answer":"B"}</Answer>']:
            self.assertEqual(json.loads(native_final(text,schema,'stop')),{'answer':'B'})
        for text,finish in [('<think>{"answer":"B"}','stop'),
                            ('{"answer":"A"}{"answer":"B"}','stop'),
                            ('{"answer":"B"}','length')]:
            self.assertIsNone(native_final(text,schema,finish))

    def test_no_labels_in_generation_payload(self):
        for p in jl(R/'frozen/model_inputs.jsonl'):
            self.assertEqual(set(p),{'prompt_id','task_type','messages','output_schema'})
        for p in jl(R/'frozen/goal_diagnostics.jsonl'):
            self.assertEqual(set(p),{'prompt_id','original','payoff_table','output_schema'})
            before=p['original'][-1]['content'].split('【本次明确目标】')[1]
            after=p['payoff_table'][-1]['content'].split('【本次明确目标】')[1]
            self.assertEqual(before,after)

    def test_full_budget_and_unique_trials(self):
        rows=tasks('main',load(R/'freeze.json'))
        self.assertEqual(sum(r['n'] for r in rows),421504)
        keys=[(r['prompt_id'],r['batch'],i) for r in rows for i in range(r['start'],r['start']+r['n'])]
        self.assertEqual(len(keys),len(set(keys)))
        goals=tasks('goal_diagnostic',load(R/'freeze.json'))
        self.assertEqual(sum(r['n'] for r in goals),90240)


if __name__=='__main__':
    unittest.main()
