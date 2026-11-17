"""Guard against reasoning-answer leakage and inconsistent delivery denominators."""
import unittest
from parsing import parse_final

SCHEMA=dict(properties=dict(action=dict(type='integer',enum=[0,2,4])),required=['action'],additionalProperties=False)


class ParsingTests(unittest.TestCase):
    def parse(self,text,finish='stop',kind='native_toggle',special=''):
        return parse_final(text,special,finish,SCHEMA,kind)

    def test_complete_reasoning_final_only(self):
        self.assertEqual(self.parse('We might choose {"action": 0}.\n</think>\n{"action": 2}')[0],{'action':2})

    def test_cannot_salvage_a_correct_thought(self):
        self.assertIsNone(self.parse('<think>Correct answer: {"action": 2}')[0])
        self.assertIsNone(self.parse('<think>{"action": 2}</think>')[0])

    def test_length_is_failure_even_with_json(self):
        self.assertEqual(self.parse('{"action": 2}',finish='length')[1],'truncated')

    def test_type_menu_and_boolean_remain_distinct(self):
        self.assertEqual(self.parse('{"action": "2"}')[1],'type')
        self.assertEqual(self.parse('{"action": true}')[1],'type')
        self.assertEqual(self.parse('{"action": 3}')[1],'menu')

    def test_only_terminal_json_can_count(self):
        self.assertIsNone(self.parse('Consider {"action": 2}. No final answer.')[0])
        self.assertEqual(self.parse('Calculation.\n{"action": 4}')[0],{'action':4})
        self.assertIsNone(self.parse('{"action": 2}\nExtra trailing text.')[0])

    def test_native_effort_requires_final_channel(self):
        self.assertIsNone(self.parse('{"action": 2}',kind='native_effort',special='analysis {"action": 2}')[0])
        special='a thought<|end|><|start|>assistant<|channel|>final<|message|>{"action": 4}<|return|>'
        self.assertEqual(self.parse('',kind='native_effort',special=special)[0],{'action':4})

    def test_same_parser_accepts_final_fence_without_answer_selection(self):
        self.assertEqual(self.parse('Reason.\n```json\n{"action": 0}\n```')[0],{'action':0})

    def test_prefilled_native_on_cannot_end_inside_reasoning(self):
        self.assertIsNone(parse_final('Still calculating.\n{"action": 2}','','stop',SCHEMA,'native_toggle','native_on',prefilled_think=True)[0])
        self.assertEqual(parse_final('Calculated.\n</think>\n{"action": 2}','','stop',SCHEMA,'native_toggle','native_on',prefilled_think=True)[0],{'action':2})

    def test_native_on_without_prefill_may_deliver_directly(self):
        self.assertEqual(parse_final('{"action": 2}','','stop',SCHEMA,'native_toggle','native_on',prefilled_think=False)[0],{'action':2})

    def test_conflicting_duplicate_fields_are_invalid(self):
        self.assertIsNone(self.parse('{"action":0,"action":2}')[0])


if __name__=='__main__':unittest.main()
