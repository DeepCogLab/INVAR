import unittest
from parsing import parse_final as frozen
from parsing_v3 import parse_final

S={'properties':{'action':{'type':'integer','enum':[0,2,4]}}}


class TerminalLineTests(unittest.TestCase):
    def parse(self,text,finish='stop'):
        return parse_final(text,text,finish,S,'prompt_reasoning')

    def test_blank_line_reproduces_defect_and_has_one_candidate(self):
        text='Reasoning.\n\n{"action":2}'
        self.assertEqual(frozen(text,text,'stop',S,'prompt_reasoning')[1],'format')
        for sep in ['\n','\n\n','\n\n\n','\r\n\r\n','\n \t\n']:
            self.assertEqual(self.parse('Reasoning.'+sep+'  {"action":2}')[0],{'action':2})

    def test_no_relaxation_of_terminal_schema_or_failure_rules(self):
        for text in ['Reasoning. {"action":2}','{"action":2}\ntrailing prose',
                     '{"action":2,"action":4}','{"action":"2"}',
                     '{"action":3}','<think>\n{"action":2}']:
            self.assertIsNone(self.parse(text)[0],text)
        self.assertEqual(self.parse('Reasoning.\n\n{"action":2}','length')[1],'truncated')

    def test_final_object_is_not_chosen_by_correctness(self):
        self.assertEqual(self.parse('{"action":2}\n\nRevision.\n\n{"action":4}')[0],{'action':4})
        self.assertIsNone(self.parse('{"action":2}\n\n{"action":4} trailing')[0])

    def test_multiline_and_fence(self):
        self.assertEqual(self.parse('Reasoning.\n\n{\n "action":2\n}')[0],{'action':2})
        self.assertEqual(self.parse('Reasoning.\n\n```json\n{"action":2}\n```')[0],{'action':2})

    def test_harmony_revision_preserved(self):
        s='<|channel|>analysis<|message|>{"action":0}<|end|><|start|>assistant<|channel|>final <|constrain|>JSON<|message|>Answer:\n\n{"action":4}<|return|>'
        self.assertEqual(parse_final('',s,'stop',S,'native_effort')[0],{'action':4})


if __name__=='__main__':unittest.main()
