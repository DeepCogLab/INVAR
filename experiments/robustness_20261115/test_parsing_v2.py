import unittest
from parsing import parse_final as old
from parsing_v2 import parse_final as new

SCHEMA=dict(properties=dict(action=dict(type='integer',enum=[0,2,4])))


def response(payload='{"action":2}',metadata='json'):
    return '<|channel|>analysis<|message|>Calculate carefully.<|end|><|start|>assistant<|channel|>final <|constrain|>'+metadata+'<|message|>'+payload+'<|return|>'


class HarmonyHeaderTests(unittest.TestCase):
    def parse(self,s,finish='stop'):return new('',s,finish,SCHEMA,'native_effort')

    def test_real_failure_reproduced_and_format_metadata_accepted(self):
        s=response();self.assertEqual(old('',s,'stop',SCHEMA,'native_effort')[1],'missing_or_ambiguous_final_channel')
        self.assertEqual(self.parse(s)[0],{'action':2})
        self.assertEqual(self.parse(response(metadata='JSON'))[0],{'action':2})

    def test_analysis_answer_never_replaces_final_answer(self):
        s=response('{"action":4}').replace('Calculate carefully.','The answer could be {"action":2}.')
        self.assertEqual(self.parse(s)[0],{'action':4})
        self.assertIsNone(self.parse('<|channel|>analysis <|constrain|>json<|message|>{"action":2}<|return|>')[0])

    def test_repeated_finals_or_unsupported_metadata_remain_invalid(self):
        self.assertIsNone(self.parse(response()+response())[0])
        self.assertIsNone(self.parse(response(metadata='xml'))[0])
        self.assertIsNone(self.parse(response().replace('assistant<|channel|>','assistant to=tool<|channel|>'))[0])

    def test_existing_schema_type_and_truncation_rules_unchanged(self):
        self.assertEqual(self.parse(response('{"action":"2"}'))[1],'type')
        self.assertEqual(self.parse(response('{"action":3}'))[1],'menu')
        self.assertEqual(self.parse(response(),finish='length')[1],'truncated')
        self.assertIsNone(self.parse(response('{"action":0,"action":2}'))[0])

    def test_metadata_in_initial_implicit_assistant_header(self):
        s='<|channel|>final<|constrain|>json<|message|>{"action":2}<|return|>'
        self.assertEqual(self.parse(s)[0],{'action':2})


if __name__=='__main__':unittest.main()
