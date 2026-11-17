"""Meaningful boundary checks before consuming remote GPU time."""
import unittest
from run_model import valid,render,seed
class ProtocolTests(unittest.TestCase):
    def test_no_wrong_type_or_truncation_acceptance(self):
        s={'properties':{'action':{'type':'integer','minimum':0,'maximum':10}}}
        self.assertTrue(valid('{"action":3}',s,'stop'))
        for text,finish in [('{"action":true}','stop'),('{"action":"3"}','stop'),('{"action":11}','stop'),('{"action":3,"answer":"A"}','stop'),('{"action":3}','length')]:
            self.assertFalse(valid(text,s,finish))
    def test_prompt_boundary_and_base_rendering(self):
        messages=[dict(role='system',content='RULE'),dict(role='user',content='SCENARIO')]
        self.assertEqual(render(None,messages,'plain'),'RULE\n\nSCENARIO\n\n')
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.messages=messages;return 'chat'
        t=Tokenizer();render(t,messages,'mistral')
        self.assertEqual(t.messages,[dict(role='user',content='RULE\n\nSCENARIO')])
    def test_common_random_seeds_and_independent_replicates(self):
        self.assertEqual(seed('p',0,'main'),seed('p',0,'main'))
        self.assertNotEqual(seed('p',0,'main'),seed('p',1,'main'))
        self.assertNotEqual(seed('p',0,'main'),seed('p',0,'repeat'))
    def test_native_author_wrapper_is_not_a_content_failure(self):
        from analyze import native_final
        s={'properties':{'answer':{'type':'string','enum':['A','B']}}}
        self.assertEqual(native_final('<think>reason</think><Answer>{"answer":"A"}</Answer>',s,'stop'),'{"answer":"A"}')
        self.assertIsNone(native_final('<think>{"answer":"A"}</think>',s,'stop'))
        self.assertIsNone(native_final('<Answer>{"answer":"A"}</Answer>',s,'length'))
if __name__=='__main__':unittest.main()
