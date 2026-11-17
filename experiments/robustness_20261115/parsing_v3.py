"""CPU scoring v3: one candidate per JSON-starting line; includes Harmony v2.

Copied from the frozen v1 parser. Only horizontal whitespace is consumed at
a JSON-starting line, so blank lines cannot duplicate one terminal object.
Schema, truncation, duplicate-key and final-channel rules remain unchanged.
"""
import json
import re
from parsing_v2 import canonical_harmony_final


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def parse_final(text, special, finish_reason, schema, kind, arm=None, prefilled_think=False):
    if finish_reason == 'length':
        return None, 'truncated', False
    s = text.strip()
    if kind == 'native_effort':
        special = canonical_harmony_final(special)
        marker = '<|channel|>final<|message|>'
        if special.count(marker) != 1:
            return None, 'missing_or_ambiguous_final_channel', False
        s = special.split(marker, 1)[1].strip()
        for token in ('<|return|>', '<|end|>', '<|fim_suffix|>', '<|im_end|>'):
            if s.endswith(token): s = s[:-len(token)].strip()
    # Inspect the actual rendered prompt rather than assuming that all native
    # templates prefill an opening tag. Current Qwen3 templates do not prefill
    # native-on; they let the model emit the opening tag itself.
    if prefilled_think and s.count('</think>')!=1:
        return None,'unclosed_prefilled_reasoning',False
    if '<think>' in s or '</think>' in s:
        if s.count('</think>') != 1 or s.count('<think>') > 1:
            return None, 'incomplete_or_ambiguous_reasoning', False
        s = s.split('</think>', 1)[1].strip()
    if s.startswith('<Answer>') and s.endswith('</Answer>'):
        s = s[len('<Answer>'):-len('</Answer>')].strip()
    original = s
    fence = re.search(r'```(?:json)?\s*(\{[^`]*\})\s*```\s*$', s, re.DOTALL | re.IGNORECASE)
    if fence: s = fence.group(1).strip()
    candidates=[]
    # A suffix JSON is accepted in both arms under exactly the same rule. It
    # must start on a line of its own and consume the remainder of the output.
    # Earlier candidate answers cannot be picked for correctness.
    for match in re.finditer(r'(?m)^[^\S\r\n]*(?=\{)', s):
        tail=s[match.end():]
        try:
            obj,end=json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(tail)
        except (ValueError,TypeError):continue
        if not tail[end:].strip():candidates.append((obj,tail[:end]))
    if len(candidates)!=1:return None,'format',False
    obj,representation=candidates[0]
    if not isinstance(obj,dict) or set(obj)!=set(schema['properties']):return None,'schema',False
    for key,spec in schema['properties'].items():
        expected=int if spec['type']=='integer' else str
        if type(obj[key]) is not expected:return None,'type',False
        if 'enum' in spec and obj[key] not in spec['enum']:return None,'menu',False
        if 'minimum' in spec and obj[key]<spec['minimum']:return None,'menu',False
        if 'maximum' in spec and obj[key]>spec['maximum']:return None,'menu',False
    return obj,None,original.strip()==representation.strip()
