"""Erratum: Harmony final channel can carry JSON content-type metadata."""
import re
from parsing import parse_final as frozen_parse_final

FINAL_JSON_HEADER=re.compile(
    r'(?P<prefix>\A|<\|start\|>assistant)<\|channel\|>final[ \t]*'
    r'<\|constrain\|>[ \t]*(?i:json)[ \t]*<\|message\|>')


def canonical_harmony_final(special):
    # Match header boundaries, never an answer or JSON inside analysis. Preserve
    # channel identity; only remove supported JSON format metadata from headers.
    return FINAL_JSON_HEADER.sub(lambda m:m['prefix']+'<|channel|>final<|message|>',special)


def parse_final(text,special,finish_reason,schema,kind,arm=None,prefilled_think=False):
    if kind=='native_effort':special=canonical_harmony_final(special)
    return frozen_parse_final(text,special,finish_reason,schema,kind,arm,prefilled_think)
