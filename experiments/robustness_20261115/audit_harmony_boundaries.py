"""Verify dataset-specific stream prerequisites for the bounded v2 correction."""
import collections
import json
from pathlib import Path
from campaign_common import R,load,sha,save,jl
from parsing import parse_final as old
from parsing_v2 import parse_final as new

M=Path('artifacts/robustness_20261115/remote')
D=Path('artifacts/robustness_20261115/corrected_v2/runs/gpt_oss_20b')


def main():
    totals=collections.Counter();counts=collections.defaultdict(collections.Counter)
    for row in jl(M/'runs/gpt_oss_20b/responses.jsonl'):
        chunks=row['response_with_special_tokens'].split('<|start|>');finals=[];arm=row['arm']
        for i,chunk in enumerate(chunks):
            head,sep,body=chunk.partition('<|message|>')
            if not sep:continue
            author,sep,channel=head.partition('<|channel|>')
            if not sep:continue
            parts=channel.split('<|constrain|>')
            if parts[0].strip()=='final':finals.append((i,author,body,parts[1:]));counts[arm]['format_'+(parts[1].strip() if len(parts)==2 else 'none')]+=1
        counts[arm]['final_count_'+str(len(finals))]+=1;counts[arm]['length']+=row['finish_reason']=='length'
        totals['rows']+=1;totals['multiple_final']+=len(finals)>1
        totals['tool_call_marker']+='<|call|>' in row['response_with_special_tokens']
        totals['nonlength_without_unique_final']+=row['finish_reason']!='length' and len(finals)!=1
        for i,author,body,metadata in finals:
            totals['final_messages']+=1;totals['final_followed_by_message']+=i!=len(chunks)-1
            totals['unexpected_recipient_or_author']+=author.strip() not in ('','assistant')
            totals['nonlength_final_not_return']+=row['finish_reason']!='length' and not body.rstrip().endswith('<|return|>')
            totals['unsupported_format_metadata']+=len(metadata)>1 or bool(metadata and metadata[0].strip().casefold()!='json')
    assert totals['rows']==11760 and all(totals[k]==0 for k in ['multiple_final','tool_call_marker','nonlength_without_unique_final','final_followed_by_message','unexpected_recipient_or_author','nonlength_final_not_return','unsupported_format_metadata'])
    # Exhaustive no-effect verification on the already completed non-Harmony model.
    ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')};identity=load(M/'runs/qwen3_8b/identity.json');unchanged=0
    for row in jl(M/'runs/qwen3_8b/responses.jsonl'):
        key=row['prompt_id']+'__'+row['arm']
        args=(row['response_text'],row['response_with_special_tokens'],row['finish_reason'],ps[row['prompt_id']]['output_schema'],'native_toggle',row['arm'],identity['prefilled_think'][key])
        assert old(*args)==new(*args);unchanged+=1
    assert unchanged==11760
    result=dict(status='passed_all_actual_stream_boundary_prerequisites',totals=dict(totals),by_arm={a:dict(v) for a,v in counts.items()},
        non_harmony_qwen_rows_unchanged=unchanged,raw_sha256=sha(M/'runs/gpt_oss_20b/responses.jsonl'),auditor_sha256=sha(Path(__file__)),
        limitation='Checks every actual stream. The bounded parser is not presented as a general-purpose Harmony parser for arbitrary tool/recipient/multiple-message inputs.')
    save(D/'STREAM_BOUNDARY_AUDIT.json',result);print(json.dumps(result))


if __name__=='__main__':main()
