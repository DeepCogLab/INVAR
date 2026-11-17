"""Describe completed native attempts and token use from verified local raw files."""
import collections
import json
import statistics
from pathlib import Path
from analyze_existing import BASE,OUT,jl,write
from audit_new_runs import sha,parse_final


def main():
    mirror=Path('artifacts/shortpaper_20261111')
    index=json.loads((mirror/'mirror_index.json').read_text(encoding='utf-8'))
    prompts={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')};rows=[];provenance=[];failures=[]
    for stage in index['stages']:
        rel=Path(stage);job,kind=rel.parts[1:]
        if kind not in ['native','native_long']:continue
        local_audit=mirror/'local_scalar_audits'/(job+'_'+kind+'.json')
        if not local_audit.exists():continue
        raw=mirror/rel/'responses.jsonl';raw_sha=sha(raw)
        assert raw_sha==index['files'][rel.as_posix()+'/responses.jsonl']['sha256']
        audit=json.loads(local_audit.read_text(encoding='utf-8'))
        assert audit['status']=='passed_independent_scalar' and audit['raw_sha256']==raw_sha
        groups=collections.defaultdict(list)
        for r in jl(raw):groups[r['batch']].append(r)
        for arm,rs in sorted(groups.items()):
            counts=collections.Counter();tokens=[];requests={}
            for r in rs:
                p=prompts[r['prompt_id']];obj,error=parse_final(r['response_text'],r['finish_reason'],p['output_schema'])
                counts['valid']+=obj is not None
                counts['correct']+=obj is not None and obj['action'] in p['expected_actions']
                if error:counts[error]+=1
                if error or obj['action'] not in p['expected_actions']:
                    final=r['response_text'].split('</think>')[-1].strip() if '</think>' in r['response_text'] else r['response_text'][-200:]
                    failures.append(dict(job=job,stage=kind,arm=arm,prompt_id=p['prompt_id'],replicate=r['replicate'],task=p['task'],policy=p['policy'],
                        error=error or 'valid_wrong_goal',expected_actions=json.dumps(p['expected_actions'],ensure_ascii=False),final_or_unclosed_tail=final))
                if 'completion_tokens' in r:tokens.append(r['completion_tokens'])
                else:
                    # API n=8 usage is a request total repeated on eight rows.
                    # Deduplicate it; per-choice token quantiles are not recorded.
                    key=r['request_id'];usage=r['request_usage']
                    if key in requests:assert requests[key]==usage
                    requests[key]=usage
            assert not (tokens and requests),'Do not mix token accounting units'
            total=sum(tokens) if tokens else sum(u['completion_tokens'] for u in requests.values())
            recorded=next(s for s in audit['summaries'] if s['arm']==arm) if kind=='native' else audit
            assert counts['valid']==recorded['valid'] and counts['correct']==recorded['correct'] and counts['truncated']==recorded['truncated']
            rows.append(dict(job=job,stage=kind,budget=1408 if kind=='native' else 4096,arm=arm,n=len(rs),**counts,
                Q_all_attempts=counts['correct']/len(rs),Q_given_parse=counts['correct']/counts['valid'] if counts['valid'] else None,
                mean_output_tokens=total/len(rs),median_output_tokens=statistics.median(tokens) if tokens else None,p95_output_tokens=sorted(tokens)[int(.95*(len(tokens)-1))] if tokens else None,
                total_output_tokens=total,correct_answers_per_1000_output_tokens=counts['correct']/total*1000,
                token_accounting='Per-choice token IDs' if tokens else 'Unique request usage, deduplicated n=8; per-choice quantiles unavailable'))
        provenance.append(dict(stage=stage,raw_sha256=raw_sha,local_audit_sha256=sha(local_audit)))
    write('native_delivery_and_token_use.csv',rows)
    write('native_failure_catalog.csv',failures)
    (OUT/'native_delivery_audit.json').write_text(json.dumps(dict(status='verified_raw_descriptive_analysis',stages=provenance,
        scope='All attempted completions, including truncations and unparsed outputs. Generated token counts describe output use, not latency, FLOPs, or training cost. Matched model comparisons concern this fixed sample and budget.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified native delivery/token summary:',len(provenance),'stages;',len(rows),'rows')


if __name__=='__main__':main()
