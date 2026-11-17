"""Score every long-budget attempt with the independent native parser."""
import argparse
import collections
import csv
import json
from audit_new_runs import R,BASE,jl,load,sha,save,digest,parse_final


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    d=R/'runs'/a.job/'native_long';cfg=load(R/'native_long_freeze.json')
    ps={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')}
    ids=load(R/'frozen/native_sample.json')['prompt_ids'];expected={(pid,i) for pid in ids for i in range(8)}
    identity=load(d/'identity.json');complete=load(d/'COMPLETE.json');raw=d/'responses.jsonl'
    assert sha(raw)==complete['sha256']
    short={(r['prompt_id'],r['replicate']):r for r in jl(R/'runs'/a.job/'native/responses.jsonl') if r['batch']=='native_thinking'}
    seen=set();counts=collections.Counter();rows=[]
    for r in jl(raw):
        key=(r['prompt_id'],r['replicate']);p=ps[r['prompt_id']]
        assert key in expected and key not in seen;seen.add(key)
        assert r['config_sha256']==digest(identity) and r['messages_sha256']==digest(p['messages'])
        assert r['sampling_seed']==short[key]['sampling_seed']
        obj,error=parse_final(r['response_text'],r['finish_reason'],p['output_schema'])
        good=obj is not None and obj['action'] in p['expected_actions']
        counts['n']+=1;counts['valid']+=obj is not None;counts['correct']+=good
        counts['truncated']+=r['finish_reason']=='length';counts['matching_prefix']+=r['same_text_prefix_as_short']
        rows.append(dict(prompt_id=p['prompt_id'],replicate=r['replicate'],task=p['task'],policy=p['policy'],valid=obj is not None,
            correct=good,truncated=r['finish_reason']=='length',matching_prefix=r['same_text_prefix_as_short']))
    assert seen==expected and len(seen)==1280
    result=dict(status='passed_independent_scalar',job=a.job,stage='native_long',**counts,
        Q_all_attempts=counts['correct']/1280,Q_given_parse=counts['correct']/counts['valid'] if counts['valid'] else None,
        raw_sha256=complete['sha256'],scorer_sha256=sha(__import__('pathlib').Path(__file__)),
        scope='Same 160 prompts, all eight attempts; fixed 4096 token budget; context/scheduling change, so prefix matching is checked rather than assumed.')
    save(d/'SCORING_COMPLETE.json',result);save(d/'independent_scalar_audit.json',result)
    with (d/'attempt_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result))


if __name__=='__main__':main()
