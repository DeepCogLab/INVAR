"""Independent scalar accounting of the original/payoff-table goal diagnostic."""
import argparse
import collections
import json
from pathlib import Path
from run_api import R,load,jl,sha,digest,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    b=Path('/workspace/project_social_simulation/invar_reliability_runs/20261024_v2')
    ps={p['prompt_id']:p for p in jl(b/'inputs.jsonl') if p['task_type']=='goal'}
    qs=jl(b/'goal_quartets.jsonl')
    inputs={p['prompt_id']:p for p in jl(R/'frozen/goal_diagnostics.jsonl')}
    d=R/'runs'/a.job/'goal_diagnostic'
    identity=load(d/'identity.json');ch=digest(identity)
    complete=load(d/'COMPLETE.json')
    assert sha(d/'responses.jsonl')==complete['sha256']
    counts=collections.Counter();correct=collections.Counter();seen=set()
    with (d/'responses.jsonl').open() as f:
        for line in f:
            row=json.loads(line);pid=row['prompt_id'];arm=row['batch'];rep=row['replicate']
            assert pid in ps and arm in ['original','payoff_table'] and 0<=rep<32
            assert (pid,arm,rep) not in seen;seen.add((pid,arm,rep))
            assert row['config_sha256']==ch
            assert row['messages_sha256']==digest(inputs[pid][arm])
            assert row['finish_reason']!='length'
            obj=json.loads(row['response_text'])
            assert set(obj)=={'action'}
            spec=inputs[pid]['output_schema']['properties']['action'];value=obj['action']
            assert type(value) is (int if spec['type']=='integer' else str)
            assert value in spec['enum'] if 'enum' in spec else spec['minimum']<=value<=spec['maximum']
            counts[pid,arm]+=1;correct[pid,arm]+=int(value in ps[pid]['expected_actions'])
    assert len(seen)==90240 and len(counts)==2820 and set(counts.values())=={32}
    slices=[]
    for arm in ['original','payoff_table']:
        for policy in ['all','own','total','equality']:
            qrows=[q for q in qs if policy=='all' or q['policy']==policy]
            bypair=collections.defaultdict(list)
            for q in qrows:
                bypair[q['pair_id']].append(sum(correct[q[t],arm]/32 for t in 'ABCD')/4)
            point=sum(sum(v)/len(v) for v in bypair.values())/len(bypair)
            slices.append(dict(arm=arm,policy=policy,Q=point,n_pairs=len(bypair)))
        recorded=next(x for x in load(d/'SCORING_COMPLETE.json')['summaries'] if x['arm']==arm)
        independent=next(x for x in slices if x['arm']==arm and x['policy']=='all')['Q']
        assert abs(independent-recorded['pair_equal_Q'])<1e-12
    result=dict(status='passed_independent_scalar',rows=len(seen),goals=len(ps),raw_sha256=sha(d/'responses.jsonl'),
        verifier_sha256=sha(Path(__file__)),slices=slices,
        scope='Fixed benchmark descriptive points. No mechanism or significance inferred from one model.')
    save(d/'independent_scalar_audit.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
