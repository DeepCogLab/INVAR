"""Independent trial accounting and native parsing; only additive audit outputs."""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

R=Path(__file__).resolve().parent
BASE=Path('/workspace/project_social_simulation/invar_reliability_runs/20261024_v2')


def jl(p):
    with p.open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def parse_final(text,finish,schema):
    if finish=='length':return None,'truncated'
    s=text.strip()
    if s.startswith('<think>'):
        if s.count('</think>')!=1:return None,'incomplete_reasoning'
        s=s[s.index('</think>')+len('</think>'):].strip()
    if s.startswith('<Answer>') and s.endswith('</Answer>'):
        s=s[len('<Answer>'):-len('</Answer>')].strip()
    if s.startswith('```') and s.endswith('```'):
        s=re.sub(r'^```(?:json)?\s*','',s[:-3]).strip()
    try:
        obj=json.loads(s)
        if not isinstance(obj,dict) or set(obj)!=set(schema['properties']):return None,'format'
        for k,spec in schema['properties'].items():
            v=obj[k]
            if type(v) is not (int if spec['type']=='integer' else str):return None,'type'
            if 'enum' in spec and v not in spec['enum']:return None,'menu'
            if 'minimum' in spec and not spec['minimum']<=v<=spec['maximum']:return None,'menu'
        return obj,None
    except (ValueError,TypeError):return None,'format'


def audit(job,stage):
    d=R/'runs'/job/stage
    complete=load(d/'COMPLETE.json');identity=load(d/'identity.json')
    raw=d/'responses.jsonl'
    assert sha(raw)==complete['sha256']
    byid={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')}
    payloads={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')}
    freeze=load(R/'freeze.json')
    assert sha(BASE/'inputs.jsonl')==freeze['source_input_sha256']
    for name,h in freeze['files'].items():assert sha(R/'frozen'/name)==h
    sample=load(R/'frozen/native_sample.json')['prompt_ids']
    repeats=set(load(R/'frozen/repeat_ids.json')['prompt_ids'])
    expected=set()
    if stage=='main':
        for p in byid.values():
            for rep in range(freeze['main_repeats'][p['task_type']]):expected.add((p['prompt_id'],'main',rep))
        for pid in repeats:
            for rep in range(256):expected.add((pid,'repeat',rep))
    else:
        for pid in sample:
            for arm in ['native_no_thinking','native_thinking']:
                for rep in range(8):expected.add((pid,arm,rep))
    seen=set();acc=collections.defaultdict(list);groups=collections.defaultdict(lambda:collections.Counter())
    normalized=collections.defaultdict(list);repeat_values=collections.defaultdict(list)
    ch=digest(identity)
    with raw.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);pid=row['prompt_id'];arm=row['batch'];rep=row['replicate']
            key=(pid,arm,rep);assert key in expected and key not in seen;seen.add(key)
            assert row['config_sha256']==ch and row['model']==job and row['status']=='ok'
            p=byid[pid]
            if 'request_id' in row:
                assert row['messages_sha256']==digest(payloads[pid]['messages'])
                seed_batch='native' if stage=='native' else arm
                seed=int(digest([20261111,pid,seed_batch,rep//32])[:8],16)%(2**31-1)
            else:
                seed=int(digest([20261111,pid,rep])[:8],16)%(2**31-1)
            assert row['sampling_seed']==seed
            obj,error=parse_final(row['response_text'],row['finish_reason'],p['output_schema'])
            g=groups[arm];g['n']+=1;g['valid']+=obj is not None;g['truncated']+=row['finish_reason']=='length'
            if stage=='main':
                assert error is None and row['schema_valid']
                if p['task_type']=='behavior':
                    v=obj['action'];o=p['outcome']
                    x=float(v==o['target_action']) if o['kind']=='probability' else v/o['denominator']
                    (normalized if arm=='main' else repeat_values)[pid].append(x)
                else:
                    v=next(iter(obj.values()));acc[pid].append(int(v in p['expected_actions'] if p['task_type']=='goal' else v==p['expected_answer']))
            else:
                good=obj is not None and obj['action'] in p['expected_actions']
                g['correct']+=good;acc[(pid,arm)].append(int(good))
    assert seen==expected and len(seen)==complete['rows']
    result=dict(status='passed_independent_scalar',job=job,stage=stage,rows=len(seen),
        raw_sha256=complete['sha256'],verifier_sha256=sha(Path(__file__)),trial_plan_identity_seed_messages_verified=True)
    if stage=='native':
        summaries=[]
        recorded=load(d/'SCORING_COMPLETE.json')['summaries']
        for arm,g in sorted(groups.items()):
            point=sum(good for (pid,a),vals in acc.items() if a==arm for good in vals)/g['n']
            old=next(r for r in recorded if r['arm']==arm)
            assert math.isclose(point,old['prompt_equal_Q'],abs_tol=1e-12)
            assert g['valid']==old['valid'] and g['truncated']==old['truncated']
            summaries.append(dict(arm=arm,**g,Q_all_attempts=point,
                Q_given_parse=g['correct']/g['valid'] if g['valid'] else None))
        result['summaries']=summaries
    else:
        # Execute the frozen separate scalar verifier with only its filesystem root overridden.
        path=BASE/'independent_recompute.py';source=path.read_text()
        needle='R=Path(__file__).resolve().parent'
        assert source.count(needle)==1
        source=source.replace(needle,f'R=Path({str(d/"scoring")!r})')
        sys.path.insert(0,str(BASE/'INVAR_dataset'));sys.argv=[str(path),'--model',job]
        exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
        report=d/'scoring/reports'/job
        cs=list(csv.DictReader((report/'control_summary.csv').open(encoding='utf-8-sig')))
        control_points=[]
        for name in ['fact_quartets.jsonl','goal_quartets.jsonl']:
            for q in jl(BASE/name):
                control_points.append(dict(pair_id=q['pair_id'],kind=q.get('probe_type','goal'),
                    accuracy=sum(statistics.mean(acc[q[t]]) for t in 'ABCD')/4))
        for kind in ['changed_fact','stable_fact','goal']:
            perpair=collections.defaultdict(list)
            for q in control_points:
                if q['kind']==kind:perpair[q['pair_id']].append(q['accuracy'])
            point=statistics.mean(statistics.mean(v) for v in perpair.values())
            row=next(r for r in cs if r['group']=='type' and r['value']==kind)
            assert math.isclose(point,float(row['accuracy']),abs_tol=1e-12)
        result['frozen_scalar_verifier_sha256']=sha(path)
        result['control_aggregate_points_verified']=True
        result['behavior_and_all_control_quartets_verified']=True
    save(d/'independent_scalar_audit.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--stage',required=True,choices=['main','native']);a=ap.parse_args()
    audit(a.job,a.stage)
