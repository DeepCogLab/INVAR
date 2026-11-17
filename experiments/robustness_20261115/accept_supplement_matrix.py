"""Independent coverage/identity audit of canonical supplement, with missing jobs explicit."""
import collections,json
from pathlib import Path
from datetime import datetime,timezone
from campaign_common import R,load,jl,sha,digest,save
from accept_main_matrix import allowed

S=R/'supplement_overnight'
M=Path('artifacts/robustness_20261115/remote')

def main():
    cfg=load(S/'freeze.json');fh=sha(S/'freeze.json');idx=load(S/'results/ACCEPTANCE_INDEX.json')
    assert fh==idx['freeze_sha256']=='d4723b218f8db2d397be7a0ea8b14a173c266c78921b510ba224471f2e952b45'
    for name,h in cfg['files'].items():assert sha(S/name)==sha(M/'supplement_overnight'/name)==h,name
    inputs=jl(S/'frozen/model_inputs.jsonl');labels=jl(S/'frozen/evaluation.jsonl')
    ps={p['prompt_id']:p for p in inputs};truth={p['prompt_id']:p for p in labels}
    assert len(ps)==len(inputs)==len(truth)==len(labels) and set(ps)==set(truth)
    refs=0;quartets={}
    for rel in ['core/behavior_quartets.jsonl','core/goal_quartets.jsonl','parameter/goal_quartets.jsonl']:
        qs=jl(S/'frozen'/rel);quartets[rel]=len(qs)
        for q in qs:
            for k in 'ABCD':assert q[k] in ps and ps[q[k]]['language']==q['language'];refs+=1
            if 'goal' in rel:
                sets={k:set(truth[q[k]]['expected_actions']) for k in 'ABCD'}
                assert all(sets.values()) and sets['A']==sets['B'] and sets['C']==sets['D']
                for k in 'ABCD':assert all(allowed(a,ps[q[k]]['output_schema']) for a in sets[k])
                if rel.startswith('parameter'):assert sets['A'].isdisjoint(sets['C'])
    accepted=[];missing=[]
    for row in idx['completion_matrix']:
        job=row['job'];plan=jl(S/'plans'/(job+'.jsonl'));spec=cfg['jobs'][job]
        n=4 if 'behavior' in job else 2
        key=lambda t:(t['prompt_id'],t['language'],t['instruction_language'],t['arm'],t['max_tokens'],t['replicate'])
        expected={key(t):t for t in plan};assert len(plan)==len(expected)==spec['expected_rows']
        prompt_ids={t['prompt_id'] for t in plan}
        assert prompt_ids=={p['prompt_id'] for p in inputs if p['task_type']==('behavior' if 'behavior' in job else 'goal')}
        perprompt=collections.Counter(t['prompt_id'] for t in plan)
        assert set(perprompt.values())=={8 if job.startswith('language_') else n*2}
        for t in plan:
            assert t['language']==ps[t['prompt_id']]['language'] and 0<=t['replicate']<n
            assert t['sampling_seed']==int(digest([20261116,'overnight',job,{k:v for k,v in t.items() if k!='sampling_seed'}])[:8],16)%(2**31-1)
            assert t['max_tokens'] in ([4096] if job.startswith('language_') else [4096,8192])
            assert t['instruction_language'] in ['zh','en']
            assert t['arm'] in (['direct','brief_reasoning'] if job.startswith('language_') else ['native_on'] if 'qwen' in job else ['effort_high'])
            if not job.startswith('language_'):assert t['instruction_language']==t['language']
        if job not in idx['accepted']:
            missing.append(dict(job=job,status=row['status'],planned=len(plan),observed_rows=row['observed_rows']));continue
        root='gpt_date_restore_20261116' if row['result_source']=='engineering_date_restoration' else 'supplement_overnight'
        d=M/root/'runs'/job;done=load(d/'COMPLETE.json');identity=load(d/'identity.json');audit=load(d/'INDEPENDENT_ACCEPTANCE.json');score=load(d/'SCORING_COMPLETE.json')
        assert sha(d/'responses.jsonl')==done['raw_sha256']==audit['raw_sha256']==score['raw_sha256']==idx['accepted'][job]['raw_sha256']
        assert done['identity_sha256']==digest(identity) and identity['freeze_sha256']==audit['freeze_sha256']==score['freeze_sha256']==fh
        assert identity['trial_plan_sha256']==digest(plan)
        assert sha(d/'SCORING_COMPLETE.json')==audit['scoring_sha256']==idx['accepted'][job]['score_sha256']
        assert identity['context']==spec['context'] and identity['max_num_seqs']==spec['max_num_seqs']
        parent=load(S/'model_identities'/(identity['model']+'.json'))
        for k in ['weight_sha256','config_sha256','software']:assert identity[k]==parent[k]
        unique_keys=set();seen=set();tokens=0;identity_hash=digest(identity)
        for r in jl(d/'responses.jsonl'):
            k=key(r);assert k in expected and k not in seen;seen.add(k)
            assert all(r[k]==v for k,v in expected[key(r)].items())
            assert r['job']==job and r['model']==identity['model'] and r['identity_sha256']==identity_hash
            ik=r['prompt_id']+'__'+'__'.join(str(r[k]) for k in ['language','instruction_language','arm','max_tokens']);unique_keys.add(ik)
            assert r['rendered_sha256']==identity['rendered_sha256'][ik] and r['prompt_token_ids_sha256']==identity['prompt_token_ids_sha256'][ik]
            assert r['messages_sha256']==identity['messages_sha256'][ik]
            assert r['completion_tokens']==len(r['token_ids'])<=r['max_tokens'];tokens+=r['completion_tokens']
        assert seen==set(expected) and len(seen)==done['rows']==score['rows']==len(plan)
        pre=load(S/'preflight.json')[job]
        assert digest(identity['rendered_sha256'])==pre['rendered_map_sha256'] and digest(identity['prompt_token_ids_sha256'])==pre['token_map_sha256']
        assert unique_keys==set(identity['rendered_sha256'])==set(identity['prompt_token_ids_sha256'])
        assert tokens==done['completion_tokens']
        accepted.append(dict(job=job,rows=len(seen),source=root,raw_sha256=done['raw_sha256'],unique_inputs=len(unique_keys),completion_tokens=tokens,
            score_sha256=sha(d/'SCORING_COMPLETE.json'),identity_sha256=sha(d/'identity.json'),audit_sha256=sha(d/'INDEPENDENT_ACCEPTANCE.json')))
    result=dict(status='passed_all_six_supplement_jobs' if not missing else 'accepted_completed_jobs_incomplete_campaign',
        expected_rows=21888,accepted_rows=sum(x['rows'] for x in accepted),accepted=accepted,missing=missing,
        verified_quartet_references=refs,quartet_counts=quartets,unique_input_count=len(inputs),
        raw_trial_plan_seed_token_identity_freeze_and_full_optimum_sets_checked=True,freeze_sha256=fh,
        audit_code_sha256=sha(Path(__file__)),checked_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-hoc supplement only. Engineering date restoration keeps original failure provenance; missing jobs are not zero scored.')
    save(S/'results/MATRIX_INTEGRITY_ACCEPTANCE.json',result);print(json.dumps(result))

if __name__=='__main__':main()
