"""Additional immutable plan, checkpoint and prefix audit of completed extensions."""
import argparse
from pathlib import Path
from audit_new_runs import R,jl,load,sha,save,digest


def audit(root,job,stage,output=None):
    assert stage in ['native_long','joint_reasoning']
    d=root/'runs'/job/stage;identity=load(d/'identity.json');complete=load(d/'COMPLETE.json')
    cfg_name='native_long_freeze.json' if stage=='native_long' else 'joint_reasoning_freeze_v2.json'
    cfg=load(root/cfg_name);primary=load(root/'freeze.json')
    for name,h in cfg['runtime_sha256'].items():assert sha(root/name)==h,name
    assert identity['protocol']==cfg and identity['job']==job
    runner='offline_native_long.py' if stage=='native_long' else 'offline_joint_reasoning_v2.py'
    assert identity['runner_sha256']==cfg['runtime_sha256'][runner]
    original=root/'runs'/job/'native';old_identity=load(original/'identity.json')
    old_weights=old_identity.get('weights_sha256',old_identity.get('checkpoint',{}).get('sha256'))
    old_config=old_identity.get('model_config_sha256',old_identity.get('checkpoint',{}).get('config_sha256'))
    assert old_weights and identity['weights_sha256']==old_weights
    assert identity['config_sha256']==old_config
    if 'template_sha256' in old_identity:
        assert identity.get('template_sha256',identity.get('tokenizer_config_sha256'))==old_identity['template_sha256']
    raw=d/'responses.jsonl';assert sha(raw)==complete['sha256']
    if stage=='native_long':
        assert sha(root/'frozen/model_inputs.jsonl')==primary['files']['model_inputs.jsonl']
        assert sha(root/'frozen/native_sample.json')==cfg['sample_sha256']
        ps={p['prompt_id']:p for p in jl(root/'frozen/model_inputs.jsonl')}
        ids=load(root/'frozen/native_sample.json')['prompt_ids']
        expected={(pid,'native_thinking',i) for pid in ids for i in range(8)}
        old_raw=original/'responses.jsonl';old_complete=load(original/'COMPLETE.json')
        assert sha(old_raw)==old_complete['sha256']==identity['original_raw_sha256']
        assert sha(original/'identity.json')==identity['original_identity_sha256']
        old={(r['prompt_id'],r['replicate']):r for r in jl(old_raw) if r['batch']=='native_thinking'}
        assert len(old)==1280
    else:
        for name,h in cfg['files'].items():assert sha(root/'joint_frozen'/name)==h
        ps={p['prompt_id']:p for p in jl(root/'joint_frozen/model_inputs.jsonl')}
        expected={(pid,arm,i) for pid in ps for arm in cfg['arms'] for i in range(cfg['repeats'])}
    assert all(set(p)=={'prompt_id','task_type','messages','output_schema'} for p in ps.values())
    seen=set();prefix=0;short_truncated_prefix=0
    for row in jl(raw):
        pid,arm,i=row['prompt_id'],row['batch'],row['replicate'];key=pid,arm,i
        assert key in expected and key not in seen;seen.add(key)
        assert row['model']==job and row['status']=='ok' and row['config_sha256']==digest(identity)
        assert row['messages_sha256']==digest(ps[pid]['messages'])
        assert type(row['completion_tokens']) is int and 0<=row['completion_tokens']<=cfg['max_tokens']
        assert row['finish_reason'] in ['length','stop']
        if stage=='native_long':
            previous=old[pid,i]
            assert row['sampling_seed']==previous['sampling_seed']
            assert row['original_finish_reason']==previous['finish_reason']
            same=row['response_text'].startswith(previous['response_text'])
            assert type(row['same_text_prefix_as_short']) is bool and row['same_text_prefix_as_short']==same
            prefix+=same;short_truncated_prefix+=same and previous['finish_reason']=='length'
        else:assert row['sampling_seed']==int(digest([20261111,'joint',pid,i])[:8],16)%(2**31-1)
    assert seen==expected and len(seen)==complete['rows']
    result=dict(status='passed_extension_identity',job=job,stage=stage,rows=len(seen),raw_sha256=complete['sha256'],
        identity_sha256=sha(d/'identity.json'),freeze_sha256=sha(root/cfg_name),verifier_sha256=sha(Path(__file__)),
        payload_allowlist_and_sha_verified=True,full_trial_plan_messages_seeds_verified=True,
        weights_and_model_config_match_accepted_short_run=True,completion_tokens_within_budget=True)
    if stage=='native_long':result.update(prefix_flags_recomputed=True,matching_prefix=prefix,short_truncated_and_matching_prefix=short_truncated_prefix)
    target=output or d/'extension_identity_audit.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert load(target)==result,'Preserve existing audit; use a new version if implementation changes'
    else:save(target,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=R);p.add_argument('--job',required=True)
    p.add_argument('--stage',choices=['native_long','joint_reasoning'],required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();print(__import__('json').dumps(audit(a.root,a.job,a.stage,a.output)))


if __name__=='__main__':main()
