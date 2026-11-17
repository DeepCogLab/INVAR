"""Additive post-generation audit of templates, checkpoints and every delivered trial."""
import argparse
from pathlib import Path
from common import R,jl,load,save,sha,digest,seed,tasks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--stage',choices=['canonical','parameter'],required=True);a=ap.parse_args()
    cfg=load(R/'freeze.json');d=R/'runs'/a.job/a.stage
    assert (d/'SCORING_COMPLETE.json').exists()
    for name,h in {**cfg['files'],**cfg['runtime_sha256']}.items():assert sha(R/name)==h
    identity=load(d.parent/'identity.json');model=Path(cfg['models'][a.job]['path'])
    assert sha(model/'config.json')==identity['config_sha256']
    assert sha(model/'tokenizer_config.json')==identity['tokenizer_config_sha256']
    assert {p.name:sha(p) for p in sorted(model.glob('*.safetensors'))}==identity['weights_sha256']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    ps,plan=tasks(R,a.stage);expected=set(plan);rendered={}
    for pid,p in ps.items():
        for arm in cfg['arms']:
            text=tokenizer.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=arm=='native_thinking')
            assert len(tokenizer.encode(text,add_special_tokens=False))+cfg['max_tokens']<=cfg['max_model_len']
            rendered[pid,arm]=digest(text)
    seen=set()
    for row in jl(d/'responses.jsonl'):
        key=row['prompt_id'],row['batch'],row['replicate'];pid,arm,rep=key
        assert key in expected and key not in seen;seen.add(key)
        assert row['rendered_sha256']==rendered[pid,arm]
        assert row['messages_sha256']==digest(ps[pid]['messages'])
        assert row['sampling_seed']==seed(a.stage,pid,rep) and row['config_sha256']==digest(identity)
        assert 0<=row['completion_tokens']<=cfg['max_tokens']
    assert seen==expected
    complete=load(d/'COMPLETE.json');assert sha(d/'responses.jsonl')==complete['sha256']
    record=dict(status='passed_delivery_identity',job=a.job,stage=a.stage,rows=len(seen),raw_sha256=complete['sha256'],
        templates_independently_recomputed=True,checkpoint_rehashed=True,complete_plan_and_seeds_verified=True,
        verifier_sha256=sha(Path(__file__)),freeze_sha256=sha(R/'freeze.json'))
    p=d/'extension_identity_audit.json'
    if p.exists():assert load(p)==record
    else:save(p,record)
    print(__import__('json').dumps(record))


if __name__=='__main__':main()
