"""Read-only delivery audit: complete attempts, rendered messages, weights and adapters."""
import argparse
import collections
import time
from common import *


def main(job):
    from transformers import AutoTokenizer
    verify_freeze();directory=out(job);target=directory/'delivery_identity_audit.json'
    assert not target.exists(),'Existing identity audit is immutable'
    model=MODELS[job]['path'];identity=load(R/'model_identity.json')[job]
    for name,h in identity['files_sha256'].items(): assert sha(Path(model)/name)==h
    tk=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    kw={'enable_thinking':False} if job=='qwen3_8b' else {}
    rendered=load(directory/'generation/rendered_identity.json')
    assert rendered['template_sha256']==digest(tk.chat_template) and rendered['kwargs']==kw
    count=0; files={}
    for source in SOURCES:
        rows={r['prompt_id']:r for r in jl(panel(source)/'model_inputs.jsonl')}
        for pid,row in rows.items():
            text=tk.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,**kw)
            expected=dict(rendered_sha256=digest(text),messages_sha256=digest(row['messages']),input_tokens=len(tk.encode(text,add_special_tokens=False)))
            assert expected==rendered['prompts'][pid]
        for state in STATES:
            meta=adapter(job,source,state)
            if meta:
                for name,h in [('adapter_model.safetensors',meta['fingerprint']['adapter_weights_sha256']),
                               ('adapter_config.json',meta['fingerprint']['adapter_config_sha256'])]:assert sha(Path(meta['path'])/name)==h
            seen=set();path=directory/'generation'/f'{source}_{state}.jsonl'
            for r in jl(path):
                pid=r['prompt_id'];rep=r['replicate'];assert (pid,rep) not in seen;seen.add((pid,rep))
                assert pid in rows and 0<=rep<reps(rows[pid]) and r['sampling_seed']==seed(job,source,state,pid,rep)
                assert r['job']==job and r['state']==state and r['source_panel']==source
                assert all(r[k]==v for k,v in rendered['prompts'][pid].items())
                assert r['adapter_sha256']==(meta['fingerprint']['content_sha256'] if meta else None) and r['lora_binding_ok']
                assert len(r['output_token_ids'])<=128 and all(type(t) is int and 0<=t<len(tk) for t in r['output_token_ids'])
            assert len(seen)==sum(reps(r) for r in rows.values());count+=len(seen)
            files[str(path.relative_to(directory))]=sha(path)
    assert count==load(R/'plan.json')['science_generated_rows']//2
    for stage in ('proxy','generation'):
        c=load(directory/stage/'COMPLETE.json');assert c['freeze_sha256']==sha(R/'freeze.json')
        for name,h in c['files_sha256'].items():assert sha(directory/stage/name)==h
    save(target,dict(status='passed_complete_delivery_identity',job=job,science_rows=count,
        model_identity_sha256=sha(R/'model_identity.json'),adapter_inventory_sha256=sha(R/'adapter_inventory.json'),
        freeze_sha256=sha(R/'freeze.json'),raw_sha256=files,auditor_sha256=sha(Path(__file__)),checked_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();main(a.job)
