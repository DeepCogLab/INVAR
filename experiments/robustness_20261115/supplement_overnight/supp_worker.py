import argparse
import importlib.metadata
import os
import subprocess
import time
from pathlib import Path
from supp_common import *


def main(job):
    cfg=verify_freeze();spec=JOBS[job];model_name=spec['model'];model=Path('/mnt/models/models')/MODELS[model_name]['folder']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    assert subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==GPU_UUID
    out=S/'runs'/job;out.mkdir(parents=True,exist_ok=False)
    ps={p['prompt_id']:p for p in jl(S/'frozen/model_inputs.jsonl')};trials=jl(S/'plans'/(job+'.jsonl'))
    assert trials==make_plan(job,list(ps.values()))
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True)
    ids={};texts={};mh={};prefill={}
    for t in trials:
        k=input_key(t)
        if k in ids:continue
        text,message_hash=render(tok,ps[t['prompt_id']],t,job)
        texts[k]=digest(text);mh[k]=message_hash;ids[k]=tok.encode(text,add_special_tokens=False)
        assert len(ids[k])+t['max_tokens']<=cfg['jobs'][job]['context']
        suffix=text.rsplit('<|im_start|>assistant',1)[-1] if MODELS[model_name]['kind']=='native_toggle' else ''
        prefill[k]=suffix.count('<think>')>suffix.count('</think>')
    preflight=load(S/'preflight.json')[job]
    assert digest(texts)==preflight['rendered_map_sha256']
    assert digest({k:digest(v) for k,v in ids.items()})==preflight['token_map_sha256']
    identity=dict(job=job,model=model_name,model_spec=MODELS[model_name],freeze_sha256=sha(S/'freeze.json'),
        source_main_freeze_sha256=cfg['main_freeze_sha256'],trial_plan_sha256=digest(trials),expected_rows=len(trials),
        weight_sha256={p.name:sha(p) for p in sorted(model.glob('*.safetensors'))},
        config_sha256={p.name:sha(p) for p in model.iterdir() if p.is_file() and p.suffix in ['.json','.jinja']},
        software={n:importlib.metadata.version(n) for n in ['vllm','torch','transformers','numpy']},gpu_uuid=GPU_UUID,
        rendered_sha256=texts,prompt_token_ids_sha256={k:digest(v) for k,v in ids.items()},messages_sha256=mh,prefilled_think=prefill,
        context=cfg['jobs'][job]['context'],max_num_seqs=cfg['jobs'][job]['max_num_seqs'],started_unix=time.time())
    prior=load(S/'model_identities'/(model_name+'.json'))
    assert identity['weight_sha256'] and identity['weight_sha256']==prior['weight_sha256']
    assert identity['config_sha256']==prior['config_sha256'] and identity['software']==prior['software']
    identity['parent_model_identity_sha256']=sha(S/'model_identities'/(model_name+'.json'))
    save(out/'identity.json',identity);identity_hash=digest(identity)
    started=time.time()
    llm=LLM(model=str(model),dtype='bfloat16',max_model_len=identity['context'],max_num_seqs=identity['max_num_seqs'],
        max_num_batched_tokens=16384,gpu_memory_utilization=.86,enforce_eager=True,enable_prefix_caching=True,
        generation_config='vllm',seed=20261116,trust_remote_code=True)
    loaded=time.time();tokens=0
    with (out/'responses.jsonl').open('x',encoding='utf-8') as f:
        for offset in range(0,len(trials),128):
            batch=trials[offset:offset+128]
            params=[SamplingParams(temperature=.7,top_p=.95,top_k=-1,max_tokens=t['max_tokens'],n=1,seed=t['sampling_seed']) for t in batch]
            begin=time.time();outputs=llm.generate([dict(prompt_token_ids=ids[input_key(t)]) for t in batch],params,use_tqdm=False)
            assert len(outputs)==len(batch)
            for t,g in zip(batch,outputs):
                k=input_key(t);o=g.outputs[0];assert list(g.prompt_token_ids)==ids[k]
                row=dict(**t,job=job,model=model_name,stage=ps[t['prompt_id']]['stage'],task_type=ps[t['prompt_id']]['task_type'],
                    identity_sha256=identity_hash,rendered_sha256=texts[k],prompt_token_ids_sha256=digest(g.prompt_token_ids),messages_sha256=mh[k],
                    response_text=o.text,response_with_special_tokens=tok.decode(o.token_ids,skip_special_tokens=False),
                    token_ids=list(o.token_ids),completion_tokens=len(o.token_ids),prompt_tokens=len(g.prompt_token_ids),finish_reason=o.finish_reason,stop_reason=o.stop_reason,
                    chunk_offset=offset,chunk_seconds=time.time()-begin)
                f.write(__import__('json').dumps(row,ensure_ascii=False)+'\n');tokens+=len(o.token_ids)
            f.flush();os.fsync(f.fileno());save(out/'progress.json',dict(rows=offset+len(batch),expected_rows=len(trials),completion_tokens=tokens,generation_seconds=time.time()-loaded,updated_unix=time.time()))
    save(out/'COMPLETE.json',dict(status='generated',rows=len(trials),raw_sha256=sha(out/'responses.jsonl'),identity_sha256=digest(identity),
        load_seconds=loaded-started,generation_seconds=time.time()-loaded,finished_unix=time.time(),completion_tokens=tokens))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True,choices=list(JOBS));main(ap.parse_args().job)
