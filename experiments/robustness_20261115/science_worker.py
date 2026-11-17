"""One checkpoint, prospectively fixed trials; no label access while generating."""
import argparse
import importlib.metadata
import json
import os
import time
import subprocess
from pathlib import Path
from campaign_common import R, MODELS, GPU_UUID, SEED, arms, render, digest, sha, jl, load, save, trial_seed


def plan(job, prompts, protocol):
    rows=[]
    for p in prompts:
        n=protocol['repeats']['parameter' if p['stage']=='parameter' else p['task_type']]
        for arm in arms(job):
            for rep in range(n):
                rows.append(dict(prompt_id=p['prompt_id'],arm=arm,replicate=rep,
                    sampling_seed=trial_seed(job,p['prompt_id'],arm,rep)))
    return sorted(rows,key=lambda x:digest([SEED,'order',job,x]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    actual=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert actual==GPU_UUID
    cfg=load(R/'freeze.json')
    for name,h in cfg['files'].items():assert sha(R/name)==h,name
    protocol=cfg['protocol'];job=MODELS[a.job];model=Path('/mnt/models/models')/job['folder']
    out=R/'runs'/a.job;out.mkdir(parents=True,exist_ok=False)
    ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')}
    trials=plan(a.job,list(ps.values()),protocol)
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True)
    texts={};mh={};input_ids={};prefilled={}
    for pid,p in ps.items():
        for arm in arms(a.job):texts[pid,arm],mh[pid,arm]=render(tokenizer,a.job,arm,p)
    for key,text in texts.items():
        input_ids[key]=tokenizer.encode(text,add_special_tokens=False)
        assert len(input_ids[key])+protocol['sampling']['max_tokens']<=protocol['max_model_len'],key
        suffix=text.rsplit('<|im_start|>assistant',1)[-1] if job['kind']=='native_toggle' else ''
        prefilled[key]=suffix.count('<think>')>suffix.count('</think>')
    assert all(texts[pid,arms(a.job)[0]]!=texts[pid,arms(a.job)[1]] for pid in ps)
    preflight=load(R/'preflight_acceptance.json')['models'][a.job]
    assert digest({pid+'__'+arm:digest(text) for (pid,arm),text in texts.items()})==preflight['rendered_map_sha256']
    assert digest({pid+'__'+arm:digest(ids) for (pid,arm),ids in input_ids.items()})==preflight['token_ids_map_sha256']
    assert sha(model/'tokenizer_config.json')==preflight['tokenizer_config_sha256']
    # Hash full model shards once per checkpoint. Never copy or expose credentials.
    weight_files=sorted(model.glob('*.safetensors')) or sorted(model.glob('pytorch_model*.bin'))
    assert weight_files
    adapter=Path('/mnt/models/models')/job['adapter_folder'] if 'adapter_folder' in job else None
    identity=dict(job=a.job,model=job,freeze_sha256=sha(R/'freeze.json'),
        weight_sha256={p.name:sha(p) for p in weight_files},
        adapter_sha256={p.name:sha(p) for p in adapter.iterdir() if p.is_file()} if adapter else None,
        config_sha256={p.name:sha(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.jinja')},
        software={n:importlib.metadata.version(n) for n in ['vllm','torch','transformers','numpy']},
        gpu_uuid=actual,trial_plan_sha256=digest(trials),expected_rows=len(trials),
        rendered_sha256={pid+'__'+arm:digest(text) for (pid,arm),text in texts.items()},
        prompt_token_ids_sha256={pid+'__'+arm:digest(ids) for (pid,arm),ids in input_ids.items()},
        max_prompt_tokens=max(map(len,input_ids.values())),tokenization='explicit HF token IDs; add_special_tokens=False after chat-template rendering',
        prefilled_think={pid+'__'+arm:value for (pid,arm),value in prefilled.items()},
        started_unix=time.time())
    save(out/'identity.json',identity)
    identity_hash=digest(identity)
    start=time.time()
    adapter_args=dict(enable_lora=True,max_lora_rank=8) if adapter else {}
    llm=LLM(model=str(model),dtype='bfloat16',max_model_len=protocol['max_model_len'],
        gpu_memory_utilization=.86,max_num_seqs=64,max_num_batched_tokens=16384,
        enforce_eager=True,enable_prefix_caching=True,generation_config='vllm',seed=SEED,trust_remote_code=True,**adapter_args)
    from vllm.lora.request import LoRARequest
    lora=LoRARequest('sotopia',1,str(adapter)) if adapter else None
    if lora:
        assert llm.llm_engine.add_lora(lora), 'Adapter must be loaded successfully before any scientific attempt'
        save(out/'adapter_loading.json',dict(status='engine_add_lora_returned_true',adapter_sha256=identity['adapter_sha256'],
            common_base_tokenizer=True,lora_name=lora.lora_name,lora_int_id=lora.lora_int_id))
    loaded=time.time();tokens=0
    with (out/'responses.jsonl').open('x',encoding='utf-8') as f:
        for offset in range(0,len(trials),256):
            batch=trials[offset:offset+256];sp=protocol['sampling']
            params=[SamplingParams(**sp,n=1,seed=t['sampling_seed']) for t in batch]
            begun=time.time()
            outputs=llm.generate([dict(prompt_token_ids=input_ids[t['prompt_id'],t['arm']]) for t in batch],params,use_tqdm=False,lora_request=lora)
            elapsed=time.time()-begun
            for t,g in zip(batch,outputs):
                if lora:assert g.lora_request is not None and g.lora_request.lora_int_id==lora.lora_int_id
                o=g.outputs[0];p=ps[t['prompt_id']];tokens+=len(o.token_ids)
                assert list(g.prompt_token_ids)==input_ids[t['prompt_id'],t['arm']]
                row=dict(**t,model=a.job,language=p['language'],stage=p['stage'],task_type=p['task_type'],
                    identity_sha256=identity_hash,messages_sha256=mh[t['prompt_id'],t['arm']],
                    rendered_sha256=digest(texts[t['prompt_id'],t['arm']]),
                    prompt_token_ids_sha256=digest(g.prompt_token_ids),
                    response_text=o.text,response_with_special_tokens=tokenizer.decode(o.token_ids,skip_special_tokens=False),
                    token_ids=list(o.token_ids),completion_tokens=len(o.token_ids),prompt_tokens=len(g.prompt_token_ids),
                    finish_reason=o.finish_reason,stop_reason=o.stop_reason,
                    chunk_offset=offset,chunk_seconds=elapsed)
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
            save(out/'progress.json',dict(rows=offset+len(batch),expected_rows=len(trials),
                completion_tokens=tokens,generation_seconds=time.time()-loaded,updated_unix=time.time()))
    save(out/'COMPLETE.json',dict(status='generated',rows=len(trials),raw_sha256=sha(out/'responses.jsonl'),
        identity_sha256=identity_hash,load_seconds=loaded-start,generation_seconds=time.time()-loaded,
        completion_tokens=tokens,finished_unix=time.time()))


if __name__=='__main__':main()
