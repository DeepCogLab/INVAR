"""One fixed all-prompt 4096-token budget extension; never reroll only failures."""
import argparse
import json
import time
from pathlib import Path
from run_api import R,load,jl,save,digest,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    cfg=load(R/'native_long_freeze.json');job=cfg['models'][a.job]
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')}
    sample=load(R/'frozen/native_sample.json')['prompt_ids']
    assert sha(R/'frozen/native_sample.json')==cfg['sample_sha256']
    original=R/'runs'/a.job/'native'
    old_complete=load(original/'COMPLETE.json')
    assert sha(original/'responses.jsonl')==old_complete['sha256']
    old={(r['prompt_id'],r['replicate']):r for r in jl(original/'responses.jsonl') if r['batch']=='native_thinking'}
    assert len(old)==1280
    model=Path(job['path']);t=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    texts={pid:t.apply_chat_template(ps[pid]['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True) for pid in sample}
    assert max(len(t.encode(text,add_special_tokens=False)) for text in texts.values())+4096<=8192
    d=R/'runs'/a.job/'native_long';d.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol=cfg,job=a.job,runner_sha256=sha(Path(__file__)),
        original_identity_sha256=sha(original/'identity.json'),original_raw_sha256=old_complete['sha256'],
        config_sha256=sha(model/'config.json'),template_sha256=sha(model/'tokenizer_config.json'),
        weights_sha256={p.name:sha(p) for p in sorted(model.glob('*.safetensors'))})
    if (d/'identity.json').exists():assert load(d/'identity.json')==identity
    else:save(d/'identity.json',identity)
    ch=digest(identity);raw=d/'responses.jsonl';done=set()
    if raw.exists():
        for r in jl(raw):
            key=(r['prompt_id'],r['replicate']);assert key not in done and r['config_sha256']==ch;done.add(key)
    expected={(pid,i) for pid in sample for i in range(8)};assert done<=expected
    kwargs=dict(model=str(model),dtype=job['dtype'],max_model_len=8192,gpu_memory_utilization=.85,
        max_num_seqs=32,enable_prefix_caching=True,generation_config='vllm',seed=20261111)
    if job.get('quantization'):kwargs['quantization']=job['quantization']
    llm=LLM(**kwargs)
    tasks=[]
    # Preserve original sampling style: API used n=8 with one seed, offline used per-replicate seeds.
    if job['original_sampling']=='n8':
        for pid in sample:
            presence=[(pid,i) in done for i in range(8)]
            assert all(presence) or not any(presence),'Partial request preserved; no reroll'
            if not all(presence):tasks.append((pid,None))
    else:tasks=[(pid,i) for pid in sample for i in range(8) if (pid,i) not in done]
    start=time.time()
    with raw.open('a',encoding='utf-8') as f:
        for offset in range(0,len(tasks),8):
            chunk=tasks[offset:offset+8];params=[]
            for pid,rep in chunk:
                seed=old[pid,0 if rep is None else rep]['sampling_seed']
                params.append(SamplingParams(n=8 if rep is None else 1,seed=seed,
                    temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,max_tokens=4096))
            outputs=llm.generate([texts[pid] for pid,rep in chunk],params,use_tqdm=False)
            for (pid,rep),g,sp in zip(chunk,outputs,params):
                assert len(g.outputs)==(8 if rep is None else 1)
                for idx,o in enumerate(sorted(g.outputs,key=lambda x:x.index)):
                    assert o.index==idx
                    i=idx if rep is None else rep
                    row=dict(model=a.job,prompt_id=pid,batch='native_thinking',replicate=i,response_text=o.text,
                        finish_reason=o.finish_reason,status='ok',completion_tokens=len(o.token_ids),sampling_seed=sp.seed,
                        config_sha256=ch,messages_sha256=digest(ps[pid]['messages']),
                        same_text_prefix_as_short=o.text.startswith(old[pid,i]['response_text']),
                        original_finish_reason=old[pid,i]['finish_reason'])
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');done.add((pid,i))
            f.flush();save(d/'progress.json',dict(done=len(done),total=1280,elapsed_seconds=time.time()-start,updated_unix=time.time()))
    assert done==expected
    save(d/'COMPLETE.json',dict(rows=1280,sha256=sha(raw),finished_unix=time.time()))


if __name__=='__main__':main()
