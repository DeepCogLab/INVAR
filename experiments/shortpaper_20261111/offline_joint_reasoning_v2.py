"""Same free-behavior prompts, native thinking toggle, equal output budget."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from run_api import R,load,jl,save,digest,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    cfg=load(R/'joint_reasoning_freeze_v2.json');job=cfg['models'][a.job]
    for name,h in cfg['files'].items():assert sha(R/'joint_frozen'/name)==h
    ps={p['prompt_id']:p for p in jl(R/'joint_frozen/model_inputs.jsonl')}
    assert all(set(p)=={'prompt_id','task_type','messages','output_schema'} for p in ps.values())
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    model=Path(job['path']);t=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    rendered={}
    for pid,p in ps.items():
        for think in [False,True]:
            text=t.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=think)
            assert len(t.encode(text,add_special_tokens=False))+4096<=8192
            rendered[pid,think]=text
    d=R/'runs'/a.job/'joint_reasoning';d.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol=cfg,job=a.job,runner_sha256=sha(Path(__file__)),
        config_sha256=sha(model/'config.json'),tokenizer_config_sha256=sha(model/'tokenizer_config.json'),
        weights_sha256={p.name:sha(p) for p in sorted(model.glob('*.safetensors'))},
        software={n:importlib.metadata.version(n) for n in ['vllm','torch','transformers']})
    if (d/'identity.json').exists():assert load(d/'identity.json')==identity
    else:save(d/'identity.json',identity)
    ch=digest(identity);raw=d/'responses.jsonl';done=set()
    if raw.exists():
        for r in jl(raw):
            key=r['prompt_id'],r['batch'],r['replicate'];assert key not in done and r['config_sha256']==ch;done.add(key)
    tasks=[(pid,think,rep) for think in [False,True] for pid in sorted(ps) for rep in range(128)]
    expected={(pid,'native_thinking' if think else 'native_no_thinking',rep) for pid,think,rep in tasks};assert done<=expected
    todo=[x for x in tasks if (x[0],'native_thinking' if x[1] else 'native_no_thinking',x[2]) not in done]
    kwargs=dict(model=str(model),dtype=job['dtype'],max_model_len=8192,gpu_memory_utilization=.85,
        max_num_seqs=32,enable_prefix_caching=True,generation_config='vllm',seed=20261111)
    if job.get('quantization'):kwargs['quantization']=job['quantization']
    llm=LLM(**kwargs);start=time.time()
    with raw.open('a',encoding='utf-8') as f:
        for offset in range(0,len(todo),32):
            chunk=todo[offset:offset+32]
            params=[SamplingParams(temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,max_tokens=4096,
                seed=int(digest([20261111,'joint',pid,rep])[:8],16)%(2**31-1)) for pid,think,rep in chunk]
            outputs=llm.generate([rendered[pid,think] for pid,think,rep in chunk],params,use_tqdm=False)
            assert len(outputs)==len(chunk)
            for (pid,think,rep),g,sp in zip(chunk,outputs,params):
                assert len(g.outputs)==1;o=g.outputs[0]
                row=dict(model=a.job,prompt_id=pid,batch='native_thinking' if think else 'native_no_thinking',replicate=rep,
                    response_text=o.text,finish_reason=o.finish_reason,status='ok',completion_tokens=len(o.token_ids),
                    sampling_seed=sp.seed,messages_sha256=digest(ps[pid]['messages']),config_sha256=ch)
                f.write(json.dumps(row,ensure_ascii=False)+'\n');done.add((pid,row['batch'],rep))
            f.flush();save(d/'progress.json',dict(done=len(done),total=len(expected),elapsed_seconds=time.time()-start,updated_unix=time.time()))
    assert done==expected;save(d/'COMPLETE.json',dict(rows=len(done),sha256=sha(raw),finished_unix=time.time()))


if __name__=='__main__':main()
