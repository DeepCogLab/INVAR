"""Matched native-reasoning diagnostic, with original messages and frozen sample."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from run_api import R,load,jl,save,digest,sha,valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',choices=['qwen3_4b_native','socialr1_4_native','qwen3_8b_native','socialr1_8_native'],required=True);a=ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    cfg=load(R/'reasoning_pair_freeze.json')
    model=Path('/workspace/models')/cfg['models'][a.job]
    ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')}
    sample=load(R/'frozen/native_sample.json')
    assert sha(R/'frozen/native_sample.json')==cfg['sample_sha256']
    assert sha(R/'frozen/model_inputs.jsonl')==cfg['model_inputs_sha256']
    d=R/'runs'/a.job/'native';d.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    tasks=[]
    for pid in sample['prompt_ids']:
        for think in [False,True]:
            text=tokenizer.apply_chat_template(ps[pid]['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=think)
            assert len(tokenizer.encode(text,add_special_tokens=False))+1408<=4096
            for rep in range(8):
                tasks.append(dict(prompt_id=pid,batch='native_thinking' if think else 'native_no_thinking',replicate=rep,text=text))
    identity=dict(protocol=cfg,job=a.job,runner_sha256=sha(Path(__file__)),
        model_config_sha256=sha(model/'config.json'),
        weights_sha256={p.name:sha(p) for p in sorted(model.glob('*.safetensors'))},
        template_sha256=sha(model/'tokenizer_config.json'),n_planned=len(tasks),max_model_len=4096)
    ip=d/'identity.json'
    if ip.exists():
        assert load(ip)==identity
    else:
        save(ip,identity)
    ch=digest(identity)
    raw=d/'responses.jsonl';done=set()
    if raw.exists():
        for row in jl(raw):
            key=(row['prompt_id'],row['batch'],row['replicate'])
            assert key not in done and row['config_sha256']==ch
            done.add(key)
    expected={(t['prompt_id'],t['batch'],t['replicate']) for t in tasks}
    assert done<=expected
    remaining=[t for t in tasks if (t['prompt_id'],t['batch'],t['replicate']) not in done]
    llm=LLM(model=str(model),dtype='bfloat16',max_model_len=4096,gpu_memory_utilization=.85,
        max_num_seqs=64,enable_prefix_caching=True,generation_config='vllm',seed=20261111)
    start=time.time()
    with raw.open('a',encoding='utf-8') as f:
        for offset in range(0,len(remaining),32):
            chunk=remaining[offset:offset+32]
            params=[SamplingParams(temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,max_tokens=1408,
                seed=int(digest([20261111,t['prompt_id'],t['replicate']])[:8],16)%(2**31-1)) for t in chunk]
            outputs=llm.generate([t['text'] for t in chunk],params,use_tqdm=False)
            assert len(outputs)==len(chunk)
            for t,g,sp in zip(chunk,outputs,params):
                o=g.outputs[0]
                row={k:t[k] for k in ['prompt_id','batch','replicate']}
                row.update(model=a.job,response_text=o.text,finish_reason=o.finish_reason,status='ok',
                    schema_valid=valid(o.text,ps[t['prompt_id']]['output_schema'],o.finish_reason),
                    completion_tokens=len(o.token_ids),sampling_seed=sp.seed,config_sha256=ch)
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                done.add((t['prompt_id'],t['batch'],t['replicate']))
            f.flush();save(d/'progress.json',dict(done=len(done),total=len(expected),elapsed_seconds=time.time()-start,updated_unix=time.time()))
    assert done==expected
    save(d/'COMPLETE.json',dict(rows=len(done),sha256=sha(raw),finished_unix=time.time()))


if __name__=='__main__':
    main()
