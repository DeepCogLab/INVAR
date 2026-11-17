"""V2 local-weight evaluation. Output grammar never receives correctness labels."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'INVAR_dataset'))
from run_vllm_batch import MODEL_PATHS, prompt_text

def read(path):
    return [json.loads(s) for s in Path(path).open(encoding='utf-8-sig') if s.strip()]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=list(MODEL_PATHS))
    ap.add_argument('--stage',default='pilot')
    ap.add_argument('--batch-size',type=int,default=192)
    ap.add_argument('--max-tokens',type=int,default=128)
    a=ap.parse_args()
    inputs={p['prompt_id']:p for p in read(ROOT/'inputs.jsonl')}
    plan=[p for p in read(ROOT/f'{a.stage}_plan.jsonl') if p['model']==a.model]
    out=ROOT/'outputs'/a.stage/(a.model+'.jsonl');out.parent.mkdir(parents=True,exist_ok=True)
    config=dict(protocol='INVAR_v2_structured_20261024',model=a.model,temperature=.7,top_p=.95,top_k=-1,
                repetition_penalty=1.,max_tokens=a.max_tokens,base_seed=20261024,batch_size=a.batch_size,
                decoder='json_schema_xgrammar_compact',correctness_labels_given_to_decoder=False,
                inputs_sha256=sha((ROOT/'inputs.jsonl').read_bytes()))
    config_hash=sha(json.dumps(config,sort_keys=True).encode())
    done=set()
    if out.exists():
        for r in read(out):
            assert r['config_sha256']==config_hash,'Do not mix configurations'
            assert r['trial_id'] not in done,'Duplicate saved trial'
            done.add(r['trial_id'])
    todo=[p for p in plan if p['trial_id'] not in done]
    if not todo:return
    from transformers import AutoTokenizer,AutoConfig
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    path=MODEL_PATHS[a.model]
    tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    native=AutoConfig.from_pretrained(path,trust_remote_code=True)
    context=min(8192,getattr(native,'max_position_embeddings',8192))
    used={p['prompt_id'] for p in plan}
    rendered={pid:prompt_text(tokenizer,inputs[pid]['messages'],a.model) for pid in used}
    lens={pid:len(tokenizer.encode(s,add_special_tokens=False)) for pid,s in rendered.items()}
    assert max(lens.values())+a.max_tokens<=context,'Context overflow'
    identity=dict(config=config,config_sha256=config_hash,resolved_weights=str(Path(path).resolve()),
                  weight_config_sha256=sha(Path(path,'config.json').read_bytes()),context=context,
                  max_input_tokens=max(lens.values()),rendered_sha256=sha(json.dumps(rendered,sort_keys=True,ensure_ascii=False).encode()))
    out.with_suffix('.identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    llm=LLM(model=path,tokenizer=path,trust_remote_code=True,dtype='bfloat16',max_model_len=context,
            gpu_memory_utilization=.90,max_num_seqs=a.batch_size,enable_prefix_caching=True,
            generation_config='vllm',seed=20261024,
            structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True})
    grammars={pid:StructuredOutputsParams(json=inputs[pid]['output_schema'],disable_any_whitespace=True) for pid in used}
    begun=time.time()
    with out.open('a',encoding='utf-8') as f:
        for start in range(0,len(todo),a.batch_size):
            batch=todo[start:start+a.batch_size]
            seeds=[int(sha(('20261024|'+r['trial_id']).encode())[:8],16)%(2**31-1) for r in batch]
            params=[SamplingParams(temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,max_tokens=a.max_tokens,
                                   seed=seed,structured_outputs=grammars[r['prompt_id']]) for r,seed in zip(batch,seeds)]
            generated=llm.generate([rendered[r['prompt_id']] for r in batch],params,use_tqdm=False)
            assert len(generated)==len(batch)
            records=[]
            for row,g,seed in zip(batch,generated,seeds):
                assert len(g.outputs)==1
                o=g.outputs[0]
                records.append(dict(**row,response_text=o.text,status='ok',finish_reason=o.finish_reason,
                                    input_tokens=lens[row['prompt_id']],completion_tokens=len(o.token_ids),
                                    sampling_seed=seed,config_sha256=config_hash))
            f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));f.flush()
            for r in records:
                schema=inputs[r['prompt_id']]['output_schema'];obj=json.loads(r['response_text'])
                assert set(obj)==set(schema['properties']),'Output schema failure retained in raw file'
                for key,s in schema['properties'].items():
                    value=obj[key]
                    assert type(value) is (int if s['type']=='integer' else str)
                    assert 'enum' not in s or value in s['enum']
                    assert 'minimum' not in s or s['minimum']<=value<=s['maximum']
                assert r['finish_reason']!='length','Truncated output retained in raw file'
            (ROOT/f'{a.stage}_{a.model}_progress.json').write_text(json.dumps(dict(model=a.model,stage=a.stage,done=len(done)+start+len(batch),total=len(plan),seconds=round(time.time()-begun,1),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
            if start//a.batch_size%20==0:
                print(json.dumps(dict(model=a.model,stage=a.stage,done=len(done)+start+len(batch),total=len(plan),
                                      seconds=round(time.time()-begun,1))),flush=True)
    print(json.dumps(dict(model=a.model,stage=a.stage,status='complete',rows=len(plan))),flush=True)

if __name__=='__main__':
    from runtime_guard import guarded_generation
    guarded_generation(ROOT,main)
