"""Actual local-weight vLLM trials. Evaluation labels never enter model inputs."""
import argparse,hashlib,importlib.metadata,json,os,time
from pathlib import Path
R=Path(__file__).resolve().parent
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def read(p):return [json.loads(s) for s in Path(p).open(encoding='utf-8-sig') if s.strip()]
def render(tokenizer,messages,policy):
    if policy=='plain':return '\n\n'.join(m['content'] for m in messages)+'\n\n'
    if policy=='mistral':messages=[dict(role='user',content='\n\n'.join(m['content'] for m in messages))]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
def seed(pid,rep,batch):return int(digest([20261104,pid,rep,batch])[:8],16)%(2**31-1)
def valid(text,schema,finish):
    if finish=='length':return False
    try:
        obj=json.loads(text)
        if not isinstance(obj,dict) or set(obj)!=set(schema['properties']):return False
        for k,s in schema['properties'].items():
            v=obj[k]
            if type(v) is not (int if s['type']=='integer' else str):return False
            if 'enum' in s and v not in s['enum']:return False
            if 'minimum' in s and not s['minimum']<=v<=s['maximum']:return False
        return True
    except (ValueError,TypeError):return False
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--stage',choices=['pilot','full','native'],required=True);a=ap.parse_args()
    job=next(j for j in json.loads((R/'jobs.json').read_text()) if j['id']==a.job)
    freeze=json.loads((R/'freeze.json').read_text());assert filehash(R/'frozen/inputs.jsonl')==freeze['input_sha256']
    full=read(R/'frozen/inputs.jsonl');byid={p['prompt_id']:p for p in full}
    plan=json.loads((R/'frozen/plan.json').read_text()) if a.stage=='full' else [dict(prompt_id=p['prompt_id'],replicate=0,batch=a.stage) for p in full]
    if a.stage=='native':
        from collections import defaultdict
        buckets=defaultdict(list)
        for p in sorted(full,key=lambda p:digest(p['prompt_id'])):buckets[p['task_type']].append(p)
        ids={p['prompt_id'] for bucket in buckets.values() for p in bucket[:12]}
        plan=[p for p in plan if p['prompt_id'] in ids]
    d=R/'runs'/a.job/a.stage;d.mkdir(parents=True,exist_ok=True)
    base=Path('/workspace/models')/job['base'];adapter=Path('/workspace/models')/job['adapter'] if job['adapter'] else None
    if adapter:
        ac=json.loads((adapter/'adapter_config.json').read_text())
        declared=ac['base_model_name_or_path'].split('/')[-1]
        assert declared==job['base'],f'Wrong base: {declared} vs {job["base"]}'
        assert not ac.get('modules_to_save'),'Non-LoRA trainable modules require separate compatibility check'
    from transformers import AutoTokenizer,AutoConfig
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.lora.request import LoRARequest
    tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    native=AutoConfig.from_pretrained(str(base),local_files_only=True)
    max_tokens=2048 if a.stage=='native' else 128
    context=min(getattr(native,'max_position_embeddings',8192),8192)
    ids=sorted({r['prompt_id'] for r in plan})
    author_reasoning=a.stage=='native' and job['id'].startswith('socialr1')
    rendered={}
    for pid in ids:
        messages=byid[pid]['messages']
        if author_reasoning:
            messages=[dict(m) for m in messages]
            messages[0]['content']+='\nFirst reason within <think> </think>, then put the requested JSON answer within <Answer> </Answer>.'
            rendered[pid]=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        else:rendered[pid]=render(tokenizer,messages,job['render'])
    lengths={pid:len(tokenizer.encode(t,add_special_tokens=False)) for pid,t in rendered.items()}
    assert max(lengths.values())+max_tokens<=context,'No truncation allowed'
    config=dict(protocol=freeze['protocol'],job=job,stage=a.stage,inputs_sha256=freeze['input_sha256'],
      plan_sha256=digest(plan),temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,max_tokens=max_tokens,
      seed_policy='sha256_prompt_replicate_batch_common_random_numbers',base_seed=20261104,
      decoder='unconstrained' if a.stage=='native' else 'json_schema_xgrammar_compact',
      render=job['render'],thinking=author_reasoning,native_author_instruction=author_reasoning,context=context,correctness_labels_given_to_decoder=False,
      rendered_sha256=digest(rendered),runner_sha256=filehash(__file__),
      versions={k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','peft']},
      base_config_sha256=filehash(base/'config.json'),adapter_config_sha256=filehash(adapter/'adapter_config.json') if adapter else None)
    # Full checkpoint content hashes are cached in this campaign, never inferred from filenames.
    for checkpoint in [base,adapter]:
        if checkpoint is None:continue
        hp=R/'checkpoint_hashes'/f'{checkpoint.name}.json';hp.parent.mkdir(exist_ok=True)
        paths=sorted(checkpoint.glob('*.safetensors'))
        stats=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
        if hp.exists():
            cached=json.loads(hp.read_text());assert cached['stats']==[list(x) for x in stats],'Checkpoint changed'
        else:
            cached=dict(stats=stats,files={p.name:filehash(p) for p in paths});save(hp,cached)
        assert paths,'No checkpoint weights'
        config[checkpoint.name+'_weights']=cached['files']
    ch=digest(config);identity=dict(config=config,config_sha256=ch,planned_rows=len(plan),max_input_tokens=max(lengths.values()))
    ip=d/'identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity,'Changed identity requires new directory'
    else:save(ip,identity)
    raw=d/'responses.jsonl';done=set()
    if raw.exists():
        for r in read(raw):
            key=(r['prompt_id'],r['replicate'],r['batch']);assert key not in done and r['config_sha256']==ch;done.add(key)
    expected={(r['prompt_id'],r['replicate'],r['batch']) for r in plan};assert done<=expected
    todo=[r for r in plan if (r['prompt_id'],r['replicate'],r['batch']) not in done]
    if not todo:return
    batch_size=96
    kwargs=dict(model=str(base),tokenizer=str(base),dtype='bfloat16',max_model_len=context,
      gpu_memory_utilization=.88,max_num_seqs=batch_size,enable_prefix_caching=True,generation_config='vllm',seed=20261104,
      structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True})
    if adapter:kwargs.update(enable_lora=True,max_lora_rank=max(16,ac['r']),max_loras=1)
    if getattr(native,'quantization_config',None):kwargs.update(quantization='bitsandbytes',load_format='bitsandbytes')
    llm=LLM(**kwargs)
    request=LoRARequest(job['id'],1,str(adapter)) if adapter else None
    grammars={pid:StructuredOutputsParams(json=byid[pid]['output_schema'],disable_any_whitespace=True) for pid in ids}
    start=time.time()
    with raw.open('a',encoding='utf-8') as f:
        for offset in range(0,len(todo),batch_size):
            chunk=todo[offset:offset+batch_size]
            params=[SamplingParams(temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,max_tokens=max_tokens,
              seed=seed(r['prompt_id'],r['replicate'],r['batch']),
              **({} if a.stage=='native' else dict(structured_outputs=grammars[r['prompt_id']]))) for r in chunk]
            generated=llm.generate([rendered[r['prompt_id']] for r in chunk],params,lora_request=request,use_tqdm=False)
            assert len(generated)==len(chunk)
            for r,g in zip(chunk,generated):
                assert len(g.outputs)==1
                if request:assert g.lora_request is not None and g.lora_request.lora_int_id==1,'Adapter not applied'
                o=g.outputs[0];pid=r['prompt_id']
                rec=dict(**r,model=a.job,trial_id=f'{a.job}|{pid}|{r["replicate"]}|{r["batch"]}',
                  response_text=o.text,status='ok',finish_reason=o.finish_reason,completion_tokens=len(o.token_ids),input_tokens=lengths[pid],
                  sampling_seed=seed(pid,r['replicate'],r['batch']),config_sha256=ch,
                  messages_sha256=digest(byid[pid]['messages']),rendered_sha256=digest(rendered[pid]),
                  schema_valid=valid(o.text,byid[pid]['output_schema'],o.finish_reason))
                f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            f.flush()
            save(d/'progress.json',dict(done=len(done)+offset+len(chunk),total=len(plan),elapsed_seconds=time.time()-start,updated_unix=time.time()))
    saved=read(raw);assert len(saved)==len(expected) and {(r['prompt_id'],r['replicate'],r['batch']) for r in saved}==expected
    save(d/'COMPLETE.json',dict(rows=len(saved),responses_sha256=filehash(raw),config_sha256=ch,
      invalid=sum(not r['schema_valid'] for r in saved),truncated=sum(r['finish_reason']=='length' for r in saved)))
if __name__=='__main__':main()
