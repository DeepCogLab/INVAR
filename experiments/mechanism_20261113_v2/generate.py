"""Independent constrained generation on the frozen checkpoint panel. No labels."""
import argparse
import time
from common import *


def main(job):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.lora.request import LoRARequest
    verify_freeze(); directory=out(job)/'generation'; directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'STARTED.json').exists(), 'No silent replay of attempted answers'
    save(directory/'STARTED.json',dict(unix=time.time(),freeze_sha256=sha(R/'freeze.json')))
    path=MODELS[job]['path']; tk=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    kw={'enable_thinking':False} if job=='qwen3_8b' else {}
    all_rows={}; texts={}; grammars={}; info={}
    for source in SOURCES:
        rows=jl(panel(source)/'model_inputs.jsonl'); all_rows[source]=rows
        for row in rows:
            pid=row['prompt_id'];text=tk.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,**kw)
            ids=tk.encode(text,add_special_tokens=False); assert len(ids)+128<=4096
            if pid in texts: assert texts[pid]==text
            texts[pid]=text
            grammars[pid]=StructuredOutputsParams(json=row['output_schema'],disable_any_whitespace=True)
            info[pid]=dict(rendered_sha256=digest(text),messages_sha256=digest(row['messages']),input_tokens=len(ids))
    save(directory/'rendered_identity.json',dict(template_sha256=digest(tk.chat_template),kwargs=kw,prompts=info))
    llm=LLM(model=path,tokenizer=path,trust_remote_code=False,dtype='bfloat16',max_model_len=4096,
        gpu_memory_utilization=.90,max_num_seqs=32,enable_prefix_caching=True,generation_config='vllm',
        seed=20261113,enable_lora=True,max_lora_rank=8,max_loras=1,max_cpu_loras=48,
        structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True})
    counter=0;done=0;begun=time.time();total=load(R/'plan.json')['science_generated_rows']//2
    def run(rows,source,state,request,meta,pilot=False):
        nonlocal done
        todo=[(r,rep) for r in rows for rep in range(1 if pilot else reps(r))]
        raw=directory/('pilot.jsonl' if pilot else f'{source}_{state}.jsonl')
        assert not raw.exists()
        with raw.open('x',encoding='utf-8') as f:
            for start in range(0,len(todo),32):
                chunk=todo[start:start+32]
                params=[SamplingParams(temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,max_tokens=128,n=1,
                    seed=seed(job,source,'pilot_'+state if pilot else state,r['prompt_id'],rep),
                    structured_outputs=grammars[r['prompt_id']]) for r,rep in chunk]
                outputs=llm.generate([texts[r['prompt_id']] for r,rep in chunk],params,lora_request=request,use_tqdm=False)
                assert len(outputs)==len(chunk)
                for (row,rep),output,param in zip(chunk,outputs,params):
                    binding=getattr(output,'lora_request',None)
                    binding_ok=(binding is None if request is None else binding is not None and
                                binding.lora_int_id==request.lora_int_id and binding.lora_name==request.lora_name)
                    assert len(output.outputs)==1
                    c=output.outputs[0]
                    record=dict(job=job,source_panel=source,state=state,prompt_id=row['prompt_id'],replicate=rep,
                        response_text=c.text,output_token_ids=list(c.token_ids),finish_reason=c.finish_reason,
                        sampling_seed=param.seed,adapter_sha256=meta['fingerprint']['content_sha256'] if meta else None,
                        lora_binding_ok=binding_ok,lora_id=binding.lora_int_id if binding else None,
                        **info[row['prompt_id']])
                    f.write(canonical(record)+'\n');f.flush()
                    assert binding_ok,'Wrong adapter binding; preserve raw and stop'
                if not pilot:done+=len(chunk)
                save(directory/'progress.json',dict(source=source,state=state,done=done,total=total,
                    elapsed=time.time()-begun,unix=time.time(),pilot=pilot))
    # Format delivery only; all eight first fixed rows retained, never accuracy-screened.
    run(all_rows[SOURCES[0]][:8],SOURCES[0],'baseline',None,None,True)
    for source in SOURCES:
        for state in STATES:
            meta=adapter(job,source,state);request=None
            if meta:
                for name,h in [('adapter_model.safetensors',meta['fingerprint']['adapter_weights_sha256']),
                               ('adapter_config.json',meta['fingerprint']['adapter_config_sha256'])]:
                    assert sha(Path(meta['path'])/name)==h
                counter+=1;request=LoRARequest(f'{job}_{source}_{state}',counter,meta['path'])
            run(all_rows[source],source,state,request,meta)
    assert done==total
    save(directory/'COMPLETE.json',dict(status='complete',rows=done,pilot_rows=8,elapsed=time.time()-begun,
        files_sha256={p.name:sha(p) for p in directory.glob('*.jsonl')},freeze_sha256=sha(R/'freeze.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();main(a.job)
