"""Fixed complete-answer likelihoods; no sampling, optimization or label access."""
import argparse
import contextlib
import importlib.util
import math
import time
from common import *


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main(job):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    cfg=verify_freeze(); directory=out(job)/'proxy'; directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'STARTED.json').exists(), 'No silent rerun of a started diagnostic'
    save(directory/'STARTED.json',dict(unix=time.time(),freeze_sha256=sha(R/'freeze.json')))
    model_path=MODELS[job]['path']
    v2=imported('frozen_candidate_helpers',OLD.parent/'20261026/train_condition_anchor.py')
    legacy=imported('frozen_suffix_helpers',OLD.parent/'20261025/train_lora.py')
    v4=imported('frozen_native_tokenization',OLD/'train_transfer.py')
    for name,h in load(R/'model_identity.json')[job]['files_sha256'].items(): assert sha(Path(model_path)/name)==h
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval()
    model.config.use_cache=False
    torch.manual_seed(20261113); torch.cuda.manual_seed_all(20261113)
    proof=[]; begun=time.time(); done=0
    for source in SOURCES:
        rows=jl(panel(source)/'model_inputs.jsonl')
        cache,audit=v4.tokenize_rows(v2,legacy,rows,tokenizer,job,4096)
        save(directory/f'{source}_tokenization.json',audit)
        for state in STATES:
            meta=adapter(job,source,state)
            if meta:
                for name,h in [('adapter_model.safetensors',meta['fingerprint']['adapter_weights_sha256']),
                               ('adapter_config.json',meta['fingerprint']['adapter_config_sha256'])]:
                    assert sha(Path(meta['path'])/name)==h
                if not isinstance(model,PeftModel): model=PeftModel.from_pretrained(model,meta['path'],adapter_name='current').eval()
                else:
                    if 'current' in model.peft_config: model.delete_adapter('current')
                    model.load_adapter(meta['path'],adapter_name='current',is_trainable=False)
                model.set_adapter('current'); model.eval()
            context=model.disable_adapter() if state=='baseline' and isinstance(model,PeftModel) else contextlib.nullcontext()
            records=[]
            with context, torch.inference_mode():
                for index,row in enumerate(rows):
                    logps,lengths=v2.candidate_logps(torch,model,cache,legacy,row)
                    scores=logps.double()
                    probabilities=torch.softmax(scores,dim=0).cpu().tolist()
                    assert all(math.isfinite(x) for x in scores.tolist()) and abs(math.fsum(probabilities)-1)<1e-10
                    if index==0:
                        # Explicit full-sequence positions independently check suffix alignment.
                        # Match projection row count to control BF16 GEMM shape effects.
                        tokens,labels=cache[(row['prompt_id'],v2.canonical(row['candidates'][0]))]
                        ids=torch.tensor([tokens],device='cuda')
                        positions=[i-1 for i,v in enumerate(labels) if v!=-100]
                        explicit=torch.tensor(positions+[len(tokens)-1],device='cuda')
                        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=explicit).logits[0,:-1].float()
                        targets=torch.tensor([labels[i+1] for i in positions],device='cuda')
                        direct=torch.log_softmax(logits,-1).gather(-1,targets[:,None]).sum()
                        difference=abs(float(direct)-float(logps[0]))
                        assert difference<.03, ('suffix_alignment',difference)
                        proof.append(dict(source=source,state=state,prompt_id=row['prompt_id'],
                            absolute_logp_difference=difference, tolerance=.03,
                            note='BF16 explicit full-sequence positions vs suffix slice, same mask and projection row count; numerical alignment check'))
                    records.append(dict(source_panel=source,state=state,prompt_id=row['prompt_id'],
                        candidates=row['candidates'],complete_logps=scores.cpu().tolist(),
                        suffix_token_lengths=lengths,probabilities=probabilities,
                        input_sha256=digest(row),adapter_sha256=meta['fingerprint']['content_sha256'] if meta else None))
                    done+=1
                    if index%8==0:
                        save(directory/'progress.json',dict(source=source,state=state,done=done,
                            total=load(R/'plan.json')['proxy_prompt_distributions']//2,elapsed=time.time()-begun,unix=time.time()))
            write_jl(directory/f'{source}_{state}.jsonl',records)
    save(directory/'alignment_checks.json',proof)
    save(directory/'COMPLETE.json',dict(status='complete',distributions=done,elapsed=time.time()-begun,
        files_sha256={p.name:sha(p) for p in directory.glob('*.jsonl')},freeze_sha256=sha(R/'freeze.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();main(a.job)
