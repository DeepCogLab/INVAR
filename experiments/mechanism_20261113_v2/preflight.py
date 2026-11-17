"""No CUDA: all selected native suffixes, adapter labels and frozen source pools."""
import importlib.util
from common import *


def main():
    from transformers import AutoTokenizer
    def imported(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    v2=imported('preflight_v2',OLD.parent/'20261026/train_condition_anchor.py')
    legacy=imported('preflight_v1',OLD.parent/'20261025/train_lora.py')
    v4=imported('preflight_v4',OLD/'train_transfer.py')
    audit={};inventory=load(R/'adapter_inventory.json')
    for job,model in MODELS.items():
        tk=AutoTokenizer.from_pretrained(model['path'],trust_remote_code=False,local_files_only=True)
        audit[job]={}
        for source in SOURCES:
            rows=jl(panel(source)/'model_inputs.jsonl')
            cache,a=v4.tokenize_rows(v2,legacy,rows,tk,job,4096)
            for row in rows:
                assert not any(k in row for k in ('expected','expected_actions','correct','target'))
                kw={'enable_thinking':False} if job=='qwen3_8b' else {}
                text=tk.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,**kw)
                ids=tk.encode(text,add_special_tokens=False)
                assert ids==tk.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,**kw)
                assert len(ids)+128<=4096
            for state,meta in inventory[job][source].items():
                old=meta['training_identity']
                assert old['training_chat_template_sha256']==v2.digest(tk.chat_template)
                assert old['heldout_source']==source and old['training_seed']==20261026
                assert old['adapter_checkpoint_step']==int(state.rsplit('_',1)[1])
            audit[job][source]=a
    for source in SOURCES:
        pool_hashes={inventory[job][source][state]['training_identity']['source_pool_sha256']
            for job in MODELS for state in STATES[1:]}
        assert len(pool_hashes)==1,'Two arms/models do not share the frozen preparation'
    save(R/'preflight.json',dict(status='passed_cpu_native_and_identity_checks',tokenization=audit,
        no_gpu_model_loaded=True,no_labels_in_worker_inputs=True,all_48_checkpoints_verified=True))


if __name__=='__main__': main()
