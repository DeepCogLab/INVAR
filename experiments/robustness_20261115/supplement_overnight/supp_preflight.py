"""CPU-only check of every new input before supplementary freeze."""
from supp_common import *


def main():
    from transformers import AutoTokenizer
    ps={p['prompt_id']:p for p in jl(S/'frozen/model_inputs.jsonl')};result={}
    for job,spec in JOBS.items():
        trials=jl(S/'plans'/(job+'.jsonl'));assert trials==make_plan(job,list(ps.values()))
        assert len(trials)==spec['expected_rows']
        tok=AutoTokenizer.from_pretrained('/mnt/models/models/'+MODELS[spec['model']]['folder'],local_files_only=True,trust_remote_code=True)
        hashes={};tokens={};maximum=0
        for t in trials:
            key=input_key(t)
            if key in hashes:continue
            p=ps[t['prompt_id']];text,mh=render(tok,p,t,job);ids=tok.encode(text,add_special_tokens=False)
            maximum=max(maximum,len(ids));assert len(ids)+t['max_tokens']<= (8192 if spec['experiment']=='instruction_language' else 12288)
            hashes[key]=digest(text);tokens[key]=digest(ids)
            if spec['experiment']=='instruction_language':
                other=dict(t,instruction_language='en' if t['instruction_language']=='zh' else 'zh')
                assert messages(p,t,job)[1:]==messages(p,other,job)[1:]
                assert messages(p,t,job)[0]!=messages(p,other,job)[0]
            else:
                other=dict(t,max_tokens=8192 if t['max_tokens']==4096 else 4096)
                assert render(tok,p,t,job)==render(tok,p,other,job)
        result[job]=dict(trials=len(trials),unique_renderings=len(hashes),max_prompt_tokens=maximum,
            rendered_map_sha256=digest(hashes),token_map_sha256=digest(tokens),
            tokenizer_config_sha256=sha(Path('/mnt/models/models')/MODELS[spec['model']]['folder']/'tokenizer_config.json'))
    save(S/'preflight.json',result)
    print(__import__('json').dumps(result))


if __name__=='__main__':main()
