"""CPU-only structural and tokenizer audit before committing scientific trials."""
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from campaign_common import R, MODELS, arms, jl, render, digest, sha, save


def main():
    ps=jl(R/'frozen/model_inputs.jsonl');labels=jl(R/'frozen/evaluation.jsonl')
    assert len(ps)==len(labels)==2574 and {p['prompt_id'] for p in ps}=={p['prompt_id'] for p in labels}
    result={};hashes={}
    for job,info in MODELS.items():
        model=Path('/mnt/models/models')/info['folder']
        tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True)
        rendered={};lengths=[];token_hashes={};prefilled={}
        for p in ps:
            for arm in arms(job):
                text,mh=render(tok,job,arm,p);ids=tok.encode(text,add_special_tokens=False)
                assert len(ids)+4096<=8192,(job,p['prompt_id'],len(ids))
                suffix=text.rsplit('<|im_start|>assistant',1)[-1] if info['kind']=='native_toggle' else ''
                prefilled[arm]=suffix.count('<think>')>suffix.count('</think>')
                rendered[p['prompt_id']+'__'+arm]=digest(text);token_hashes[p['prompt_id']+'__'+arm]=digest(ids);lengths.append(len(ids))
        hashes[job]=rendered
        assert all(rendered[p['prompt_id']+'__'+arms(job)[0]]!=rendered[p['prompt_id']+'__'+arms(job)[1]] for p in ps)
        result[job]=dict(max_prompt_tokens=max(lengths),min_prompt_tokens=min(lengths),
            tokenizer_config_sha256=sha(model/'tokenizer_config.json'),rendered_map_sha256=digest(rendered),
            token_ids_map_sha256=digest(token_hashes),vocab_size=len(tok),prefilled_think_by_arm=prefilled)
    assert hashes['mistral7b_v01']==hashes['sotopia_mistral7b']
    coverage={}
    for path in sorted((R/'frozen').rglob('*quartets.jsonl')):
        qs=jl(path);assert all(q[k] in {p['prompt_id'] for p in ps} for q in qs for k in 'ABCD')
        coverage[str(path.relative_to(R))]=dict(quartets=len(qs),pairs_by_language={lang:len({q['pair_id'] for q in qs if q['language']==lang}) for lang in ['zh','en']},sources=dict(Counter(q.get('source_id','parameter') for q in qs)))
    save(R/'preflight_acceptance.json',dict(status='passed_cpu_all_inputs_all_checkpoints',models=result,
        coverage=coverage,sotopia_and_baseline_render_identically=True,no_scientific_responses_generated=True))
    print('passed_cpu_all_inputs_all_checkpoints')


if __name__=='__main__':main()
