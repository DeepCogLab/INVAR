"""Entire scoring path on synthetic oracle, failure and shared-prompt controls."""
import tempfile
from pathlib import Path
import common
import score
from common import canonical,load,save,write_jl,sha,STATES,seed,digest


def exercise(root,all_missing=False):
    root.mkdir();f=root/'frozen'/'test';d=root/'runs'/'synthetic'
    rows=[dict(prompt_id=p,task_type=task,field='action',candidates=[0,10],
        messages=[dict(role='system',content='synthetic'),dict(role='user',content=p)])
        for p,task in [('a','behavior'),('b','behavior'),('c','goal'),('e','goal')]]
    write_jl(f/'model_inputs.jsonl',rows)
    write_jl(f/'edges.jsonl',[dict(edge_id=k,left='a',right='b',stratum=k) for k in ('train','heldout_source','heldout_variant')])
    write_jl(f/'goal_membership.jsonl',[dict(prompt_id=p,stratum='train') for p in ('c','e')])
    write_jl(f/'goal_labels.jsonl',[dict(prompt_id='c',expected=[0]),dict(prompt_id='e',expected=[10])])
    save(root/'freeze.json',dict(synthetic=True));freeze=sha(root/'freeze.json')
    for state in STATES:
        js=state.startswith('anchor_js');proxies=[];raw=[]
        for r in rows:
            pid=r['prompt_id'];correct=0 if pid in ('a','c') else 10
            answer=0 if pid in ('a','b') and js else correct if r['task_type']=='behavior' or js else 10-correct
            dist=[1.,0.] if answer==0 else [0.,1.]
            proxies.append(dict(prompt_id=pid,input_sha256=digest(r),candidates=[0,10],
                complete_logps=[0.,-1000.] if answer==0 else [-1000.,0.],probabilities=dist))
            for rep in range(64 if r['task_type']=='behavior' else 16):
                raw.append(dict(job='synthetic',source_panel='test',state=state,prompt_id=pid,replicate=rep,
                    response_text=canonical(dict(action=True if all_missing else answer)),finish_reason='stop',
                    sampling_seed=seed('synthetic','test',state,pid,rep),lora_binding_ok=True,adapter_sha256=None,
                    messages_sha256=digest(r['messages'])))
        write_jl(d/'proxy'/f'test_{state}.jsonl',proxies)
        write_jl(d/'generation'/f'test_{state}.jsonl',raw)
    for stage in ('proxy','generation'):
        save(d/stage/'COMPLETE.json',dict(freeze_sha256=freeze,files_sha256={p.name:sha(p) for p in (d/stage).glob('*.jsonl')}))
    common.R=root;score.R=root;score.SOURCES=['test'];score.verify_freeze=lambda:None;score.adapter=lambda *args:None
    score.main('synthetic')
    contrasts=common.jl(d/'analysis/contrasts.jsonl')
    goals=common.jl(d/'analysis/goals.jsonl')
    for c in contrasts:
        assert abs(c['delta_proxy_W1']+1)<1e-12
        if all_missing:
            assert c['delta_decoder_W1_conditional'] is None
            assert c['delta_decoder_W1_shared_prompt_bootstrap']['undefined']==500
        else:
            assert c['delta_decoder_W1_conditional']==-1
            assert c['delta_decoder_W1_shared_prompt_bootstrap']['empirical_range']==[-1.,-1.]
    assert all(g['Q']==(0 if all_missing or not g['state'].startswith('anchor_js') else 1) for g in goals)
    return dict(status='passed',all_missing=all_missing,contrasts=len(contrasts))


def main():
    original=common.R
    with tempfile.TemporaryDirectory(prefix='invar_mechanism_synthetic_') as tmp:
        checks=[exercise(Path(tmp)/'oracle'),exercise(Path(tmp)/'all_missing',True)]
    save(original/'synthetic_pipeline_validation.json',dict(status='passed',checks=checks,
        scientific_answers=0,shared_prompts_reused_within_each_resample=True,all_missing_conditional_not_filled=True))
    print('full synthetic scoring passed')


if __name__=='__main__':main()
