"""One native model per GPU; every attempt retained, no answer-dependent retry."""
import argparse
import importlib.metadata
import time
import json
import sys
from pathlib import Path
from common import R, load, jl, save, sha, digest, seed, tasks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', required=True); a = ap.parse_args()
    cfg = load(R/'freeze.json'); job = cfg['models'][a.job]
    for name, h in cfg['files'].items(): assert sha(R/name) == h
    for name, h in cfg['runtime_sha256'].items(): assert sha(R/name) == h
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model = Path(job['path']); tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    all_inputs = {}
    for stage in cfg['stages']:
        ps, _ = tasks(R, stage); all_inputs.update(ps)
    all_inputs.update({p['prompt_id']: p for p in jl(R/'frozen/pilot.jsonl')})
    rendered = {}
    for pid, p in all_inputs.items():
        assert set(p) == {'prompt_id', 'task_type', 'messages', 'output_schema'}
        for arm in cfg['arms']:
            text = tokenizer.apply_chat_template(p['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=arm == 'native_thinking')
            assert len(tokenizer.encode(text, add_special_tokens=False)) + cfg['max_tokens'] <= cfg['max_model_len']
            rendered[pid, arm] = text
    identity = dict(protocol_sha256=sha(R/'freeze.json'), job=a.job, runner_sha256=sha(Path(__file__)),
        config_sha256=sha(model/'config.json'), tokenizer_config_sha256=sha(model/'tokenizer_config.json'),
        weights_sha256={p.name: sha(p) for p in sorted(model.glob('*.safetensors'))},
        software={n: importlib.metadata.version(n) for n in ['vllm', 'torch', 'transformers']})
    previous = load(Path(cfg['previous_verified_runs_root'])/'runs'/a.job/'joint_reasoning/identity.json')
    assert identity['weights_sha256'] == previous['weights_sha256']
    assert identity['config_sha256'] == previous['config_sha256']
    assert identity['tokenizer_config_sha256'] == previous['tokenizer_config_sha256']
    d = R/'runs'/a.job; d.mkdir(parents=True, exist_ok=True)
    assert not (d/'identity.json').exists(), 'Do not resume by reinitializing model/RNG silently'
    save(d/'identity.json', identity)
    kwargs = dict(model=str(model), dtype=job['dtype'], max_model_len=cfg['max_model_len'], gpu_memory_utilization=.85,
        max_num_seqs=32, enable_prefix_caching=True, generation_config='vllm', seed=cfg['seed_base'])
    if job.get('quantization'): kwargs['quantization'] = job['quantization']
    llm = LLM(**kwargs)
    sys.path.insert(0,str(R/'vendor'))
    from audit_new_runs import parse_final
    pilot = {p['prompt_id']: p for p in jl(R/'frozen/pilot.jsonl')}
    for stage in ['pilot'] + cfg['stages']:
        if stage == 'pilot': ps = pilot; plan = [(pid, arm, 0) for arm in cfg['arms'] for pid in sorted(ps)]
        else: ps, plan = tasks(R, stage)
        sd = d/stage; sd.mkdir(); save(sd/'plan.json', dict(rows=len(plan),plan_sha256=digest(plan)))
        start = time.time(); raw = sd/'responses.jsonl'
        with raw.open('x', encoding='utf-8') as f:
            for offset in range(0, len(plan), 32):
                chunk = plan[offset:offset+32]
                params = [SamplingParams(temperature=cfg['temperature'], top_p=cfg['top_p'], top_k=cfg['top_k'],
                    repetition_penalty=cfg['repetition_penalty'], max_tokens=cfg['max_tokens'], seed=seed(stage,pid,rep)) for pid, arm, rep in chunk]
                outputs = llm.generate([rendered[pid,arm] for pid,arm,rep in chunk], params, use_tqdm=False)
                assert len(outputs) == len(chunk)
                for (pid,arm,rep), g, sp in zip(chunk,outputs,params):
                    assert len(g.outputs) == 1
                    o = g.outputs[0]
                    row = dict(model=a.job,stage=stage,prompt_id=pid,batch=arm,replicate=rep,response_text=o.text,
                        finish_reason=o.finish_reason,status='ok',completion_tokens=len(o.token_ids),sampling_seed=sp.seed,
                        messages_sha256=digest(ps[pid]['messages']),rendered_sha256=digest(rendered[pid,arm]),config_sha256=digest(identity))
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                f.flush(); save(sd/'progress.json',dict(done=min(offset+32,len(plan)),total=len(plan),elapsed_seconds=time.time()-start,updated_unix=time.time()))
        save(sd/'COMPLETE.json',dict(rows=len(plan),sha256=sha(raw),finished_unix=time.time()))
        if stage == 'pilot':
            rates={arm:0 for arm in cfg['arms']}
            for row in jl(raw):
                obj,error=parse_final(row['response_text'],row['finish_reason'],ps[row['prompt_id']]['output_schema'])
                rates[row['batch']]+=obj is not None
            save(sd/'delivery_check.json',dict(valid_by_arm=rates,criterion='At least one parsed final per arm; no correctness selection or reroll.'))
            assert all(v>=1 for v in rates.values()),'All pilot outputs in an arm fail final delivery; inspect original outputs'
    save(d/'GENERATION_COMPLETE.json',dict(rows=sum(cfg[s]['rows_per_model'] for s in cfg['stages']),pilot_rows=20,finished_unix=time.time()))


if __name__ == '__main__': main()
