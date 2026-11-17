"""Resumable supplementary trials on existing localhost vLLM services."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

R = Path(__file__).resolve().parent


def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def jl(p):
    with p.open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
    t.replace(p)


def valid(text,schema,finish):
    if finish=='length':
        return False
    try:
        x=json.loads(text)
        if not isinstance(x,dict) or set(x)!=set(schema['properties']):
            return False
        for k,s in schema['properties'].items():
            v=x[k]
            if type(v) is not (int if s['type']=='integer' else str):
                return False
            if 'enum' in s and v not in s['enum']:
                return False
            if 'minimum' in s and not s['minimum']<=v<=s['maximum']:
                return False
        return True
    except (ValueError,TypeError):
        return False


def tasks(stage,freeze):
    ps=jl(R/'frozen/model_inputs.jsonl')
    if stage=='pilot':
        selected=[]
        for typ in freeze['main_repeats']:
            selected.extend(sorted([p for p in ps if p['task_type']==typ],key=lambda p:digest(p['prompt_id']))[:4])
        return [dict(prompt_id=p['prompt_id'],batch='pilot',chunk=0,start=0,n=2,messages=p['messages'],schema=p['output_schema'],thinking=False) for p in selected]
    rows=[]
    if stage=='main':
        repeats=set(load(R/'frozen/repeat_ids.json')['prompt_ids'])
        for p in ps:
            for batch,total in [('main',freeze['main_repeats'][p['task_type']])]+([('repeat',256)] if p['prompt_id'] in repeats else []):
                for start in range(0,total,32):
                    rows.append(dict(prompt_id=p['prompt_id'],batch=batch,chunk=start//32,start=start,n=min(32,total-start),messages=p['messages'],schema=p['output_schema'],thinking=False))
    elif stage=='goal_diagnostic':
        for p in jl(R/'frozen/goal_diagnostics.jsonl'):
            for arm in ['original','payoff_table']:
                rows.append(dict(prompt_id=p['prompt_id'],batch=arm,chunk=0,start=0,n=32,messages=p[arm],schema=p['output_schema'],thinking=False))
    elif stage=='native':
        ids=set(load(R/'frozen/native_sample.json')['prompt_ids'])
        for p in ps:
            if p['prompt_id'] in ids:
                for thinking in [False,True]:
                    rows.append(dict(prompt_id=p['prompt_id'],batch='native_thinking' if thinking else 'native_no_thinking',chunk=0,start=0,n=8,messages=p['messages'],schema=p['output_schema'],thinking=thinking))
    else:
        raise ValueError(stage)
    return sorted(rows,key=lambda t:digest([t['prompt_id'],t['batch'],t['chunk']]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--job',required=True)
    ap.add_argument('--stage',required=True,choices=['pilot','main','goal_diagnostic','native'])
    ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args()
    import fcntl
    d=R/'runs'/a.job/a.stage
    d.mkdir(parents=True,exist_ok=True)
    lock=(d/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=next(x for x in load(R/'jobs.json')['jobs'] if x['id']==a.job)
    freeze=load(R/'freeze.json')
    for name,h in freeze['files'].items():
        assert sha(R/'frozen'/name)==h
    todo=tasks(a.stage,freeze)
    base=f'http://127.0.0.1:{job["port"]}'
    metadata=json.load(urllib.request.urlopen(base+'/v1/models',timeout=10))
    server=next(m for m in metadata['data'] if m['id']==job['model'])
    services=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if '--port' in argv and argv[argv.index('--port')+1]==str(job['port']) and 'serve' in argv:
                services.append(dict(pid=int(p.name),argv=argv))
        except (FileNotFoundError,PermissionError,ValueError):
            pass
    assert len(services)==1,'Cannot uniquely identify the inference server'
    modelroot=Path(server['root'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(modelroot),local_files_only=True)
    max_tokens=1408 if a.stage=='native' else 128
    max_len=0
    for t in todo:
        text=tokenizer.apply_chat_template(t['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=t['thinking'])
        n=len(tokenizer.encode(text,add_special_tokens=False))
        assert n+max_tokens<=server['max_model_len'],(t['prompt_id'],n,max_tokens)
        max_len=max(max_len,n)
    # Hash all actual weight files once; the main model is never inferred from its label.
    fp=R/f'checkpoint_{a.job}.json'
    weights=sorted(modelroot.glob('*.safetensors'))
    stats=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in weights]
    assert weights
    if fp.exists():
        checkpoint=load(fp)
        assert checkpoint['stats']==[list(x) for x in stats]
    else:
        checkpoint=dict(stats=stats,sha256={p.name:sha(p) for p in weights},config_sha256=sha(modelroot/'config.json'))
        save(fp,checkpoint)
    identity=dict(protocol=freeze['protocol'],freeze_sha256=sha(R/'freeze.json'),job=job,stage=a.stage,
        runner_sha256=sha(Path(__file__)),checkpoint=checkpoint,server=server,service=services[0],
        max_tokens=max_tokens,max_input_tokens=max_len,n_planned=sum(t['n'] for t in todo),
        plan_sha256=digest(todo),sampling=dict(temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.),
        structured=a.stage!='native',invalid_output_policy='Retain every output; no answer-based retries or rerolls.')
    ip=d/'identity.json'
    if ip.exists():
        assert load(ip)==identity,'Changed runtime identity requires a new experiment directory'
    else:
        save(ip,identity)
    ch=digest(identity)
    raw=d/'responses.jsonl'
    done={}
    if raw.exists():
        with raw.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                key=(row['prompt_id'],row['batch'],row['replicate'])
                assert key not in done and row['config_sha256']==ch
                done[key]=row['schema_valid']
    initial=len(done)
    pending=[]
    for t in todo:
        keys=[(t['prompt_id'],t['batch'],i) for i in range(t['start'],t['start']+t['n'])]
        presence=[k in done for k in keys]
        assert all(presence) or not any(presence),'Partial saved request requires explicit recovery; refusing to reroll'
        if not any(presence):
            pending.append(t)
    start_time=time.time()

    def request(t):
        # Same seed for original/table, and native thinking/no-thinking. Equal seeds do not imply token-level coupling.
        seed_batch='goal' if a.stage=='goal_diagnostic' else 'native' if a.stage=='native' else t['batch']
        seed=int(digest([20261111,t['prompt_id'],seed_batch,t['chunk']])[:8],16)%(2**31-1)
        body=dict(model=job['model'],messages=t['messages'],n=t['n'],temperature=.7,top_p=.95,top_k=-1,
            repetition_penalty=1.,max_tokens=max_tokens,seed=seed,chat_template_kwargs={'enable_thinking':t['thinking']})
        if a.stage!='native':
            body['structured_outputs']={'json':t['schema'],'disable_any_whitespace':True}
        req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body,ensure_ascii=False).encode(),{'Content-Type':'application/json'})
        for attempt in range(4):
            try:
                data=json.load(urllib.request.urlopen(req,timeout=600))
                break
            except urllib.error.HTTPError as e:
                error=e.read().decode(errors='replace')
                if e.code<500:
                    raise RuntimeError(f'HTTP {e.code}: {error[:1000]}') from e
                if attempt==3:
                    raise
                time.sleep(2**attempt)
            except (TimeoutError,urllib.error.URLError):
                if attempt==3:
                    raise
                time.sleep(2**attempt)
        assert data['model']==job['model'] and len(data['choices'])==t['n']
        assert {c['index'] for c in data['choices']}==set(range(t['n']))
        rows=[]
        for choice in sorted(data['choices'],key=lambda c:c['index']):
            text=choice['message'].get('content') or ''
            reasoning=choice['message'].get('reasoning_content') or choice['message'].get('reasoning')
            rows.append(dict(model=a.job,prompt_id=t['prompt_id'],batch=t['batch'],replicate=t['start']+choice['index'],
                response_text=text,reasoning_content=reasoning,finish_reason=choice['finish_reason'],status='ok',
                schema_valid=valid(text,t['schema'],choice['finish_reason']),sampling_seed=seed,
                request_id=data['id'],request_usage=data.get('usage'),choice_index=choice['index'],
                messages_sha256=digest(t['messages']),config_sha256=ch))
        return rows

    with raw.open('a',encoding='utf-8') as f, concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        # Bounded submission prevents the entire pending corpus from becoming queued work after an error.
        iterator=iter(pending)
        active={pool.submit(request,t) for t in [next(iterator,None) for _ in range(a.workers)] if t is not None}
        while active:
            finished,active=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                rows=future.result()
                for row in rows:
                    key=(row['prompt_id'],row['batch'],row['replicate'])
                    assert key not in done
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    done[key]=row['schema_valid']
                f.flush()
                save(d/'progress.json',dict(done=len(done),total=identity['n_planned'],elapsed_seconds=time.time()-start_time,
                    rows_this_session=len(done)-initial,updated_unix=time.time()))
                t=next(iterator,None)
                if t is not None:
                    active.add(pool.submit(request,t))
    expected={(t['prompt_id'],t['batch'],i) for t in todo for i in range(t['start'],t['start']+t['n'])}
    assert set(done)==expected
    save(d/'COMPLETE.json',dict(rows=len(done),sha256=sha(raw),invalid=sum(not v for v in done.values()),
        scope='Generation complete; scientific scores require separate verification',finished_unix=time.time()))
    print(json.dumps(dict(job=a.job,stage=a.stage,rows=len(done),status='generation_complete')),flush=True)


if __name__=='__main__':
    main()
