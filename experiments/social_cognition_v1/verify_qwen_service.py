import concurrent.futures,json,time,urllib.request
from pathlib import Path
D=Path('/workspace/project_social_simulation/model_services/qwen25_7b_20261104')
def get(port,path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=5) as f:return f.status,f.read().decode()
def trial(port):
    payload=dict(model='Qwen2.5-7B-Instruct',messages=[dict(role='user',content='19加23等于多少？只输出数字。')],temperature=0,max_tokens=16)
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.time()
    with urllib.request.urlopen(req,timeout=90) as f:result=json.loads(f.read())
    answer=result['choices'][0]['message']['content'];assert answer.strip()=='42',answer
    return dict(port=port,answer=answer,seconds=round(time.time()-start,3),usage=result['usage'])
results=[]
for port in [18130,18131]:
    status,body=get(port,'/health');assert status==200
    _,body=get(port,'/v1/models');models=json.loads(body)['data'];assert any(m['id']=='Qwen2.5-7B-Instruct' and m['max_model_len']==32768 for m in models)
    results.append(dict(port=port,health='passed',models=models))
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    trials=list(pool.map(trial,[18130]*8+[18131]*8))
data=dict(status='ready',checked=time.time(),endpoints=results,concurrent_generation_checks=trials)
(D/'verification.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
deployment=json.loads((D/'deployment.json').read_text());deployment['status']='ready';deployment['verified_at']=time.time()
(D/'deployment.json').write_text(json.dumps(deployment,ensure_ascii=False,indent=2))
print(json.dumps(dict(status='ready',health_ports=[18130,18131],successful_concurrent_requests=len(trials))))
