import hashlib,json,struct
from pathlib import Path
r=Path('/workspace/project_social_simulation/invar_social_cognition_runs/20261104');m=Path('/workspace/models')
inventory=json.loads((r/'inventory.json').read_text());result=[]
cards=r/'model_cards';cards.mkdir(exist_ok=True)
for entry in inventory:
    item=dict(id=entry['id'],base=entry['base'],adapter=entry['adapter'])
    for name in [entry['base'],entry['adapter']]:
        if not name:continue
        card=m/name/'README.md'
        if card.exists():
            raw=card.read_bytes();(cards/(name+'.md')).write_bytes(raw)
            item[name+'_card_sha256']=hashlib.sha256(raw).hexdigest()
    if entry['adapter']:
        p=m/entry['adapter']/'adapter_model.safetensors'
        if p.exists():
            with p.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
            tensors={k:v['shape'] for k,v in header.items() if k!='__metadata__'}
            item['adapter_tensors']=len(tensors);item['unexpected_non_lora_keys']=[k for k in tensors if 'lora_' not in k]
    result.append(item)
(r/'model_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps([dict(id=i['id'],unexpected=i.get('unexpected_non_lora_keys',[])) for i in result]))
