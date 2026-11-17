"""Validate actual saved V4 checkpoints before long generation, without CUDA."""
import argparse
import importlib.util
import json
import math
import hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v4_actual_checkpoint_validator',R/'run_transfer_evaluation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--include-final',action='store_true');a=p.parse_args()
    results=[]
    candidates=list((R/'learning').rglob('adapter_checkpoint_025'))
    if a.include_final:
        candidates += [f.parent/'adapter' for f in (R/'learning').rglob('training_complete.json')]
    for adapter in candidates:
        final=adapter.name=='adapter'
        if not final and not (adapter/'checkpoint_metadata.json').is_file():continue
        config=json.loads((adapter.parent/'training_config.json').read_text(encoding='utf-8'))
        identity=m.adapter_identity(adapter,config['arm'],config['model_id'],Path(config['model']),
                                    a.baseline,m.V2/'teacher_cache')
        assert identity['training_identity']['adapter_checkpoint_step']==(150 if final else 25)
        if final:
            logs=[json.loads(s) for s in (adapter.parent/'training_log.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
            assert [r['step'] for r in logs]==list(range(1,151))
            assert all(math.isfinite(float(v)) for row in logs for v in row.values() if isinstance(v,(int,float)))
            actual=json.loads((adapter.parent/'actual_exposures.json').read_text(encoding='utf-8'))
            assert actual['total_prompt_exposures']==4800
            identity['final_training_log_audit']=dict(optimizer_steps=150,all_numeric_log_values_finite=True,
                                                      actual_prompt_exposures=4800)
        results.append(identity)
    out=R/'cpu_checks/actual_checkpoint_validation.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(status='actual_checkpoint_metadata_verified_not_generation',
                    checked=len(results),include_final=a.include_final,
                    validator_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    checked_final_adapters=sum('final_training_log_audit' in r for r in results),
                    identities=results),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(checked_actual_checkpoints=len(results),models=[r['training_identity']['model_id'] for r in results],
                          no_generation_claim=True)))
