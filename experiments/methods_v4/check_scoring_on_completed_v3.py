"""CPU regression against immutable completed V3 data, including a no-goal fold.

All fixtures are labelled analysis projections, not new model calls/results.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

R=Path(__file__).resolve().parent
OLD=Path(os.environ.get('INVAR_METHODS_V3',R.parent/'methods_v3'))
BASE=Path(os.environ.get('INVAR_BASELINE',R.parent/'baseline_v2'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parent=json.loads((OLD/'precision_manifest.json').read_text(encoding='utf-8'))
    for scope in ['all_v3','PGG_no_goal']:
        out=R/'cpu_checks/scoring_regression'/scope
        out.mkdir(exist_ok=True,parents=True)
        selected=set(parent['selected_prompt_ids'])
        if scope=='PGG_no_goal':
            selected={pid for pid in selected if parent['provenance_sources'][pid]==['PGG']}
        inputs={json.loads(s)['prompt_id']:json.loads(s) for s in (BASE/'inputs.jsonl').read_text(encoding='utf-8-sig').splitlines()}
        manifest=dict(parent,scope='CPU_existing_V3_fixture_'+scope,selected_prompt_ids=sorted(selected),
                      initial_models=['qwen3_8b'],model_methods={'qwen3_8b':['baseline','anchor_sft','anchor_js']},
                      calls_per_model_method=sum(parent['repetitions'][inputs[pid]['task_type']] for pid in selected))
        if scope=='PGG_no_goal':
            pairs={p['pair_id'] for p in parent['selected_pair_details'] if p['source_id']=='PGG'}
            manifest['selected_pairs']=sorted(pairs)
            manifest['selected_pair_details']=[p for p in parent['selected_pair_details'] if p['pair_id'] in pairs]
            paths={'behavior':'INVAR_dataset/data/quartets.jsonl','fact':'fact_quartets.jsonl','goal':'goal_quartets.jsonl'}
            manifest['selected_quartet_ids']={k:[q['quartet_id'] for s in (BASE/p).read_text(encoding='utf-8-sig').splitlines()
                if (q:=json.loads(s))['pair_id'] in pairs] for k,p in paths.items()}
            assert not manifest['selected_quartet_ids']['goal']
        mp=out/'fixture_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
        target=out/'qwen3_8b';target.mkdir(exist_ok=True)
        for method in ['baseline','anchor_sft','anchor_js']:
            src=OLD/'screen/qwen3_8b'/(method+'.jsonl');dst=target/src.name
            mark=json.loads(src.with_suffix('.complete.json').read_text(encoding='utf-8'))
            assert sha(src)==mark['responses_sha256']
            identity=json.loads(src.with_suffix('.run_identity.json').read_text(encoding='utf-8'))
            if not dst.exists():
                if scope=='all_v3':os.link(src,dst)
                else:
                    with src.open('rb') as a,dst.open('wb') as b:
                        for line in a:
                            if json.loads(line)['prompt_id'] in selected:b.write(line)
            projection=dict(kind='CPU_regression_existing_V3_not_new_generation',parent_raw_sha256=sha(src))
            mark.update(rows=manifest['calls_per_model_method'],responses_sha256=sha(dst),analysis_projection=projection)
            identity['plan']['selected_prompt_ids']=sorted(selected)
            identity['plan']['planned_rows']=manifest['calls_per_model_method']
            identity['plan']['source_partitions']={k:v for k,v in identity['plan']['source_partitions'].items() if k in selected}
            identity['analysis_projection']=projection
            dst.with_suffix('.complete.json').write_text(json.dumps(mark),encoding='utf-8')
            dst.with_suffix('.run_identity.json').write_text(json.dumps(identity),encoding='utf-8')
        subprocess.run([sys.executable,str(R/'score_transfer.py'),'--baseline',str(BASE),'--manifest',str(mp),
                        '--out',str(out),'--model','qwen3_8b'],check=True)
    print(json.dumps(dict(scoring_regression='passed',scopes=['full_V3','PGG_without_goal'],no_new_model_calls=True)))


if __name__=='__main__':main()
