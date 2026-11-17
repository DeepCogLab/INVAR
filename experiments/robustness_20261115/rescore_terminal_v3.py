"""CPU-only scoring correction for every completed checkpoint, without regeneration."""
import argparse
import collections
import json
from datetime import datetime,timezone
from pathlib import Path
import score as frozen_score
import analyze as frozen_analysis
import parsing_v3
from parsing import parse_final as v1
from parsing_v2 import parse_final as v2
from campaign_common import R,load,save,sha,jl
from rescore_harmony_v2 import link_same

MIRROR=Path('artifacts/robustness_20261115/remote')
STAGE=Path('artifacts/robustness_20261115/corrected_v3')


def main(job):
    cfg=load(R/'freeze.json');source=MIRROR/'runs'/job;target=STAGE/'runs'/job
    assert job in cfg['models']
    assert sha(source/'responses.jsonl')==load(source/'COMPLETE.json')['raw_sha256']
    for name in list(cfg['files'])+['freeze.json']:link_same(R/name,STAGE/name)
    for name in ['identity.json','COMPLETE.json','responses.jsonl']:link_same(source/name,target/name)
    manifest_path=STAGE/'CORRECTION_MANIFEST.json'
    if not manifest_path.exists():
        save(manifest_path,dict(version='terminal_line_and_harmony_v3',created_at_utc=datetime.now(timezone.utc).isoformat(),
            original_scientific_freeze_sha256=sha(R/'freeze.json'),original_parser_sha256=sha(R/'parsing.py'),
            corrected_parser_sha256=sha(R/'parsing_v3.py'),harmony_helper_sha256=sha(R/'parsing_v2.py'),
            correction_runner_sha256=sha(Path(__file__)),new_model_generations=0,
            rationale='Frozen ^\\s* line regex consumed preceding blank lines and counted one terminal JSON more than once. Restrict to horizontal whitespace. Includes accepted Harmony v2 metadata correction.',
            scope='All eight checkpoints, both arms/languages, all task types; same completed raw, labels, all-attempt denominators and 500 resamples; no answer-based selection.'))
    manifest=load(manifest_path)
    assert manifest['corrected_parser_sha256']==sha(R/'parsing_v3.py')
    assert manifest['harmony_helper_sha256']==sha(R/'parsing_v2.py')
    assert manifest['correction_runner_sha256']==sha(Path(__file__))
    if (target/'CORRECTION_COMPLETE.json').exists():
        done=load(target/'CORRECTION_COMPLETE.json')
        assert done['manifest_sha256']==sha(manifest_path)
        assert done['corrected_score_sha256']==sha(target/'SCORING_COMPLETE.json')
        print(json.dumps(dict(job=job,status='existing_v3_preserved')));return
    ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')};identity=load(source/'identity.json')
    changes=[];counts=collections.Counter();v2_changes=0
    raw_lines=(source/'responses.jsonl').open(encoding='utf-8')
    for row in (json.loads(line) for line in raw_lines):
        p=ps[row['prompt_id']];key=row['prompt_id']+'__'+row['arm']
        args=(row['response_text'],row['response_with_special_tokens'],row['finish_reason'],p['output_schema'],cfg['models'][job]['kind'],row['arm'],identity['prefilled_think'][key])
        old=v1(*args);previous=v2(*args);new=parsing_v3.parse_final(*args)
        if old[0] is not None:assert old==new
        if previous!=new:
            assert previous[0] is None and previous[1]=='format' and new[0] is not None
            v2_changes+=1
        if old!=new:
            assert old[0] is None and old[1] in ['format','missing_or_ambiguous_final_channel']
            changes.append(dict(prompt_id=row['prompt_id'],arm=row['arm'],replicate=row['replicate'],language=row['language'],task_type=row['task_type'],old_error=old[1],new_error=new[1],now_delivered=new[0] is not None))
            counts[row['arm']+'__'+row['language']+'__'+row['task_type']]+=1
    raw_lines.close()
    frozen_score.R=STAGE;frozen_score.parse_final=parsing_v3.parse_final
    result=frozen_score.score(job);result['scoring_revision']=manifest;save(target/'SCORING_COMPLETE.json',result)
    frozen_analysis.R=STAGE;frozen_analysis.analyze_job(job,cfg['protocol']['mc_resamples']);frozen_analysis.report()
    save(target/'CHANGED_CLASSIFICATIONS.json',dict(rows=changes,counts=dict(counts),total_changed_from_v1=len(changes),total_changed_from_harmony_v2=v2_changes))
    save(target/'CORRECTION_COMPLETE.json',dict(status='rescored_and_resampled_original_raw_v3',manifest_sha256=sha(manifest_path),
        corrected_score_sha256=sha(target/'SCORING_COMPLETE.json'),original_score_sha256=sha(source/'SCORING_COMPLETE.json'),
        changed_rows_from_v1=len(changes),changed_rows_from_v2=v2_changes,raw_sha256=sha(source/'responses.jsonl'),new_generations=0))
    print(json.dumps(dict(job=job,status='corrected_v3_completed',changed_rows=len(changes),changed_from_v2=v2_changes,by_cell=dict(counts))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);main(ap.parse_args().job)
