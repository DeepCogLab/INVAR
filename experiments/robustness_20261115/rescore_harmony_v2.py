"""Local CPU-only erratum. Original frozen code, scores and raw stay intact."""
import json
import os
import shutil
from datetime import datetime,timezone
from pathlib import Path
import score as frozen_score
import analyze as frozen_analysis
import parsing_v2
from campaign_common import R,load,save,sha,jl,digest

MIRROR=Path('artifacts/robustness_20261115/remote')
STAGE=Path('artifacts/robustness_20261115/corrected_v2')


def link_same(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst)
    elif src.drive.casefold()==dst.drive.casefold():os.link(src,dst)
    else:shutil.copyfile(src,dst)


def main():
    cfg=load(R/'freeze.json');job='gpt_oss_20b';source=MIRROR/'runs'/job
    assert sha(source/'responses.jsonl')==load(source/'COMPLETE.json')['raw_sha256']
    for name in list(cfg['files'])+['freeze.json']:link_same(R/name,STAGE/name)
    for name in ['identity.json','COMPLETE.json','responses.jsonl']:link_same(source/name,STAGE/'runs'/job/name)
    target=STAGE/'runs'/job
    assert not (target/'CORRECTION_COMPLETE.json').exists(),'Do not overwrite accepted correction'
    manifest=dict(version='harmony_final_header_v2',created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_scientific_freeze_sha256=sha(R/'freeze.json'),original_raw_sha256=sha(source/'responses.jsonl'),
        original_score_sha256=sha(source/'SCORING_COMPLETE.json'),original_parser_sha256=sha(R/'parsing.py'),
        corrected_parser_sha256=sha(R/'parsing_v2.py'),correction_runner_sha256=sha(Path(__file__)),
        rationale='Final-channel JSON/json metadata was falsely treated as missing final channel. Restore declared final-channel semantics, keeping schema/type/menu/truncation and all-attempt denominators.',
        source='https://developers.openai.com/cookbook/articles/openai-harmony',new_model_generations=0,
        scope='Both effort arms, both languages, all task types; no answer-based row selection.')
    save(STAGE/'CORRECTION_MANIFEST.json',manifest)
    frozen_score.R=STAGE;frozen_score.parse_final=parsing_v2.parse_final
    result=frozen_score.score(job)
    result['scoring_revision']=manifest
    save(target/'SCORING_COMPLETE.json',result)
    frozen_analysis.R=STAGE;frozen_analysis.analyze_job(job,cfg['protocol']['mc_resamples']);frozen_analysis.report()
    # Verify the precise set of classification changes against the original,
    # without changing a response, action, type, numeric menu or truth label.
    ps={p['prompt_id']:p for p in jl(R/'frozen/model_inputs.jsonl')};identity=load(source/'identity.json')
    changes=[]
    from parsing import parse_final as v1
    import collections
    counts=collections.Counter()
    with (source/'responses.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);p=ps[row['prompt_id']];key=row['prompt_id']+'__'+row['arm']
            args=(row['response_text'],row['response_with_special_tokens'],row['finish_reason'],p['output_schema'],'native_effort',row['arm'],identity['prefilled_think'][key])
            old=v1(*args);new=parsing_v2.parse_final(*args)
            if old!=new:
                assert old[0] is None and old[1]=='missing_or_ambiguous_final_channel'
                assert parsing_v2.canonical_harmony_final(row['response_with_special_tokens'])!=row['response_with_special_tokens']
                changes.append(dict(prompt_id=row['prompt_id'],arm=row['arm'],replicate=row['replicate'],language=row['language'],task_type=row['task_type'],old_error=old[1],new_error=new[1],now_delivered=new[0] is not None))
                counts[row['arm']+'__'+row['language']]+=1
    save(target/'CHANGED_CLASSIFICATIONS.json',dict(rows=changes,counts=dict(counts),total_changed=len(changes)))
    save(target/'CORRECTION_COMPLETE.json',dict(status='rescored_and_resampled_original_raw',manifest_sha256=sha(STAGE/'CORRECTION_MANIFEST.json'),
        corrected_score_sha256=sha(target/'SCORING_COMPLETE.json'),changed_rows=len(changes),raw_sha256=sha(source/'responses.jsonl'),new_generations=0))
    print(json.dumps(dict(status='corrected_v2_completed',changed_rows=len(changes),by_cell=dict(counts))))


if __name__=='__main__':main()
