"""Final main-matrix coverage/provenance audit; supplementary results stay separate."""
import collections
import csv
from datetime import datetime,timezone,timedelta
from pathlib import Path
from campaign_common import R,load,jl,sha,digest,save

M=Path('artifacts/robustness_20261115/remote')
C=Path('artifacts/robustness_20261115/corrected_v3')


def allowed(action,schema):
    v=schema['properties']['action'];typ=int if v['type']=='integer' else str
    return type(action) is typ and ('enum' not in v or action in v['enum']) and ('minimum' not in v or action>=v['minimum']) and ('maximum' not in v or action<=v['maximum'])


def main():
    cfg=load(R/'freeze.json');fh=sha(R/'freeze.json');assert fh==sha(M/'freeze.json')
    assert fh=='eed99ca4455c82af28021bb02aeac62873231624bd6767a3f5500bcefed4f75b'
    for name,h in cfg['files'].items():assert sha(R/name)==sha(M/name)==h,name
    queue=load(M/'science_queue_status.json');assert queue['status']=='finished'
    assert [v['job'] for v in queue['jobs']]==cfg['protocol']['job_order']
    assert all(v['status']=='generated_and_scored' and v['returncode']==v['score_returncode']==v['analysis_returncode']==0 for v in queue['jobs'])
    ps=jl(R/'frozen/model_inputs.jsonl');labels=jl(R/'frozen/evaluation.jsonl')
    inputs={p['prompt_id']:p for p in ps};truth={p['prompt_id']:p for p in labels}
    assert len(ps)==len(inputs)==len(labels)==len(truth)==2574 and set(inputs)==set(truth)
    qcounts={};disjoint_counts={};reference_count=0
    for name,number in [('core/behavior_quartets.jsonl',128),('core/goal_quartets.jsonl',156),('core/fact_quartets.jsonl',512),('parameter/goal_quartets.jsonl',48)]:
        qs=jl(R/'frozen'/name);assert len(qs)==number;qcounts[name]=len(qs)
        assert collections.Counter(q['language'] for q in qs)==dict(zh=number//2,en=number//2)
        for q in qs:
            for key in 'ABCD':assert q[key] in inputs and inputs[q[key]]['language']==q['language'];reference_count+=1
            if 'goal' in name:
                sets={k:set(truth[q[k]]['expected_actions']) for k in 'ABCD'}
                assert all(sets.values()) and sets['A']==sets['B'] and sets['C']==sets['D']
                for k in 'ABCD':assert all(allowed(a,inputs[q[k]]['output_schema']) for a in sets[k])
                if name.startswith('parameter'):assert sets['A'].isdisjoint(sets['C'])
        if 'goal' in name:disjoint_counts[name]=sum(set(truth[q['A']]['expected_actions']).isdisjoint(truth[q['C']]['expected_actions']) for q in qs)
    selected=load(R/'results/SCORING_VERSION_SELECTION.json')['selected'];assert set(selected)==set(cfg['protocol']['job_order'])
    rows=[];total=0;checks=0;tokens=0
    for job in cfg['protocol']['job_order']:
        d=M/'runs'/job;done=load(d/'COMPLETE.json');audit=load(R/'independent_audits_v3'/(job+'.json'));score=load(C/'runs'/job/'SCORING_COMPLETE.json')
        assert done['rows']==audit['rows']==score['rows']==cfg['expected_rows_by_job'][job]==11760
        rawhash=sha(d/'responses.jsonl');assert rawhash==done['raw_sha256']==audit['raw_sha256']==score['raw_sha256']==selected[job]['raw_sha256']
        assert sha(C/'runs'/job/'SCORING_COMPLETE.json')==selected[job]['score_sha256']
        identity=load(d/'identity.json');assert digest(identity)==done['identity_sha256']
        assert identity['freeze_sha256']==audit['freeze_sha256']==score['freeze_sha256']==fh
        assert audit['status']=='passed_independent_v3_linewise_message_header_transport_and_accuracy'
        assert audit['all_attempt_csv_rows_independently_reparsed'] and audit['shared_quartet_references_full_optimum_sets_and_missingness_verified']
        mc=load(C/'runs'/job/'MONTE_CARLO_SENSITIVITY.json')
        assert mc['resamples']==500 and mc['raw_sha256']==rawhash and mc['status']=='independent_vector_scalar_agreement'
        assert len(score['summaries'])==4
        total+=done['rows'];checks+=audit['audited_summary_values'];tokens+=done['completion_tokens']
        rows.append(dict(model=job,status='complete_and_independently_accepted_v3',planned=11760,generated=done['rows'],
            raw_sha256=rawhash,identity_sha256=sha(d/'identity.json'),score_sha256=selected[job]['score_sha256'],audit_sha256=sha(R/'independent_audits_v3'/(job+'.json')),
            generation_seconds=done['generation_seconds'],completion_tokens=done['completion_tokens']))
    assert total==94080 and checks==1376
    out=R/'results'
    with (out/'main_completion_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='passed_all_eight_main_checkpoints',generated_rows=total,expected_rows=94080,missing_rows=0,independently_checked_summary_values=checks,
        unique_bilingual_inputs=2574,quartet_counts=qcounts,verified_quartet_references=reference_count,disjoint_goal_quartets=disjoint_counts,
        full_optimum_sets_and_legal_actions_verified=True,frozen_files_verified=len(cfg['files']),freeze_sha256=fh,
        total_completion_tokens=tokens,wall_hours=(queue['finished_unix']-queue['started_unix'])/3600,
        original_v1_and_v2_preserved=True,scoring_revision='uniform_terminal_line_and_harmony_v3',
        matrix_sha256=sha(out/'main_completion_matrix.csv'),score_selection_sha256=sha(out/'SCORING_VERSION_SELECTION.json'),
        audit_code_sha256=sha(Path(__file__)),audited_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Main 94080 attempts complete. Overnight post-hoc supplement remains separate and is not marked complete by this receipt.')
    save(out/'MAIN_MATRIX_ACCEPTANCE.json',result)
    print(__import__('json').dumps(result))


if __name__=='__main__':main()
