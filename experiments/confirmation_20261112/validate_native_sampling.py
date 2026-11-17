"""Synthetic controls for complete and missing resamples, never scientific answers."""
import collections,json,os
from pathlib import Path
import numpy as np
from common import R,load,save,sha
from analyze_native_sampling import Panel,behavior,analyze,read,span


def main():
    # A resample may have no legal output. It must become undefined, not zero.
    q=dict(A='a',B='b',C='c',D='d',pair_id='test',source_id='test',task='test')
    missing={'a':[None]*8,'b':[0.]*8,'c':[1.]*8,'d':[1.]*8}
    panel=Panel(missing,np.random.default_rng(1),20)
    p,b=behavior([q],panel,'pair_equal')
    assert np.isnan(p['E']) and span(b['E'])['undefined_draws']==20
    # Shared prompt cancellation must precede all missing/mean calculations.
    shared=dict(q,A='a',B='a',C='c',D='c')
    p,b=behavior([shared],panel,'pair_equal')
    assert p['E']==p['G']==0 and np.all(b['G']==0)
    # A partially missing prompt is resampled as nine attempts, not eight legal draws.
    panel=Panel({'a':[0.,1.,None]*3,'b':[0.,1.]*4+[None]},np.random.default_rng(2),100)
    assert all(np.isnan(panel.data['a'][4]) | ((panel.data['a'][4]>=0)&(panel.data['a'][4]<=1)))
    # Existing full-plan synthetic scoring is linked into a separate fixture.
    src=Path('artifacts/confirmation_20261112/synthetic_prelaunch')
    fixture=Path('artifacts/confirmation_20261112/native_statistics_synthetic_v2')
    assert not fixture.exists();fixture.mkdir(parents=True)
    for p in src.rglob('*'):
        if not p.is_file():continue
        target=fixture/p.relative_to(src);target.parent.mkdir(exist_ok=True,parents=True);os.link(p,target)
    audits=fixture/'local_scalar_audits';audits.mkdir()
    for stage in ['canonical','parameter']:
        p=fixture/'runs/qwen3_14b'/stage/'SCORING_COMPLETE.json'
        (audits/f'qwen3_14b_{stage}.json').write_bytes(p.read_bytes())
    files={str(p.relative_to(fixture)).replace('\\','/'):dict(sha256=sha(p),size=p.stat().st_size) for p in fixture.rglob('*') if p.is_file()}
    save(fixture/'mirror_index.json',dict(files=files,synthetic_fixture_only=True))
    outputs=[]
    for stage in ['canonical','parameter']:
        dest=fixture/'derived_statistics'/stage
        result=analyze('qwen3_14b',stage,fixture,dest,draws=30,subdraws=20)
        assert result['missing_attempts']==0
        summary=read(dest/'sampling_summary.csv')
        for row in summary:
            assert float(row['delta'])==float(row['empirical_low'])==float(row['empirical_high'])==0
            if row['kind']=='goal':
                assert np.isclose(float(row['off_point']),1,atol=1e-12) and np.isclose(float(row['on_point']),1,atol=1e-12)
                assert float(row['Q_pointwise_bound_low'])<0<float(row['Q_pointwise_bound_high'])
        outputs.append(dict(stage=stage,rows=len(summary),result_sha256=sha(dest/'COMPLETE.json')))
    record=dict(status='passed_native_statistics_synthetic_controls',all_missing_is_undefined=True,
        shared_prompt_cancellation_verified=True,all_correct_Q_bound_nonzero=True,constant_empirical_bootstrap_zero_not_invariance_proof=True,
        stages=outputs,fixture=str(fixture),script_sha256=sha(Path(__file__)),analysis_script_sha256=sha(R/'analyze_native_sampling.py'),
        scope='Synthetic test only; excluded from every scientific denominator. No generation or frozen code changed.')
    save(R/'native_statistics_validation.json',record);print(json.dumps(record))


if __name__=='__main__':main()
