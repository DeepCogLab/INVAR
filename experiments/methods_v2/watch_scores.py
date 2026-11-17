"""CPU-only scoring whenever a complete immutable method appears."""
import json
import os
import subprocess
import time
from pathlib import Path
from stage_learning import R, V1, BASE, PY, write


def completion_status(finished, normalized, latest, previous):
    """Require the final immutable marker set to have been scored."""
    if finished is None or not normalized:
        return None
    if finished.get('failures'):
        return dict(status='partial_failure', failures=finished['failures'])
    if latest != previous:
        return None
    expected = {(model, method + '.complete.json')
                for model in ['qwen3_8b', 'qwen25_14b_instruct', 'glm4_9b']
                for method in ['baseline', 'canonical_modules', 'canonical_profile']}
    expected.update(('qwen3_8b', arm + '.complete.json')
                    for arm in ['anchor_sft', 'anchor_js', 'anchor_kl', 'anchor_js_kl'])
    actual = {(Path(path).parent.name, Path(path).name) for path, _ in latest}
    assert actual == expected, 'Campaign lacks a planned method or contains an unexpected method'
    return dict(status='completed', completed_methods=len(latest))


if __name__ == '__main__':
    with (R / 'scoring.lock').open('x') as f:
        f.write(str(os.getpid()))
    previous = None
    methods = 'baseline,canonical_modules,canonical_profile,anchor_sft,anchor_js,anchor_kl,anchor_js_kl'
    env = dict(os.environ, INVAR_METHODS_V1=str(V1), OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
    while True:
        signature = [(str(p), p.stat().st_mtime_ns) for p in sorted((R / 'screen').glob('*/*.complete.json'))]
        if signature and signature != previous:
            for script in ['score_methods.py', 'finite_sample_audit.py', 'verify_methods.py']:
                args = [PY, str(R / script), '--baseline', str(BASE), '--manifest', str(V1 / 'split_manifest.json'), '--out', str(R / 'screen')]
                if script != 'verify_methods.py':
                    args += ['--methods', methods]
                with (R / 'logs' / ('score_' + script + '.log')).open('a') as log:
                    result = subprocess.run(args, env=env, cwd=R, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write(R / 'status/scoring_FAILED.json', dict(script=script, exit_code=result.returncode, time_unix=time.time()))
                    raise SystemExit(result.returncode)
            previous = signature
            write(R / 'status/scoring.json', dict(status='updated', completed_methods=len(signature), time_unix=time.time()))
        finished_path = R / 'learning_campaign_finished.json'
        finished = json.loads(finished_path.read_text()) if finished_path.exists() else None
        latest = [(str(p), p.stat().st_mtime_ns) for p in sorted((R / 'screen').glob('*/*.complete.json'))]
        completion = completion_status(finished, (R / 'normalization_complete.json').exists(), latest, previous)
        if completion is not None:
            write(R / 'status/scoring_complete.json', dict(completion, time_unix=time.time()))
            break
        time.sleep(30)
