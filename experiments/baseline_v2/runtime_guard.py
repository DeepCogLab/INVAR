"""Process locks for queue rebalancing; no inference or scoring changes."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def arg(name, default=None):
    return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def locked(root, name, action):
    locks=Path(root)/'locks'
    locks.mkdir(exist_ok=True)
    with (locks/(name+'.lock')).open('a') as handle:
        if os.name=='nt':
            import msvcrt
            if handle.tell()==0:
                handle.write('0');handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                    break
                except OSError:
                    time.sleep(.1)
            try:
                return action()
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX)
            return action()


def guarded_generation(root, action):
    # The runner re-reads saved trial IDs only after obtaining this lock.
    return locked(root,'generate_'+arg('--model')+'_'+arg('--stage','pilot'),action)


def guarded_scoring(root, action):
    root=Path(root)
    model=arg('--model')
    def run():
        cache=root/'locks'/('score_'+model+'.done.json')
        report=root/'reports'/model
        required=[root/'score_v2.py',root/'runtime_guard.py',root/'inputs.jsonl',
                  root/'main_plan.jsonl',root/'frozen_manifest.json',
                  root/'outputs'/'main'/(model+'.jsonl')]
        fingerprint={str(p.relative_to(root)):digest(p) for p in required}
        if cache.exists():
            prior=json.loads(cache.read_text())
            files=prior.get('reports',{})
            if (prior.get('fingerprint')==fingerprint and files and
                all((root/p).is_file() and digest(root/p)==h for p,h in files.items())):
                print(json.dumps(dict(model=model,status='scoring_already_verified')),flush=True)
                return
        action()
        audit=json.loads((report/'completion_audit.json').read_text())
        assert audit['status']=='passed'
        # Auxiliary independent recomputation is produced separately.
        files={str(p.relative_to(root)):digest(p) for p in sorted(report.iterdir())
               if p.is_file() and p.name!='independent_recompute.json'}
        temp=cache.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(fingerprint=fingerprint,reports=files),indent=2))
        temp.replace(cache)
    return locked(root,'score_'+model,run)
