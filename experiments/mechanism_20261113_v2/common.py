"""Additive checkpoint diagnostics; prior experiments are read-only inputs."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
REMOTE = '/workspace/project_social_simulation/invar_mechanism_runs/20261113_v2'
OLD = Path('/workspace/project_social_simulation/invar_methods_runs/20261028')
BASE = Path('/workspace/project_social_simulation/invar_reliability_runs/20261024_v2')
SOURCES = ['HPVBG', 'IGVOFO', 'PGG', 'PIQ572']
MODELS = {'qwen3_8b': dict(path='/mnt/models/models/Qwen3-8B', gpu=1),
          'qwen25_14b_instruct': dict(path='/mnt/models/models/Qwen2.5-14B-Instruct', gpu=0)}
STATES = ['baseline'] + [f'{arm}_{step}' for step in (50, 100, 150) for arm in ('anchor_sft', 'anchor_js')]


def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def jl(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'); t.replace(p)


def write_jl(p, rows):
    p = Path(p); assert not p.exists(), p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(canonical(x) + '\n' for x in rows), encoding='utf-8')


def reps(row): return 64 if row['task_type'] == 'behavior' else 16
def seed(job, source, state, pid, rep): return int(digest([20261113, 'mechanism_v1', job, source, state, pid, rep])[:8], 16) % (2**31-1)
def panel(source): return R / 'frozen' / source
def out(job): return R / 'runs' / job


def verify_freeze():
    cfg = load(R / 'freeze.json')
    for name, h in cfg['files_sha256'].items(): assert sha(R / name) == h, name
    for name, h in cfg['external_code_sha256'].items():
        path=Path(name)
        if not path.exists(): path=R/'vendor/external'/f'{h}.py'
        assert sha(path) == h, name
    return cfg


def adapter(job, source, state):
    if state == 'baseline': return None
    return load(R / 'adapter_inventory.json')[job][source][state]
