"""Independent confirmation utilities; older runs are never writable targets."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
REMOTE = '/workspace/project_social_simulation/invar_confirmation_runs/20261112'
MODELS = {'qwen3_14b': dict(path='/mnt/models/models/Qwen3-14B', dtype='bfloat16', gpu=1),
          'qwen3_30b_a3b_gptq': dict(path='/mnt/models/models/Qwen3-30B-A3B-GPTQ-Int4', dtype='float16', quantization='gptq_marlin', gpu=0)}


def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def jl(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return [json.loads(s) for s in f if s.strip()]


def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8'); t.replace(p)


def write_jl(p, rows):
    p = Path(p); assert not p.exists(), p
    with p.open('w', encoding='utf-8') as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')


def seed(stage, pid, rep):
    return int(digest([20261112, 'confirmation', stage, pid, rep])[:8], 16) % (2**31 - 1)


def repetitions(p): return 128 if p['task_type'] == 'behavior' else 8


def tasks(root, stage):
    ps = {p['prompt_id']: p for p in jl(root / 'frozen' / stage / 'model_inputs.jsonl')}
    return ps, [(pid, arm, rep) for arm in ['native_no_thinking', 'native_thinking']
                for pid in sorted(ps) for rep in range(repetitions(ps[pid]))]
