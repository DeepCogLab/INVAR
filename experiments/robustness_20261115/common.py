"""Prospectively frozen bilingual extension; older campaigns are read-only."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
REMOTE = '/workspace/project_social_simulation/invar_robustness_runs/20261115'
GPU_UUID = 'GPU-REDACTED'
SEED = 20261115
MODELS = {
    'qwen3_8b': dict(folder='Qwen3-8B', kind='native_toggle', family='Qwen3', role='general_instruct'),
    'socialr1_8b': dict(folder='SocialR1-8B', kind='native_toggle', family='Qwen3', role='social_specialized'),
    'socio_8b': dict(folder='Socio-Foundation-8B', kind='native_toggle', family='Qwen3', role='social_specialized'),
    'gpt_oss_20b': dict(folder='gpt-oss-20b', kind='native_effort', family='gpt_oss', role='general_reasoning'),
    'llama31_8b': dict(folder='Meta-Llama-3.1-8B-Instruct', kind='prompt_reasoning', family='Llama3.1', role='general_instruct'),
    'mistral7b': dict(folder='Mistral-7B-Instruct-v0.3', kind='prompt_reasoning', family='Mistral', role='general_instruct'),
    'glm4_9b': dict(folder='GLM-4-9B-0414', kind='prompt_reasoning', family='GLM4', role='general_instruct'),
    'qwen3_8b_base': dict(folder='Qwen3-8B-Base', kind='base_completion', family='Qwen3', role='pretrained_base'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(p)


def write_jl(path, values):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def arms(job):
    kind = MODELS[job]['kind']
    return ['effort_low', 'effort_high'] if kind == 'native_effort' else (
        ['native_off', 'native_on'] if kind == 'native_toggle' else ['direct', 'brief_reasoning'])


def messages_for(p, kind, arm):
    messages = [dict(m) for m in p['messages']]
    lang = p['language']
    # Common neutral final-answer instruction; only the predeclared prompt
    # intervention may request a scratchpad. No correct labels are present.
    if lang == 'zh':
        system = '依据给定角色、信息和规则作答，不增添题目未给的条件。最终答案必须是题目要求的单个JSON对象。'
        if kind in ('prompt_reasoning', 'base_completion'):
            system += ('直接输出最终JSON，不写解释。' if arm == 'direct' else
                       '先简短核对相关规则并逐步计算，最后在单独一行输出最终JSON。')
    else:
        system = 'Answer using only the given roles, information, and rules. Do not add unstated conditions. The final answer must be the single JSON object requested in the task.'
        if kind in ('prompt_reasoning', 'base_completion'):
            system += (' Give the final JSON directly, without explanation.' if arm == 'direct' else
                       ' First briefly check the relevant rules and calculate step by step; finish with the final JSON on a separate line.')
    messages[0] = dict(role='system', content=system)
    for m in messages[1:]:
        text = m['content']
        for old, new in [('只输出一个JSON对象', '最终答案是一个JSON对象'),
                         ('只输出JSON对象', '最终答案是JSON对象'),
                         ('只输出题目要求的JSON对象', '最终答案是题目要求的JSON对象'),
                         ('Output only one JSON object', 'The final answer is one JSON object'),
                         ('Output only a JSON object', 'The final answer is a JSON object'),
                         ('Only output a JSON object', 'The final answer is a JSON object')]:
            text = text.replace(old, new)
        m['content'] = text
    return messages


def render(tokenizer, job, arm, p):
    kind = MODELS[job]['kind']
    messages = messages_for(p, kind, arm)
    if kind == 'base_completion':
        # Same two format-only examples in both arms; no social payoff labels.
        examples = ('Input: Return action 7.\nOutput: {"action": 7}\n'
                    'Input: Return answer B.\nOutput: {"answer": "B"}\n\n')
        text = examples + '\n\n'.join(m['content'] for m in messages) + '\n\nResponse:\n'
    else:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if kind == 'native_toggle': kwargs['enable_thinking'] = arm == 'native_on'
        if kind == 'native_effort': kwargs['reasoning_effort'] = arm.removeprefix('effort_')
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return text, digest(messages)


def trial_seed(job, pid, arm, replicate, stage='science'):
    # Independent streams across languages, configurations and checkpoints.
    # A shared replicate index is never represented as an identical RNG path.
    return int(digest([SEED, stage, job, pid, arm, replicate])[:8], 16) % (2**31 - 1)
