"""Independent engineering pilot. No scientific scoring or answer-based retries."""
import argparse
import importlib.metadata
import os
import time
from pathlib import Path

from common import R, MODELS, GPU_UUID, arms, render, digest, sha, save, trial_seed


def prompts():
    result = []
    for language in ('zh', 'en'):
        for i in range(8):
            total = 120 + 12*i; threshold = 36 + 3*i
            if language == 'zh':
                body = (f'你是分配者P1，P2是接收者。共有{total}点，只能选择给P2 0、{threshold}或{total}点。'
                        f'若给出少于{threshold}点，两人最终都得到0；否则P1得到{total}减给出额，P2得到给出额。'
                        'P2不能拒绝或更改，不存在其他收益。请只最大化P1的最终收益。'
                        '最终答案为一个仅有action字段的JSON对象，action是所列合法整数之一。')
            else:
                body = (f'You are allocator P1; P2 is the recipient. There are {total} points. You may give P2 only 0, {threshold}, or {total} points. '
                        f'If the amount is below {threshold}, both receive 0; otherwise P1 receives {total} minus that amount and P2 receives that amount. '
                        'P2 cannot reject or change it. There are no other payoffs. Maximize only P1\'s final payoff. '
                        'The final answer must be one JSON object with only the action field, containing one of the listed integers.')
            result.append(dict(prompt_id=f'engineering_{language}_{i}', language=language, task_type='goal',
                               messages=[dict(role='system',content=''),dict(role='user',content=body)]))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    import subprocess
    actual_uuid=subprocess.run(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    assert actual_uuid==GPU_UUID
    job=MODELS[a.job];model=Path('/mnt/models/models')/job['folder']
    out=R/'engineering_v2'/a.job;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(), 'Do not rerun a completed pilot'
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True)
    ps=prompts();texts={};message_hashes={}
    for p in ps:
        for arm in arms(a.job):
            text,mh=render(tokenizer,a.job,arm,p);texts[p['prompt_id'],arm]=text;message_hashes[p['prompt_id'],arm]=mh
    assert all(texts[p['prompt_id'],arms(a.job)[0]]!=texts[p['prompt_id'],arms(a.job)[1]] for p in ps)
    save(out/'template_preflight.json',dict(status='arms_render_differently',
        rendered_sha256={pid+'__'+arm:digest(text) for (pid,arm),text in texts.items()},
        tokenizer_config_sha256=sha(model/'tokenizer_config.json'),kind=job['kind']))
    start=time.time()
    llm=LLM(model=str(model),dtype='bfloat16',max_model_len=8192,gpu_memory_utilization=.86,
        max_num_seqs=64,max_num_batched_tokens=16384,enforce_eager=True,
        enable_prefix_caching=True,generation_config='vllm',seed=20261115,trust_remote_code=True)
    loaded=time.time();plan=[(p,arm) for arm in arms(a.job) for p in ps]
    params=[SamplingParams(temperature=.7,top_p=.95,top_k=-1,max_tokens=4096,n=1,
             seed=trial_seed(a.job,p['prompt_id'],arm,0,'engineering')) for p,arm in plan]
    outputs=llm.generate([texts[p['prompt_id'],arm] for p,arm in plan],params,use_tqdm=False)
    import json
    with (out/'responses.jsonl').open('x',encoding='utf-8') as f:
        for (p,arm),g,sp in zip(plan,outputs,params):
            o=g.outputs[0]
            row=dict(model=a.job,language=p['language'],prompt_id=p['prompt_id'],arm=arm,
                response_text=o.text,response_with_special_tokens=tokenizer.decode(o.token_ids,skip_special_tokens=False),
                token_ids=list(o.token_ids),completion_tokens=len(o.token_ids),finish_reason=o.finish_reason,
                sampling_seed=sp.seed,messages_sha256=message_hashes[p['prompt_id'],arm],
                rendered_sha256=digest(texts[p['prompt_id'],arm]))
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    elapsed=time.time()-loaded
    tokens=sum(len(g.outputs[0].token_ids) for g in outputs)
    save(out/'COMPLETE.json',dict(status='engineering_completed_not_scientific',rows=len(plan),
        load_seconds=loaded-start,generation_seconds=elapsed,completion_tokens=tokens,tokens_per_second=tokens/elapsed,
        raw_sha256=sha(out/'responses.jsonl'),max_tokens=4096,
        software={n:importlib.metadata.version(n) for n in ['vllm','torch','transformers']},
        criterion='compatibility and final-delivery presence only, never objective accuracy; no rerolls'))


if __name__=='__main__':main()
