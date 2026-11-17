"""Freeze once after CPU checks; no new scores may exist."""
import importlib.metadata
import shutil
from common import *


def main():
    assert not (R/'freeze.json').exists() and not (R/'runs').exists()
    assert load(R/'preflight.json')['status']=='passed_cpu_native_and_identity_checks'
    assert load(R/'synthetic_validation.json')['status']=='passed'
    assert load(R/'synthetic_pipeline_validation.json')['status']=='passed'
    assert load(R/'synthetic_adapter_validation.json')['status']=='passed_cpu_adapter_switch_and_disable'
    assert load(R/'revision_from_v1.json')['status']=='root_cause_reproduced_no_scientific_raw_overwritten'
    plan=load(R/'plan.json');assert plan['science_generated_rows']==309120 and plan['proxy_prompt_distributions']==5754
    externals=[OLD/'train_transfer.py', OLD/'run_transfer_evaluation.py', OLD/'verify_methods.py',
        OLD.parent/'20261026/train_condition_anchor.py', OLD.parent/'20261026/run_anchor_evaluation.py',
        OLD.parent/'20261025/train_lora.py', OLD.parent/'20261025/run_prompt_methods.py']
    external={str(p):sha(p) for p in externals}
    for p in externals:
        dest=R/'vendor/external'/f'{external[str(p)]}.py';dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists();shutil.copyfile(p,dest)
    protocol=dict(protocol='INVAR_fixed_proxy_checkpoint_mechanism_20261113_v2',
        authorization='2026-11-13 user: 继续，看看还能做什么实验，在做一做',
        plan=plan,training_seed=20261026,checkpoint_steps=[50,100,150],new_training_runs=0,
        selection='16 edges and 8 goals per source/task-balanced SHA-selected stratum; use all if fewer',
        same_prompts_for_all_states=True,expected_labels_separate=True,
        generation=dict(temperature=.7,top_p=.95,top_k=-1,repetition_penalty=1.,n=1,max_tokens=128,
            context=4096,behavior_n=64,goal_n=16,decoder='xgrammar_json_schema_compact',
            native_template=True,qwen3_enable_thinking=False),
        proxy='normalized complete assistant JSON plus native end-of-turn likelihood, frozen training helper, eval no dropout',
        statistics=dict(bootstrap=500,include_missing_bin=True,shared_prompt_counts_reused=True,
            undefined_not_zero=True,empirical_not_calibrated_CI=True,downsample_n=[16,32,64],
            downsample_repeats=100,missing_quality_separate=True,
            primary_contrasts='JS minus SFT at each checkpoint and each stratum, W1 on identical action support',
            no_best_checkpoint_selection=True,no_outcome_budget_adjustment=True),
        limits=['retrospective observed-source small-menu diagnostic; one preselected old seed',
            'heldout-variant edges can share original endpoints with training',
            'same-menu W1 comparison does not isolate individual decoding components',
            'no canonical aggregate G/R or free-behavior normative correctness claim'])
    save(R/'protocol.json',protocol)
    names=[p for p in R.rglob('*') if p.is_file() and any(x in p.relative_to(R).parts for x in ('frozen','vendor','code_history'))]
    names += [p for p in R.iterdir() if p.suffix in ('.py','.md','.json') and p.name!='freeze.json']
    save(R/'freeze.json',dict(protocol=protocol['protocol'],models=MODELS,
        files_sha256={str(p.relative_to(R)):sha(p) for p in sorted(set(names))},external_code_sha256=external,
        environment_versions={x:importlib.metadata.version(x) for x in ('torch','transformers','peft','vllm','xgrammar')},
        scientific_scores_present=False,new_training_runs=0))
    print('freeze SHA256 '+sha(R/'freeze.json'))


if __name__=='__main__':main()
