"""Read-only frozen-input, independent oracle, parser, metric and template checks."""
import argparse
import collections
import sys
from pathlib import Path
from common import R,load,jl,save,sha,tasks
from parameter_cases import build,independent_payoffs,optima


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokenize',action='store_true');a=ap.parse_args()
    cfg=load(R/'freeze.json')
    for name,h in {**cfg['files'],**cfg['runtime_sha256']}.items():assert sha(R/name)==h
    sys.path.insert(0,str(R/'vendor'))
    from audit_new_runs import parse_final
    from score_joint_reasoning_v2 import metrics
    schema=dict(properties=dict(action=dict(type='integer',enum=[0,1000,2000])))
    assert parse_final('<think>检查</think>\n{"action":1000}','stop',schema)[0]=={'action':1000}
    assert parse_final('{"action":1200}','stop',schema)[1]=='menu'
    assert parse_final('{"action":"1000"}','stop',schema)[1]=='type'
    assert parse_final('{"action":1000}','length',schema)[1]=='truncated'
    assert parse_final('<think>未闭合 {"action":1000}','stop',schema)[1]=='incomplete_reasoning'
    q=dict(A='a',B='b',C='c',D='d')
    fixed=metrics(q,dict(a=[0]*8,b=[0]*8,c=[0]*8,d=[0]*8),8)
    responsive=metrics(q,dict(a=[0]*8,b=[0]*8,c=[1]*8,d=[1]*8),8)
    assert fixed['E_conditional']==fixed['G_conditional']==fixed['R_conditional']==0
    assert responsive['E_conditional']==responsive['G_conditional']==0 and responsive['R_conditional']==1
    # This catches mixing rule conditions into equivalent-presentation edges.
    assert responsive['R_conditional']>fixed['R_conditional']
    labels={p['prompt_id']:p for p in jl(R/'frozen/parameter/evaluation.jsonl')}
    cases={p['case_id']:p for p in jl(R/'frozen/parameter/cases.jsonl')}
    for case in cases.values():
        s=case['spec'];fresh=build(s['family'],s['index'],s['condition'])
        assert all(fresh[k]==case[k] for k in ['spec','role','rules','situation','menu'])
        for action in case['menu']:
            assert independent_payoffs(s,action)==case['payoffs'][str(action)]
    for lab in labels.values():
        case=cases[lab['case_id']]
        ledger={a:independent_payoffs(case['spec'],a) for a in case['menu']}
        assert set(lab['expected_actions'])==set(optima(ledger,lab['policy']))
    payloads={}
    for stage in cfg['stages']:
        ps,plan=tasks(R,stage);assert len(plan)==cfg[stage]['rows_per_model'] and len(plan)==len(set(plan))
        assert all(set(p)=={'prompt_id','task_type','messages','output_schema'} for p in ps.values());payloads.update(ps)
    checks=[]
    if a.tokenize:
        from transformers import AutoTokenizer
        for job,spec in cfg['models'].items():
            t=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True);maximum=0;rendered=0
            for p in payloads.values():
                for think in [False,True]:
                    ids=t.apply_chat_template(p['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=think)
                    assert len(ids)+cfg['max_tokens']<=cfg['max_model_len']
                    maximum=max(maximum,len(ids));rendered+=1
            checks.append(dict(job=job,max_prompt_tokens=maximum,native_templates_checked=rendered,weights_loaded=False))
    record=dict(status='passed_both_native_templates' if a.tokenize else 'passed_cpu_preflight',freeze_sha256=sha(R/'freeze.json'),
        independent_parameter_cases=80,parameter_goal_labels=480,metric_null_and_responsive_controls=True,
        parser_type_menu_truncation_and_reasoning_checks=True,complete_plans_verified=True,models=checks)
    target=R/('preflight.json' if a.tokenize else 'cpu_preflight.json');assert not target.exists();save(target,record)
    save(R/'symbolic_controls.json',dict(fixed_behavior=fixed,responsive_behavior=responsive,
        scope='Exact constructed distributions, not generated samples; low E alone does not distinguish fixed output from correct response. R has no normative ranking for free behavior.'))
    print(__import__('json').dumps(record))


if __name__=='__main__':main()
