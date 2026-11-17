"""Descriptive writing tables from the accepted main v3 matrix."""
import collections
import csv
from pathlib import Path
from campaign_common import R,load,jl,sha,save


def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    acceptance=load(R/'results/MAIN_MATRIX_ACCEPTANCE.json');selection=load(R/'results/SCORING_VERSION_SELECTION.json')
    assert acceptance['status']=='passed_all_eight_main_checkpoints'
    assert acceptance['score_selection_sha256']==sha(R/'results/SCORING_VERSION_SELECTION.json')
    detailed=[];by_group=[]
    for job,spec in selection['selected'].items():
        path=Path(spec['score_file']);score=load(path);assert sha(path)==spec['score_sha256']
        attempt={(x['prompt_id'],x['arm'],int(x['replicate'])):x for x in csv.DictReader((path.parent/'attempt_scores.csv').open(encoding='utf-8-sig',newline=''))}
        for s in score['summaries']:
            row=dict(model=job,language=s['language'],arm=s['arm'],
                core_Q_pair_equal=s['goal_pair_equal_all_attempt_Q'],core_Q_source_equal=s['goal_source_equal_all_attempt_Q'],core_Q_task_equal=s['goal_task_equal_all_attempt_Q'],
                U_changed=s['changed_fact_pair_equal_all_attempt_U'],U_stable=s['stable_fact_pair_equal_all_attempt_U'],
                parameter_Q=s['parameter_pair_equal_all_attempt_Q'],parameter_joint=s['parameter_disjoint_joint_all_four_correct'],
                behavior_E=s['behavior_pair_equal']['E_conditional'],behavior_E_low=s['behavior_pair_equal']['E_low'],behavior_E_high=s['behavior_pair_equal']['E_high'],
                behavior_G=s['behavior_pair_equal']['G_conditional'],split_half_W1=s['same_prompt_split_half_W1_conditional'])
            for stage,info in s['delivery_and_cost'].items():
                for key in ['attempts','valid','mean_completion_tokens','p95_completion_tokens','delivery_rate']:
                    row[stage+'_'+key]=info[key]
                row[stage+'_truncations']=info.get('truncated',0)
                row[stage+'_truncation_rate']=info.get('truncated',0)/info['attempts']
            detailed.append(row)
            for stage,name in [('core','core/goal_quartets.jsonl'),('parameter','parameter/goal_quartets.jsonl')]:
                qs=[q for q in jl(R/'frozen'/name) if q['language']==s['language']]
                groups=collections.defaultdict(list)
                for q in qs:groups[q['task'],q['policy']].append(q)
                for (task,policy),gqs in groups.items():
                    perpair=collections.defaultdict(list)
                    for q in gqs:perpair[q['pair_id']].append(sum(attempt[q[k],s['arm'],rep]['correct']=='True' for k in 'ABCD' for rep in range(2))/8)
                    value=sum(sum(v)/len(v) for v in perpair.values())/len(perpair)
                    by_group.append(dict(model=job,language=s['language'],arm=s['arm'],stage=stage,task=task,policy=policy,
                        condition_pairs=len(perpair),quartets=len(gqs),all_attempt_Q_pair_equal=value,
                        scope='Post-hoc descriptive breakdown of accepted main fixed panel; no new significance claim.'))
    out=R/'results';write(out/'paper_controls_delivery_cost.csv',detailed);write(out/'goal_task_policy_breakdown.csv',by_group)
    save(out/'WRITING_TABLE_PROVENANCE.json',dict(main_acceptance_sha256=sha(out/'MAIN_MATRIX_ACCEPTANCE.json'),
        selection_sha256=sha(out/'SCORING_VERSION_SELECTION.json'),builder_sha256=sha(Path(__file__)),
        files={n:sha(out/n) for n in ['paper_controls_delivery_cost.csv','goal_task_policy_breakdown.csv']},
        rows=dict(controls_delivery_cost=len(detailed),goal_task_policy=len(by_group)),scope='Main v3 only. Supplementary follow-up stays separate. Subgroup table is descriptive, selected after generation.'))
    print('Writing tables:',len(detailed),'cells;',len(by_group),'goal subgroups')


if __name__=='__main__':main()
