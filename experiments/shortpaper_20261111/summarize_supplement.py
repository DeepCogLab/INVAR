"""Summarize only independently accepted new stages; no inference on partial jobs."""
import collections
import csv
import itertools
import json
from pathlib import Path
from analyze_existing import BASE,OUT,jl,csvread,write,avg

R=Path(__file__).resolve().parent


def main():
    runs=R/'remote_records/runs';ps={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')}
    pairs={p['pair_id']:p for p in jl(BASE/'INVAR_dataset/data/pairs.jsonl')}
    gq=jl(BASE/'goal_quartets.jsonl');buckets=collections.defaultdict(dict)
    for q in gq:
        for side in 'ABCD':buckets[q['pair_id'],q['variant_id'],side][q['policy']]=q[side]
    conflicts=[];seen=set()
    for (pair,variant,side),policies in sorted(buckets.items()):
        for p1,p2 in itertools.combinations(sorted(policies),2):
            a,b=policies[p1],policies[p2];key=tuple(sorted([a,b]))
            if key in seen or set(ps[a]['expected_actions']) & set(ps[b]['expected_actions']):continue
            seen.add(key);conflicts.append(dict(pair_id=pair,prompt1=a,prompt2=b,task=pairs[pair]['task'],source=pairs[pair]['source_id']))
    assert len(conflicts)==862
    mains=[];goal_rows=[];native=[];disc=[];joint=[]
    for folder in sorted(runs.iterdir()):
        job=folder.name
        d=folder/'main'
        if (d/'independent_scalar_audit.json').exists():
            audit=json.loads((d/'independent_scalar_audit.json').read_text());assert audit['status']=='passed_independent_scalar'
            report=d/'scoring/reports'/job
            bs=csvread(report/'behavior_summary.csv');overall=next(x for x in bs if x['group']=='all')
            cs={x['value']:x for x in csvread(report/'control_summary.csv') if x['group']=='type'}
            bqs=csvread(report/'behavior_quartets.csv')
            for name,subset in [('all',bqs),('strict_format_only',[r for r in bqs if r['variant_id'] in ['V1_table','V2_rules_first','V2_situation_first','V3_situation_list','V4_rules_table']])]:
                for r in subset:r['sign_flip']=int(float(r['delta_original'])*float(r['delta_variant'])<0 and min(abs(float(r['delta_original'])),abs(float(r['delta_variant'])))>=.05)
                mains.append(dict(job=job,subset=name,n_pairs=len({r['pair_id'] for r in subset}),n_quartets=len(subset),
                    E=avg(subset,'E'),G=avg(subset,'G'),R=avg(subset,'R'),noise_reference=avg(subset,'E_permutation_reference'),
                    changed_fact_U=float(cs['changed_fact']['accuracy']),stable_fact_U=float(cs['stable_fact']['accuracy']),goal_Q=float(cs['goal']['accuracy']),
                    sign_flip_quartets=sum(r['sign_flip'] for r in subset),sign_flip_pair_references=len({r['pair_id'] for r in subset if r['sign_flip']})))
            for group in ['source_id','task']:
                sub=[r for r in bs if r['group']==group]
                mains.append(dict(job=job,subset=group+'_equal',n_groups=len(sub),
                    **{m:sum(float(r[m]) for r in sub)/len(sub) for m in ['E','G','R']}))
        d=folder/'goal_diagnostic'
        if (d/'independent_scalar_audit.json').exists():
            audit=json.loads((d/'independent_scalar_audit.json').read_text());assert audit['status']=='passed_independent_scalar'
            for r in audit['slices']:goal_rows.append(dict(job=job,**r))
            stats={(r['prompt_id'],r['arm']):float(r['accuracy_all_attempts']) for r in csvread(d/'prompt_scores.csv')}
            for arm in ['original','payoff_table']:
                rows=[dict(**c,mean_Q=(stats[c['prompt1'],arm]+stats[c['prompt2'],arm])/2,
                    worst_policy=min(stats[c['prompt1'],arm],stats[c['prompt2'],arm]),
                    both_correct_independent=stats[c['prompt1'],arm]*stats[c['prompt2'],arm]) for c in conflicts]
                for group in ['all']+sorted({r['task'] for r in rows}):
                    rs=[r for r in rows if group=='all' or r['task']==group]
                    disc.append(dict(job=job,arm=arm,group=group,n_unique_policy_prompt_pairs=len(rs),
                        **{m:sum(r[m] for r in rs)/len(rs) for m in ['mean_Q','worst_policy','both_correct_independent']},
                        weighting='Unique disjoint-policy prompt pairs; supplementary, not main Q.'))
        d=folder/'native'
        if (d/'independent_scalar_audit.json').exists():
            audit=json.loads((d/'independent_scalar_audit.json').read_text());assert audit['status']=='passed_independent_scalar'
            for r in audit['summaries']:native.append(dict(job=job,budget=1408,**r))
        d=folder/'native_long'
        if (d/'independent_scalar_audit.json').exists():
            audit=json.loads((d/'independent_scalar_audit.json').read_text());assert audit['status']=='passed_independent_scalar'
            native.append(dict(job=job,budget=4096,arm='native_thinking',**{k:audit[k] for k in ['n','valid','correct','truncated','matching_prefix','Q_all_attempts','Q_given_parse']}))
        d=folder/'joint_reasoning'
        if (d/'independent_scalar_audit.json').exists():
            audit=json.loads((d/'independent_scalar_audit.json').read_text());assert audit['status']=='passed_independent_scalar'
            for r in audit['summaries']:
                for weight in ['pair_equal','source_equal','task_equal']:
                    joint.append(dict(job=job,arm=r['arm'],weighting=weight,n=r['n'],valid=r['valid'],truncated=r['truncated'],parse_rate=r['parse_rate'],**r[weight]))
    write('new_model_main_summary.csv',mains);write('new_goal_diagnostic_summary.csv',goal_rows)
    write('new_native_summary.csv',native);write('new_goal_discrimination.csv',disc)
    write('new_joint_reasoning_summary.csv',joint)
    print(json.dumps(dict(main_models=sorted({r['job'] for r in mains}),goal_models=sorted({r['job'] for r in goal_rows}),
        native_models=sorted({r['job'] for r in native}))))


if __name__=='__main__':main()
