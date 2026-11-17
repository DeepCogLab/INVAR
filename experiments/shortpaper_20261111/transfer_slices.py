"""Expose prespecified V4 transfer dimensions from frozen per-quartet scores."""
import collections
from pathlib import Path
from analyze_existing import BASE,OUT,jl,csvread,write,avg


def main():
    ps={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')}
    pairs={p['pair_id']:p for p in jl(BASE/'INVAR_dataset/data/pairs.jsonl')}
    tasks_by_source=collections.defaultdict(set)
    for p in pairs.values():
        tasks_by_source[p['source_id']].add(p['task'])
    root=Path('artifacts/methods_v4_20261028/evaluations/transfer')
    results=[]
    for file in sorted(root.rglob('quartets.csv')):
        if file.parent.name!='scores':
            continue
        model,source,seed=file.relative_to(root).parts[:3]
        rows=[r for r in csvread(file) if r['kind']=='behavior']
        seen_tasks=set.union(*(t for s,t in tasks_by_source.items() if s!=source))
        for r in rows:
            space=ps[r['A']]['action_space']
            r['menu_size']=len(space['values']) if 'values' in space else space['max']-space['min']+1
        for method in ['baseline','anchor_sft','anchor_js']:
            mr=[r for r in rows if r['method']==method]
            subsets={
                'all':mr,
                'strict_format_only':[r for r in mr if r['variant_id'] in ['V1_table','V2_rules_first','V2_situation_first','V3_situation_list','V4_rules_table']],
                'without_V6':[r for r in mr if r['variant_id']!='V6_relation'],
                'heldout_expression_V4':[r for r in mr if r['variant_id']=='V4_rules_table'],
                'heldout_expression_V6':[r for r in mr if r['variant_id']=='V6_relation'],
                'small_menu_le16':[r for r in mr if r['menu_size']<=16],
                'large_menu_gt16':[r for r in mr if r['menu_size']>16],
                'task_present_in_other_sources':[r for r in mr if r['task'] in seen_tasks],
                'task_absent_in_other_sources':[r for r in mr if r['task'] not in seen_tasks],
            }
            for subset,rs in subsets.items():
                if rs:
                    results.append(dict(model=model,source=source,seed=seed,method=method,subset=subset,
                        n_pairs=len({r['pair_id'] for r in rs}),n_quartets=len(rs),
                        E=avg(rs,'E'),G=avg(rs,'G'),R=avg(rs,'R')))
    write('training_transfer_slices.csv',results)
    differences=[]
    for key in sorted({(r['model'],r['source'],r['seed'],r['subset']) for r in results}):
        rows={r['method']:r for r in results if (r['model'],r['source'],r['seed'],r['subset'])==key}
        s,j=rows['anchor_sft'],rows['anchor_js']
        differences.append(dict(zip(['model','source','seed','subset'],key),
            n_pairs=s['n_pairs'],E_JS_minus_SFT=j['E']-s['E'],G_JS_minus_SFT=j['G']-s['G'],R_JS_minus_SFT=j['R']-s['R']))
    write('training_transfer_slice_differences.csv',differences)
    print('Saved',len(results),'descriptive slices and',len(differences),'matched JS-SFT contrasts.')


if __name__=='__main__':
    main()
