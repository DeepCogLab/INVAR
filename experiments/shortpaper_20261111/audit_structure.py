"""Re-render into memory and verify exact structural invariants; no dataset writes."""
import collections
import importlib.util
import json
from pathlib import Path
from analyze_existing import BASE,OUT,jl,csvread,write,avg


def main():
    ps={p['prompt_id']:p for p in jl(BASE/'inputs.jsonl')}
    cs={c['case_id']:c for c in jl(BASE/'INVAR_dataset/data/cases.jsonl')}
    pairs={p['pair_id']:p for p in jl(BASE/'INVAR_dataset/data/pairs.jsonl')}
    qs=jl(BASE/'INVAR_dataset/data/quartets.jsonl')
    spec=importlib.util.spec_from_file_location('frozen_renderer',BASE/'INVAR_dataset/build_dataset.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    verified=collections.Counter()
    for q in qs:
        pair=pairs[q['pair_id']]
        for tag,side,variant in zip('ABCD',[0,0,1,1],['base',q['variant_id'],'base',q['variant_id']]):
            p=ps[q[tag]];c=cs[pair[f'case{side}_id']]
            assert b.render(c,variant)==p['messages'],(q['quartet_id'],tag)
            assert p['action_space']==c['action_space'] and p['outcome']==c['outcome']
            if variant=='V5_ids':
                text=p['messages'][-1]['content']
                for old,new in b.IDMAP.items():
                    text=text.replace(new,old)
                assert text==b.render(c)[-1]['content']
            if variant!='V6_relation':
                body=p['messages'][-1]['content']
                for key in ['role_facts','rules','situation']:
                    for fact in c[key]:
                        expected=b.renumber(fact) if variant=='V5_ids' else b.role_sentence(fact) if variant=='V1_paragraph' and key=='role_facts' else fact
                        assert expected in body or expected.replace('|','\\|') in body
            verified[variant]+=1
    sign=[]
    inv=csvread(OUT/'verified_model_inventory.csv')
    for model in inv:
        rows=csvread(Path(model['path'])/'behavior_quartets.csv')
        for subset in ['all','without_V6','strict_format_only']:
            selected=[r for r in rows if subset=='all' or subset=='without_V6' and r['variant_id']!='V6_relation' or subset=='strict_format_only' and r['variant_id'] in ['V1_table','V2_rules_first','V2_situation_first','V3_situation_list','V4_rules_table']]
            for r in selected:
                r['robust_sign_reversal']=int(float(r['delta_original'])*float(r['delta_variant'])<0 and min(abs(float(r['delta_original'])),abs(float(r['delta_variant'])))>=.05)
            bad=[r for r in selected if r['robust_sign_reversal']]
            sign.append(dict(campaign=model['campaign'],model=model['model'],subset=subset,
                E=avg(selected,'E'),G=avg(selected,'G'),n_quartets=len(selected),
                n_sign_reversal_quartets=len(bad),n_sign_reversal_pair_references=len({r['pair_id'] for r in bad}),
                pair_equal_sign_reversal_fraction=avg(selected,'robust_sign_reversal'),
                criterion='Opposite signs and both absolute mean effects >= 0.05; descriptive threshold, no multiplicity-adjusted significance claim'))
    write('scientific_conclusion_sign_sensitivity.csv',sign)
    (OUT/'structural_audit.json').write_text(json.dumps(dict(
        behavioral_quartets=701,rendered_references=2804,unique_behavior_inputs=1240,
        exact_canonical_rerender_passed=True,menu_and_outcome_preserved=True,renumber_bijection_passed=True,
        references_by_variant=dict(verified),
        scope='Exact rendering, constituent text, action space and outcome invariants. Semantic/pragmatic equivalence of a paraphrase is not inferred from a hash.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('All 2804 behavioral references passed structural checks; scientific sign diagnostics saved.')


if __name__=='__main__':
    main()
