"""CPU-only deterministic selection before inspecting any new scores."""
import collections
import importlib.util
import os
import sys
from pathlib import Path
from common import *


def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def balanced(rows, n, key):
    groups = collections.defaultdict(list)
    for row in rows: groups[key(row)].append(row)
    for group in groups.values(): group.sort(key=lambda x: digest(['selection_20261113', x]))
    chosen = []
    while len(chosen) < n:
        added = False
        for k in sorted(groups):
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop(0)); added = True
        if not added: break
    return chosen


def main():
    assert not (R/'freeze.json').exists()
    sys.path.insert(0, str(OLD.parent/'20261025'))
    sys.path.insert(0, str(OLD.parent/'20261026'))
    os.environ['INVAR_METHODS_V1'] = str(OLD.parent/'20261025')
    os.environ['INVAR_METHODS_V2'] = str(OLD.parent/'20261026')
    v2 = module('old_v2_prepare', OLD.parent/'20261026/train_condition_anchor.py')
    ev = module('old_v4_validate', OLD/'run_transfer_evaluation.py')
    inputs = {x['prompt_id']: x for x in jl(BASE/'inputs.jsonl')}
    pairs = {x['pair_id']: x for x in jl(BASE/'INVAR_dataset/data/pairs.jsonl')}
    refs, sources = v2.source_closure(BASE, inputs, pairs)
    qs = jl(BASE/'INVAR_dataset/data/quartets.jsonl')
    variants = {'V4_rules_table', 'V6_relation'}
    preparation, totals = {}, {}
    for source in SOURCES:
        train = OLD/'learning/qwen3_8b'/source/'seed20261026/anchor_sft'
        train_edges = {tuple(sorted([e['left'],e['right']])) for e in jl(train/'free_edges.jsonl')}
        train_goals = {x['prompt_id'] for x in jl(train/'supervised_inputs.jsonl') if x['task_type']=='goal'}
        pools = {k:{} for k in ('train', 'heldout_source', 'heldout_variant')}
        for q in qs:
            src = q['source_id']; variant = q['variant_id']
            stratum = ('heldout_source' if src == source and variant not in variants else
                       'heldout_variant' if src != source and variant in variants else
                       'train' if src != source and variant not in variants else None)
            if stratum is None: continue
            for side in ('AB','CD'):
                ends = tuple(sorted(q[s] for s in side))
                if ends[0] == ends[1] or any(sources[p] != [src] for p in ends): continue
                menus = [v2.public_menu(inputs[p]) for p in ends]
                if menus[0] != menus[1] or len(menus[0][1]) > 16: continue
                if stratum == 'train': assert ends in train_edges
                else: assert ends not in train_edges
                eid = digest(list(ends))[:24]
                if eid not in pools[stratum]:
                    pools[stratum][eid] = dict(edge_id=eid, left=ends[0], right=ends[1],
                        source_id=src, task=pairs[q['pair_id']]['task'], stratum=stratum, references=[])
                pools[stratum][eid]['references'].append(dict(pair_id=q['pair_id'], quartet_id=q['quartet_id'], side=side, variant=variant))
        edges = []
        for stratum, pool in pools.items():
            edges.extend(balanced(list(pool.values()), 16, lambda x:(x['source_id'],x['task'])))
        selected = {e[k] for e in edges for k in ('left','right')}
        goal_groups = {k:[] for k in pools}
        for pid,row in inputs.items():
            if row['task_type'] != 'goal' or len(sources[pid])!=1: continue
            field, menu = v2.public_menu(row)
            if len(menu)>16: continue
            src = sources[pid][0]; var = row.get('variant_id','base')
            stratum = ('train' if pid in train_goals else
                       'heldout_source' if src==source and var not in variants else
                       'heldout_variant' if src!=source and var in variants else None)
            if stratum is None: continue
            goal_groups[stratum].append(dict(prompt_id=pid, source_id=src, task=row['task'], stratum=stratum))
        goals = [x for pool in goal_groups.values() for x in balanced(pool,8,lambda x:(x['source_id'],x['task']))]
        selected.update(x['prompt_id'] for x in goals)
        rows, labels = [], []
        for pid in sorted(selected):
            row = inputs[pid]; field, menu = v2.public_menu(row)
            assert len(menu)<=16 and [m['role'] for m in row['messages']]==['system','user']
            rows.append(dict(prompt_id=pid, task_type=row['task_type'], field=field, candidates=menu,
                             messages=[dict(role=m['role'],content=m['content']) for m in row['messages']],
                             output_schema=row['output_schema']))
            if row['task_type']=='goal':
                expected = row['expected_actions']; assert expected and all(x in menu for x in expected)
                labels.append(dict(prompt_id=pid, expected=expected))
        write_jl(panel(source)/'model_inputs.jsonl', rows)
        write_jl(panel(source)/'goal_labels.jsonl', labels)
        write_jl(panel(source)/'edges.jsonl', edges)
        write_jl(panel(source)/'goal_membership.jsonl', goals)
        totals[source] = dict(prompts=len(rows), behavior=sum(x['task_type']=='behavior' for x in rows),
                             goals=len(goals), edges=len(edges), calls_per_state=sum(reps(x) for x in rows))
        preparation[source] = dict(eligible_edges={k:len(v) for k,v in pools.items()},
            selected_edges=dict(collections.Counter(e['stratum'] for e in edges)),
            eligible_goals={k:len(v) for k,v in goal_groups.items()},
            selected_goals=dict(collections.Counter(x['stratum'] for x in goals)))
    inventory = {}
    model_identity = {}
    for job,model in MODELS.items():
        weights = Path(model['path']); inventory[job] = {}
        model_identity[job] = dict(path=str(weights.resolve()), files_sha256={p.name:sha(p) for p in sorted(weights.iterdir())
            if p.suffix in ('.json','.safetensors') or p.name in ('merges.txt','vocab.json')})
        for source in SOURCES:
            inventory[job][source] = {}
            for state in STATES[1:]:
                arm, step = state.rsplit('_',1)
                path = OLD/'learning'/job/source/'seed20261026'/arm/f'adapter_checkpoint_{int(step):03}'
                identity = ev.adapter_identity(path,arm,job,weights,BASE,OLD.parent/'20261026/teacher_cache')
                assert identity['training_identity']['adapter_checkpoint_step']==int(step)
                inventory[job][source][state] = identity
    save(R/'adapter_inventory.json', inventory); save(R/'model_identity.json', model_identity)
    save(R/'preparation_audit.json', dict(status='passed_fixed_outcome_blind_selection', panels=preparation, totals=totals,
        original_inputs_sha256=sha(BASE/'inputs.jsonl'), old_8666_unchanged=len(inputs)==8666,
        selection='round-robin source/task strata then SHA256; 16 behavior edges and 8 goals per stratum, use all if fewer',
        prospective_scores_inspected=False, training_seed=20261026, checkpoint_steps=[50,100,150],
        complete_small_menus_only=True, new_training_runs=0, native_thinking_comparison=False))
    save(R/'plan.json', dict(models=MODELS, sources=SOURCES, states=STATES, panels=totals,
        science_generated_rows=sum(x['calls_per_state'] for x in totals.values())*len(STATES)*len(MODELS),
        proxy_prompt_distributions=sum(x['prompts'] for x in totals.values())*len(STATES)*len(MODELS),
        pilot_rows_per_model=8, budget_expansion_allowed=False))
    print(canonical(load(R/'plan.json')))


if __name__=='__main__': main()
