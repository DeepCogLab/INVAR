"""Build paired Chinese/English inputs and separate immutable evaluation labels."""
import collections
import copy
import re
from common import R, jl, load, save, sha, digest, write_jl


def translate_line(line, mapping):
    if not re.search('[\u4e00-\u9fff]', line):
        return line
    prefix = ''
    s = line.strip()
    if s.startswith('- '):
        prefix = '- '; s = s[2:]
    if s.startswith('| ') and s.endswith(' |'):
        first, second = s[2:-2].split(' | ', 1)
        first = {'条目':'Item', '规则':'Rule', '角色':'Role', '规则编号':'Rule number'}.get(first, first)
        assert not re.search('[\u4e00-\u9fff]', first), first
        return '| ' + first + ' | ' + mapping[second] + ' |'
    assert s in mapping, s
    return prefix + mapping[s]


def numeric_signature(text):
    return collections.Counter(re.findall(r'\d+(?:\.\d+)?', text))


def make_bilingual(p, lab, stage, mapping):
    results=[]
    original='\n'.join(m['content'] for m in p['messages'])
    for lang in ('zh','en'):
        messages=copy.deepcopy(p['messages'])
        if lang=='en':
            for m in messages:
                m['content']='\n'.join(translate_line(s,mapping) for s in m['content'].splitlines())
            translated='\n'.join(m['content'] for m in messages)
            assert not re.search('[\u4e00-\u9fff]',translated)
            assert numeric_signature(translated)==numeric_signature(original),p['prompt_id']
            # Unicode word boundaries treat adjacent Chinese letters as part of
            # the same word. Use ASCII identifier boundaries in both languages.
            for pattern in [r'(?<![A-Za-z0-9_])P\d+(?![A-Za-z0-9_])',
                            r'(?<![A-Za-z0-9_])(?:LEFT|RIGHT|NORTH|EAST|SOUTH|WEST|X|Y)(?![A-Za-z0-9_])']:
                assert collections.Counter(re.findall(pattern,translated))==collections.Counter(re.findall(pattern,original)),p['prompt_id']
        pid=stage+'_'+lang+'_'+p['prompt_id']
        payload=dict(prompt_id=pid,parent_prompt_id=p['prompt_id'],language=lang,stage=stage,
                     task_type=p['task_type'],messages=messages,output_schema=p['output_schema'])
        label=copy.deepcopy(lab);label.update(prompt_id=pid,parent_prompt_id=p['prompt_id'],language=lang,stage=stage)
        results.append((payload,label))
    return results


def main():
    assert not (R/'frozen').exists(),'Do not overwrite prepared inputs'
    baseline=R.parent/'baseline_v2';old=R.parent/'confirmation_20261112/frozen/parameter'
    panel=load(R/'panel_selection.json');selected=set(panel['prompt_ids'])
    mapping=load(R/'translation_map.json');mapping.update(load(R/'parameter_translation_map.json'))
    pairs={p['pair_id']:p for p in panel['pairs']}
    all_pairs={p['pair_id']:p for p in jl(baseline/'INVAR_dataset/data/pairs.jsonl')}
    inputs=[];labels=[]
    for p in jl(baseline/'inputs.jsonl'):
        if p['prompt_id'] not in selected:continue
        lab={k:copy.deepcopy(v) for k,v in p.items() if k not in ['messages','output_schema','content_sha256','character_count']}
        if p['task_type']=='behavior':
            uses=[u for u in p['uses'] if u['pair_id'] in pairs]
            sources={pairs[u['pair_id']]['source_id'] for u in uses}
            assert len(sources)==1
            lab.update(source_id=next(iter(sources)),task=pairs[uses[0]['pair_id']]['task'])
        else:
            # A content-deduplicated control prompt may have been first stored
            # under an unselected pair, while selected quartets reference it.
            # Preserve that parent identity; scoring uses explicit quartet refs.
            lab['source_id']=all_pairs[p['pair_id']]['source_id']
        for a,b in make_bilingual(p,lab,'core',mapping):inputs.append(a);labels.append(b)
    qparams=[q for q in jl(old/'goal_quartets.jsonl') if q['category']=='disjoint_optimum_sets']
    pids={q[k] for q in qparams for k in 'ABCD'}
    evals={p['prompt_id']:p for p in jl(old/'evaluation.jsonl')}
    for p in jl(old/'model_inputs.jsonl'):
        if p['prompt_id'] not in pids:continue
        for a,b in make_bilingual(p,evals[p['prompt_id']],'parameter',mapping):inputs.append(a);labels.append(b)
    for stage,types in [('core',['behavior','goal','fact']),('parameter',['goal'])]:
        for typ in types:
            name={'behavior':'behavior_quartets','goal':'goal_quartets','fact':'fact_quartets'}[typ]
            originals=panel[name] if stage=='core' else qparams
            qs=[]
            for lang in ('zh','en'):
                for q in originals:
                    item=copy.deepcopy(q);item['parent_quartet_id']=q['quartet_id']
                    item['quartet_id']=stage+'_'+lang+'_'+q['quartet_id'];item['language']=lang
                    if stage=='core':item['source_id']=pairs[q['pair_id']]['source_id']
                    for k in 'ABCD':item[k]=stage+'_'+lang+'_'+q[k]
                    qs.append(item)
            write_jl(R/'frozen'/stage/(name+'.jsonl'),qs)
    write_jl(R/'frozen/model_inputs.jsonl',inputs)
    write_jl(R/'frozen/evaluation.jsonl',labels)
    assert len(inputs)==len({p['prompt_id'] for p in inputs})
    assert all(set(p)=={'prompt_id','parent_prompt_id','language','stage','task_type','messages','output_schema'} for p in inputs)
    counts=collections.Counter((p['stage'],p['language'],p['task_type']) for p in inputs)
    save(R/'preparation_audit.json',dict(status='passed_structural_bilingual_checks',
        counts={':'.join(k):v for k,v in counts.items()},
        paired_prompt_count=len(inputs)//2,canonical_pairs=32,sources=4,tasks=6,
        parameter_pairs=len({q['pair_id'] for q in qparams}),parameter_quartets=len(qparams),
        input_sha256=sha(baseline/'inputs.jsonl'),panel_sha256=sha(R/'panel_selection.json'),
        translations_sha256={p.name:sha(p) for p in [R/'translation_map.json',R/'parameter_translation_map.json']},
        checks=['no remaining Chinese in English','all numerical multiplicities preserved',
                'participant and symbolic action identities preserved','output schemas unchanged',
                'evaluation labels separate from model payloads','A/B/C/D parent identities preserved'],
        limitations='AI-translated derivative of observed Chinese inputs, machine reviewed; no independent human certification or new empirical source'))
    print(dict(counts))


if __name__=='__main__':main()
