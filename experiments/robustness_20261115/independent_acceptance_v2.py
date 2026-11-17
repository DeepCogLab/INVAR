"""Post-generation independent raw audit; imports no scientific scorer/parser."""
import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MIRROR=Path('artifacts/robustness_20261115/corrected_v2')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def signature(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(4*1024*1024):h.update(data)
    return h.hexdigest()


def no_duplicates(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate')
        d[k]=v
    return d


def delivered(row,schema,kind,prefilled):
    if row['finish_reason']=='length':return None
    s=row['response_text'].strip()
    if kind=='native_effort':
        special=row['response_with_special_tokens'];blocks=special.split('<|start|>');finals=[]
        for i,block in enumerate(blocks):
            head,separator,body=block.partition('<|message|>')
            if not separator:continue
            author,separator,channel=head.partition('<|channel|>')
            if not separator:continue
            channel_parts=channel.split('<|constrain|>')
            if channel_parts[0].strip()!='final':continue
            if author.strip() not in ([''] if i==0 else ['assistant']):return None
            if len(channel_parts)>2:return None
            if len(channel_parts)==2 and channel_parts[1].strip().casefold()!='json':return None
            if i!=len(blocks)-1 or not body.rstrip().endswith('<|return|>'):return None
            finals.append(body.rstrip()[:-len('<|return|>')].strip())
        if len(finals)!=1:return None
        s=finals[0]
    if prefilled and s.count('</think>')!=1:return None
    if '<think>' in s or '</think>' in s:
        if s.count('<think>')>1 or s.count('</think>')!=1:return None
        s=s.partition('</think>')[2].strip()
    if s.startswith('<Answer>') and s.endswith('</Answer>'):s=s[8:-9].strip()
    fence=re.search(r'```(?:json)?\s*(\{[^`]*\})\s*```\s*$',s,re.S|re.I)
    if fence:s=fence[1].strip()
    candidates=[];parts=s.splitlines(keepends=True)
    for i,line in enumerate(parts):
        if not line.lstrip().startswith('{'):continue
        try:o=json.loads(''.join(parts[i:]).strip(),object_pairs_hook=no_duplicates)
        except (ValueError,TypeError):continue
        candidates.append(o)
    if len(candidates)!=1:return None
    o=candidates[0]
    if type(o) is not dict or set(o)!=set(schema['properties']):return None
    for k,v in o.items():
        spec=schema['properties'][k]
        if type(v) is not {'integer':int,'string':str}[spec['type']]:return None
        if 'enum' in spec and v not in spec['enum']:return None
        if v<spec.get('minimum',v) or v>spec.get('maximum',v):return None
    return o


def transport(a,b):
    """Optimal monotone mass transport; independent of CDF/quantile-grid code."""
    if not a or not b:return None
    a=sorted(a);b=sorted(b);i=j=0;wa=1/len(a);wb=1/len(b);result=0.
    while i<len(a) and j<len(b):
        mass=min(wa,wb);result+=mass*abs(a[i]-b[j]);wa-=mass;wb-=mass
        if wa<1e-12:i+=1;wa=1/len(a)
        if wb<1e-12:j+=1;wb=1/len(b)
    return result


def behavior(q,values,n):
    result={}
    for label,edges in [('E',['AB','CD']),('R',['AC','BD'])]:
        ds=[]
        for edge in edges:
            x,y=[q[k] for k in edge];a,b=values[x],values[y]
            if x==y:ds.append((0.,0.,0.));continue
            point=transport(a,b);missing=(n-len(a)+n-len(b))/n
            ds.append((point,0. if point is None else max(0.,point-missing),1. if point is None else min(1.,point+missing)))
        result[label+'_conditional']=None if any(v[0] is None for v in ds) else sum(v[0] for v in ds)/2
        result[label+'_low']=sum(v[1] for v in ds)/2;result[label+'_high']=sum(v[2] for v in ds)/2
    coeff=collections.Counter()
    for k,c in zip('ABCD',[1,-1,-1,1]):coeff[q[k]]+=c
    low=high=0.;point=0.;defined=True
    for pid,c in coeff.items():
        if c==0:continue
        a=values[pid];lo=sum(a)/n;hi=lo+(n-len(a))/n
        low+=c*(lo if c>0 else hi);high+=c*(hi if c>0 else lo)
        if a:point+=c*sum(a)/len(a)
        else:defined=False
    result['G_conditional']=abs(point) if defined else None
    result['G_low']=0. if low<=0<=high else min(abs(low),abs(high))
    result['G_high']=min(max(abs(low),abs(high)),2*result['E_high'])
    return result


def average(v):return statistics.mean(v) if v and all(x is not None for x in v) else None


def weighted(qs,vals,key=None):
    pairs=collections.defaultdict(list);meta={q['pair_id']:q for q in qs}
    for q,v in zip(qs,vals):pairs[q['pair_id']].append(v)
    groups=collections.defaultdict(list)
    for pid,vs in pairs.items():groups[meta[pid][key] if key else 'all'].append(average(vs))
    return average([average(v) for v in groups.values()])


def equals(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-10),(a,b)


def audit(job):
    cfg=read(ROOT/'freeze.json');d=MIRROR/'runs'/job;identity=read(d/'identity.json');complete=read(d/'COMPLETE.json');scored=read(d/'SCORING_COMPLETE.json')
    assert filehash(ROOT/'freeze.json')==filehash(MIRROR/'freeze.json')==identity['freeze_sha256']==scored['freeze_sha256']
    rawhash=filehash(d/'responses.jsonl');assert rawhash==complete['raw_sha256']==scored['raw_sha256']
    for name,h in cfg['files'].items():assert filehash(MIRROR/name)==h,name
    ps={p['prompt_id']:p for p in lines(ROOT/'frozen/model_inputs.jsonl')};labels={p['prompt_id']:p for p in lines(ROOT/'frozen/evaluation.jsonl')}
    kind=cfg['models'][job]['kind'];arms=['native_off','native_on'] if kind=='native_toggle' else ['effort_low','effort_high'] if kind=='native_effort' else ['direct','brief_reasoning']
    repeats=cfg['protocol']['repeats'];expected={(pid,a,i) for pid,p in ps.items() for a in arms for i in range(repeats['parameter' if p['stage']=='parameter' else p['task_type']])}
    csvrows={(v['prompt_id'],v['arm'],int(v['replicate'])):v for v in csv.DictReader((d/'attempt_scores.csv').open(encoding='utf-8-sig',newline=''))}
    assert set(csvrows)==expected
    seen=set();correct={};values=collections.defaultdict(lambda:collections.defaultdict(list));stats=collections.defaultdict(collections.Counter);ih=signature(identity)
    with (d/'responses.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);pid,arm,rep=row['prompt_id'],row['arm'],row['replicate'];k=pid,arm,rep
            assert k in expected and k not in seen;seen.add(k);p=ps[pid];lab=labels[pid]
            assert row['model']==job and row['identity_sha256']==ih
            assert row['sampling_seed']==int(signature([cfg['protocol']['seed'],'science',job,pid,arm,rep])[:8],16)%(2**31-1)
            assert row['rendered_sha256']==identity['rendered_sha256'][pid+'__'+arm]
            assert row['prompt_token_ids_sha256']==identity['prompt_token_ids_sha256'][pid+'__'+arm]
            assert row['completion_tokens']==len(row['token_ids'])
            obj=delivered(row,p['output_schema'],kind,identity['prefilled_think'][pid+'__'+arm]);valid=obj is not None
            assert valid==(csvrows[k]['valid']=='True'),k
            good=False;x=None
            if valid:
                if p['task_type']=='behavior':
                    o=lab['outcome'];x=float(obj['action']==o['target_action']) if o['kind']=='probability' else obj['action']/o['denominator'];values[arm][pid].append(x)
                elif p['task_type']=='goal':good=obj['action'] in lab['expected_actions']
                else:good=obj['answer']==lab['expected_answer']
            if p['task_type']=='behavior':equals(x,float(csvrows[k]['normalized_outcome']) if csvrows[k]['normalized_outcome'] else None)
            else:assert good==(csvrows[k]['correct']=='True'),k
            correct[k]=good;c=stats[arm,p['language'],p['stage'],p['task_type']]
            c['attempts']+=1;c['valid']+=valid;c['correct']+=good;c['tokens']+=row['completion_tokens']
    assert seen==expected and len(seen)==11760
    bqs=lines(ROOT/'frozen/core/behavior_quartets.jsonl');gqs=lines(ROOT/'frozen/core/goal_quartets.jsonl');fqs=lines(ROOT/'frozen/core/fact_quartets.jsonl');pqs=lines(ROOT/'frozen/parameter/goal_quartets.jsonl')
    comparisons=0
    for summary in scored['summaries']:
        arm,lang=summary['arm'],summary['language'];qs=[q for q in bqs if q['language']==lang];rs=[behavior(q,values[arm],repeats['behavior']) for q in qs]
        for name,key in [('behavior_pair_equal',None),('behavior_source_equal','source_id'),('behavior_task_equal','task')]:
            for m in rs[0]:equals(weighted(qs,[r[m] for r in rs],key),summary[name][m]);comparisons+=1
        for stage,typ in [('core','goal'),('core','changed_fact'),('core','stable_fact'),('parameter','goal')]:
            cq=[q for q in (pqs if stage=='parameter' else gqs if typ=='goal' else fqs) if q['language']==lang and (typ=='goal' or q['probe_type']==typ)]
            n=repeats['parameter' if stage=='parameter' else typ]
            vals=[sum(correct[q[c],arm,i] for c in 'ABCD' for i in range(n))/(4*n) for q in cq]
            name='parameter_pair_equal_all_attempt_Q' if stage=='parameter' else 'goal_pair_equal_all_attempt_Q' if typ=='goal' else typ+'_pair_equal_all_attempt_U'
            equals(weighted(cq,vals),summary[name]);comparisons+=1
            if stage=='core' and typ=='goal':
                for weight,key in [('source','source_id'),('task','task')]:equals(weighted(cq,vals,key),summary['goal_'+weight+'_equal_all_attempt_Q']);comparisons+=1
            if typ=='goal':
                for q in cq:
                    assert set(labels[q['A']]['expected_actions'])==set(labels[q['B']]['expected_actions'])
                    assert set(labels[q['C']]['expected_actions'])==set(labels[q['D']]['expected_actions'])
                switch=[q for q in cq if set(labels[q['A']]['expected_actions']).isdisjoint(labels[q['C']]['expected_actions'])]
                equals(average([sum(all(correct[q[c],arm,i] for c in 'ABCD') for i in range(n))/n for q in switch]),summary[stage+'_disjoint_joint_all_four_correct']);comparisons+=1
            c=stats[arm,lang,stage,typ];reported=summary['delivery_and_cost'][stage+'_'+typ]
            equals(c['valid']/c['attempts'],reported['delivery_rate']);equals(c['tokens']/c['attempts'],reported['mean_completion_tokens']);comparisons+=2
    result=dict(status='passed_independent_v2_message_header_transport_and_accuracy',job=job,rows=len(seen),raw_sha256=rawhash,
        freeze_sha256=filehash(ROOT/'freeze.json'),auditor_sha256=filehash(Path(__file__)),audited_summary_values=comparisons,
        whole_trial_set_seeds_identity_and_token_counts_verified=True,all_attempt_csv_rows_independently_reparsed=True,
        shared_quartet_references_full_optimum_sets_and_missingness_verified=True,
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        correction_manifest_sha256=filehash(MIRROR/'CORRECTION_MANIFEST.json'),
        original_frozen_audit_preserved=True,
        limits='Post-generation verification of frozen estimands, not new data, calibrated uncertainty, human translation review, or an independent empirical source.')
    out=ROOT/'independent_audits_v2';out.mkdir(exist_ok=True);(out/(job+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);audit(ap.parse_args().job)
