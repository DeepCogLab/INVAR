"""Read-only accepted-result tables, including every checkpoint and every stratum."""
import csv
import math
from datetime import datetime,timezone
from common import *
MIRROR=Path('artifacts/mechanism_20261113_v2')
RESULT=R.parents[1]/'results/20261113_训练机制补实验'


def mean(values):
    return math.fsum(values)/len(values) if values and all(x is not None for x in values) else None


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    acceptance=load(MIRROR/'local_mirror_acceptance.json')
    assert acceptance['jobs'] and acceptance['local_full_raw_rescore'] and acceptance['independent_scalar_audit']=='passed_all_mirrored_jobs'
    assert acceptance['freeze_sha256']=='3fa77d5c406c43cd12d01742903f9b6a79dad8c47cd6d5bea3cd25924fca23a5'
    RESULT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    history=RESULT/'table_history'/stamp;history.mkdir(parents=True,exist_ok=True)
    for p in RESULT.iterdir():
        if p.is_file() and p.name.startswith(('accepted_','机制结果与全部检查点','mechanism_table_provenance')):
            (history/p.name).write_bytes(p.read_bytes())
    points=[];contrasts=[];input_hashes={};signs=[]
    for job in acceptance['jobs']:
        scalar=MIRROR/'local_scalar_audits'/f'{job}.json'
        assert sha(scalar)==acceptance['scalar_audit_files'][job] and load(scalar)['status']=='passed_independent_raw_scalar_reconstruction'
        a=MIRROR/'runs'/job/'analysis';complete=load(a/'SCORING_COMPLETE.json')
        for name,h in complete['inputs_sha256'].items(): assert sha(MIRROR/'runs'/job/name)==h
        for name,h in complete['output_sha256'].items():assert sha(a/name)==h
        input_hashes[job]=dict(scalar_audit_sha256=sha(scalar),scoring_sha256=sha(a/'SCORING_COMPLETE.json'),raw_sha256=complete['inputs_sha256'])
        edges=jl(a/'edges.jsonl');goals=jl(a/'goals.jsonl');bridge=jl(a/'bridge.jsonl')
        for source in SOURCES:
            memberships=jl(MIRROR/'frozen'/source/'goal_membership.jsonl')
            for state in STATES:
                for stratum in ('train','heldout_source','heldout_variant'):
                    es=[e for e in edges if e['source']==source and e['state']==state and e['stratum']==stratum]
                    gs=[g for g in goals if g['source']==source and g['state']==state and g['stratum']==stratum]
                    used={e[k] for e in es for k in ('left','right')}|{g['prompt_id'] for g in gs}
                    bs=[b for b in bridge if b['source']==source and b['state']==state and b['prompt_id'] in used]
                    point=dict(model=job,heldout_source=source,state=state,stratum=stratum,edges=len(es),goal_prompts=len(gs),
                        unique_prompts=len(used),attempts=sum(b['attempts'] for b in bs),failures=sum(b['failures'] for b in bs))
                    for metric in ('proxy_W1','proxy_JS','decoder_W1_conditional','decoder_JS_conditional'):
                        point[metric]=mean([e[metric] for e in es])
                    point.update(missing_quality_W1_lower=mean([e['missing_quality_W1_bounds'][0] for e in es]),
                        missing_quality_W1_upper=mean([e['missing_quality_W1_bounds'][1] for e in es]),
                        proxy_correct_set_mass=mean([g['proxy_correct_set_mass'] for g in gs]),
                        Q_all_attempts=mean([g['Q'] for g in gs]),
                        proxy_decoder_TV=mean([b['proxy_decoder_tv'] for b in bs]))
                    points.append(point)
        cs=jl(a/'contrasts.jsonl')
        for c in cs:
            flat=dict(model=job)
            for k,v in c.items():
                if isinstance(v,dict):
                    flat[k+'_defined']=v['defined'];flat[k+'_undefined']=v['undefined']
                    flat[k+'_empirical_lower']=v['empirical_range'][0] if v['empirical_range'] else None
                    flat[k+'_empirical_upper']=v['empirical_range'][1] if v['empirical_range'] else None
                else:flat[k]=v
            contrasts.append(flat)
        for stratum in ('train','heldout_source','heldout_variant'):
            group=[c for c in cs if c['stratum']==stratum]
            signs.append(dict(model=job,stratum=stratum,comparisons=len(group),
                proxy_W1_decreases=sum(c['delta_proxy_W1'] is not None and c['delta_proxy_W1'] < -1e-12 for c in group),
                decoder_W1_decreases=sum(c['delta_decoder_W1_conditional'] is not None and c['delta_decoder_W1_conditional'] < -1e-12 for c in group),
                proxy_W1_down_decoder_W1_up=sum(c['delta_proxy_W1'] is not None and c['delta_decoder_W1_conditional'] is not None and
                    c['delta_proxy_W1'] < -1e-12 and c['delta_decoder_W1_conditional'] > 1e-12 for c in group)))
    csv_write(RESULT/'accepted_points_all_checkpoints.csv',points)
    csv_write(RESULT/'accepted_JS_minus_SFT_all_checkpoints.csv',contrasts)
    csv_write(RESULT/'accepted_descriptive_sign_counts.csv',signs)
    lines=['# 训练机制实验：已验收结果与全部检查点','',
        '验收范围：'+('两模型全部完成。' if len(acceptance['jobs'])==2 else '部分模型；未完成模型不填科学结果。'),
        '已验收模型：'+', '.join(acceptance['jobs'])+'。全部原始逐SHA、本地完整评分及另写标量实现通过。','',
        '下表统计固定面板内JS−SFT点差值的符号，每模型每层包含四来源×三个检查点共12项。它是描述计数，不是独立重复或显著性检验；全部50/100/150步及目标Q详见CSV。','',
        '| 模型 | 固定边分层 | 比较项 | 代理W1下降 | 实际生成W1下降 | 代理下降但生成上升 |',
        '|---|---|---:|---:|---:|---:|']
    for s in signs:lines.append('| '+ ' | '.join(str(s[k]) for k in ('model','stratum','comparisons','proxy_W1_decreases','decoder_W1_decreases','proxy_W1_down_decoder_W1_up'))+' |')
    lines+=['','全部检查点表：`accepted_points_all_checkpoints.csv`；配对差值及500次共享提示经验范围：`accepted_JS_minus_SFT_all_checkpoints.csv`。',
        '', '目标Q以所选目标提示等权、全部尝试为分母；PGG来源留出层没有目标题，保持空值。正确集合代理概率保留所有平局行动。',
        '', 'W1/JS是所选等价边均值，生成值为合法可解析子集的conditional量；缺失质量W1界单列，不包含采样误差。16/32/64次敏感性和完整直方图见D盘已验收analysis目录。',
        '', '500次bootstrap的范围是经验重采样范围，不是校准置信区间；同层共享提示在抽样中复用计数，跨层不宣称联合覆盖。未定义抽样保留数量，不填0。',
        '', '此诊断只有一个预定旧种子、小菜单、已观察来源；呈现留出可共用原表达端。不选最佳检查点，也不把此处边均值改写为原92对主权重E或未闭合G/R。',
        '', '完整序列归一化代理与token约束、温度、top-p及推理后端共同构成不同接口。即使同菜单W1异向，也不能单独归因某一解码组件。低熵不代表自由社会行为错误。']
    (RESULT/'机制结果与全部检查点.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(RESULT/'mechanism_table_provenance.json',dict(status='complete' if len(acceptance['jobs'])==2 else 'partial',
        jobs=acceptance['jobs'],freeze_sha256=acceptance['freeze_sha256'],inputs=input_hashes,
        files_sha256={p.name:sha(p) for p in RESULT.iterdir() if p.is_file() and p.name.startswith(('accepted_','机制结果与全部检查点'))},
        writer_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
