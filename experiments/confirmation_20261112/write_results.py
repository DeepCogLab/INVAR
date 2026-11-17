"""Compact evidence tables gated on independent local raw-answer acceptance."""
import csv
import io
import json
from pathlib import Path
from common import R, load, sha

ROOT = R.parents[1]
OUT = ROOT / 'results/20261112_确认补实验'
MIRROR = Path('artifacts/confirmation_20261112')


def write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding='utf-8-sig')


def number(value):
    return '未定义' if value is None else f'{value:.6f}'


def main():
    receipt = load(R / 'local_mirror_acceptance.json')
    index = load(MIRROR / 'mirror_index.json')['files']
    behavior, goals, deliveries, transitions, provenance = [], [], [], [], []
    lines = ['# 第二轮确认实验：已验收结果与正文表', '',
             '仅纳入已从D盘原始回答独立复算通过的完整阶段；未完成模型不作排名。旧批次不拼接进本轮分母。', '',
             f"已验收{receipt['scientific_rows']:,}/280,128条科学回答，另40条工程先导单列。", '']
    for item in receipt['accepted']:
        job, stage = item['job'], item['stage']
        relative = f'runs/{job}/{stage}/SCORING_COMPLETE.json'
        source = MIRROR / relative
        assert sha(source) == index[relative]['sha256']
        audit = load(source)
        assert audit['raw_sha256'] == item['raw_sha256']
        identity_path = MIRROR / f'runs/{job}/{stage}/extension_identity_audit.json'
        identity = load(identity_path)
        assert identity['raw_sha256'] == item['raw_sha256']
        stats_path = OUT / 'native_statistics' / job / stage / 'COMPLETE.json'
        statistics = load(stats_path) if stats_path.exists() else None
        if statistics:
            assert statistics['source_raw_sha256'] == item['raw_sha256']
            assert statistics['script_sha256'] == load(OUT/'native_statistics/protocol.json')['script_sha256']
        provenance.append(dict(job=job, stage=stage, raw_sha256=item['raw_sha256'],
                               scoring_sha256=sha(source), identity_audit_sha256=sha(identity_path),
                               statistics_complete_sha256=sha(stats_path) if statistics else None))
        for summary in audit['summaries']:
            arm = summary['arm']
            for task_type, counts in summary['counts'].items():
                deliveries.append(dict(job=job, stage=stage, arm=arm, task_type=task_type,
                                       n=counts['n'], valid=counts['valid'], truncated=counts['truncated'],
                                       other_invalid=counts['n']-counts['valid']-counts['truncated'],
                                       correct=counts.get('correct'), completion_tokens=counts['completion_tokens']))
            for weighting in ['pair_equal', 'source_equal', 'task_equal']:
                value = summary['goal'][weighting]
                goals.append(dict(job=job, stage=stage, arm=arm, weighting=weighting,
                                  Q_all_attempts=value['Q'], worst_side_Q=value['worst_side']))
                if 'behavior' in summary:
                    value = summary['behavior'][weighting]
                    behavior.append(dict(job=job, arm=arm, weighting=weighting,
                                         **{k: value[k] for k in ['E_conditional','G_conditional','R_conditional',
                                                                  'E_low','E_high','G_low','G_high','R_low','R_high']}))
            for transition, value in summary['goal'].get('by_transition', {}).items():
                transitions.append(dict(job=job, stage=stage, arm=arm, transition=transition,
                                        Q_all_attempts=value['Q'], worst_side_Q=value['worst_side']))
        lines += [f'## {job} / {stage}', '',
                  f"完整阶段{item['rows']:,}条；原始SHA、完整计划与交付身份已验收。原生采样分析：{'完成' if statistics else '待完成'}。", '',
                  '| 模式 | 全尝试目标Q（条件对等权） | 四侧最差Q的加权均值 | 目标合法/全部 | 行为合法/全部 |',
                  '|---|---:|---:|---:|---:|']
        for summary in audit['summaries']:
            goal = summary['counts']['goal']
            b = summary['counts'].get('behavior')
            lines.append(f"| {summary['arm']} | {number(summary['goal']['pair_equal']['Q'])} | "
                         f"{number(summary['goal']['pair_equal']['worst_side'])} | {goal['valid']}/{goal['n']} | "
                         f"{str(b['valid'])+'/'+str(b['n']) if b else '不适用'} |")
        if stage == 'canonical':
            lines += ['', '| 模式 | 权重 | 条件E | 条件G | 条件R |', '|---|---|---:|---:|---:|']
            for row in [x for x in behavior if x['job'] == job]:
                lines.append(f"| {row['arm']} | {row['weighting']} | {number(row['E_conditional'])} | "
                             f"{number(row['G_conditional'])} | {number(row['R_conditional'])} |")
            lines += ['', '条件对等权的全尝试缺失质量差值范围（开启−关闭）：', '']
            for metric, bounds in audit['delta_all_attempt_bounds'].items():
                lines.append(f"- {metric}：[{number(bounds[0])}, {number(bounds[1])}]")
            lines += ['', '以上范围只处理缺失行动的未知质量，不包含采样误差，也不是置信区间。']
        if any(s['goal'].get('by_transition') for s in audit['summaries']):
            lines += ['', '| 模式 | 最优集合变化 | 全尝试Q | 四侧最差Q的加权均值 |', '|---|---|---:|---:|']
            for row in [x for x in transitions if x['job'] == job and x['stage'] == stage]:
                lines.append(f"| {row['arm']} | {row['transition']} | {number(row['Q_all_attempts'])} | {number(row['worst_side_Q'])} |")
        if statistics:
            lines += ['', '| 类型 | 权重 | 指标 | 开启−关闭 | 经验重采样范围 |', '|---|---|---|---:|---|']
            with (stats_path.parent/'sampling_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                for row in csv.DictReader(stream):
                    def val(key):
                        return number(float(row[key])) if row.get(key) not in ['',None] else '未定义'
                    lines.append(f"| {row['kind']} | {row['weighting']} | {row['metric']} | {val('delta')} | "
                                 f"[{val('empirical_low')}, {val('empirical_high')}] |")
            lines += ['', '经验范围使用500次全尝试共享提示重采样；未定义抽样保留，范围不可得时不补零。该范围不是校准置信区间。点态Q保守界、16/32/64/128次敏感性、条件E置换参照及全部目标/来源/任务切片见本阶段native_statistics结果。']
        lines += ['']
    training_path=OUT/'statistics_v2/paired_training_sampling.csv'
    training_rows=[]
    if (training_path.parent/'COMPLETE.json').exists():
        with training_path.open(encoding='utf-8-sig',newline='') as stream:
            training=list(csv.DictReader(stream))
        lines += ['## 旧训练的固定输入采样核查', '',
                  '下表逐模型保留四个留出来源×三个种子的全部匹配比较。差值均为JS−SFT；E/G下降为改善，Q上升为改善。PGG没有目标真值，因此Q仅9项。来源和种子共享材料，这些计数不是独立研究次数。', '',
                  '| 模型 | 指标 | 差值负/零/正 | 经验范围全负/含零/全正 | Q点态保守范围全负/含零/全正 |',
                  '|---|---|---|---|---|']
        for model in sorted({row['model'] for row in training}):
            for metric in ['E','G','Q']:
                rows=[row for row in training if row['model']==model and row['metric']==metric]
                delta=[float(row['JS_minus_SFT']) for row in rows]
                signs=[sum(value<0 for value in delta),sum(value==0 for value in delta),sum(value>0 for value in delta)]
                ranges=[]
                for low,high in [('empirical_resampling_low','empirical_resampling_high'),('Q_bound95_low','Q_bound95_high')]:
                    available=[(float(row[low]),float(row[high])) for row in rows if row[low] and row[high]]
                    counts=[sum(b<0 for a,b in available),sum(a<=0<=b for a,b in available),sum(a>0 for a,b in available)]
                    ranges.append('/'.join(map(str,counts)) if available else '不适用')
                lines.append(f"| {model} | {metric} | {'/'.join(map(str,signs))} | {ranges[0]} | {ranges[1]} |")
                training_rows.append(dict(model=model,metric=metric,n_matched_panels=len(rows),negative=signs[0],zero=signs[1],positive=signs[2],empirical_range_counts=ranges[0],Q_pointwise_bound_counts=ranges[1]))
        lines += ['', '经验范围不是校准置信区间。Q保守范围依赖固定提示上的有界独立抽样假设，合并共享提示系数，未作多重比较或新来源保证；不能把范围未跨零的数量写成普遍显著。原有三种子、来源和所有失败完整保留，未调λ或挑种子。', '']
        write_csv(OUT/'training_sampling_readout.csv',training_rows)
    lines += ['## 解读范围', '',
              'Q以全部尝试为分母，截断、类型/标签/菜单失败均保留。E/G/R按合法行动子集计算，明确标为conditional；无效行动不补零。G≤2E，R没有统一优劣方向。', '',
              '条件对等权为主，来源和任务等权完整保留。目标Q权重与按提示平摊的正确率不同。经验重采样范围只描述固定提示的有限采样敏感性，不能保证新来源；与缺失质量范围分开报告。', '',
              '参数化诊断是五类已定义机制的新参数条件，不是新经验来源。目标最优集合互斥、相交和相同均保留；共享提示和四元组引用不作为独立情境数。互斥目标的两调用乘积仅为独立调用概率估计。', '',
              '充分推理若提高Q并降低E/G，应承认其正面结果。不能将关闭思考的失败外推为模型能力上限；自由行为低熵不自动是错误，高R也不自动是正确。', '']
    for name, rows in [('accepted_behavior.csv', behavior), ('accepted_goal.csv', goals),
                       ('accepted_delivery.csv', deliveries), ('accepted_parameter_transitions.csv', transitions)]:
        write_csv(OUT/name, rows)
    path = OUT/'已验收结果与正文表.md'
    if path.exists():
        history=OUT/'report_history';history.mkdir(exist_ok=True)
        old=history/(sha(path)[:16]+'.md')
        if not old.exists():old.write_bytes(path.read_bytes())
    path.write_text('\n'.join(lines), encoding='utf-8')
    (OUT/'confirmation_table_provenance.json').write_text(json.dumps(dict(
        receipt_sha256=sha(R/'local_mirror_acceptance.json'), writer_sha256=sha(Path(__file__)),
        stages=provenance, training_sampling_sha256=sha(training_path) if training_rows else None), indent=2), encoding='utf-8')
    print('Wrote accepted confirmation tables:', len(provenance), 'stages')


if __name__ == '__main__':
    main()
