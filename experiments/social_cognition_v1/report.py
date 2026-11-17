"""Continuously refreshed campaign ledger; no incomplete arm is ranked."""
import csv,json
from pathlib import Path
from run_model import R,save
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig'))) if p.exists() else []
def main():
    jobs=json.loads((R/'jobs.json').read_text());ledger=[]
    lines=['# 社会认知模型扩展测评：当前记录','',
      '只有正式全量评分验收通过的模型才进入比较。先导成绩与原生格式诊断均单列，不作最终模型排名。',
      'E/G越小表示在本任务中越稳定；R没有统一优劣方向。中文迁移、训练目标、接口差异均限制解释。','',
      '| 模型 | 规模B | 先导 | 原生诊断 | 正式调用 | 评分状态 |','|---|---:|---:|---:|---:|---|']
    for j in jobs:
        r=R/'runs'/j['id'];entry=dict(model=j['id'],size=j['size'])
        for stage in ['pilot','native','full']:
            cp=r/stage/'COMPLETE.json';progress=r/stage/'progress.json'
            entry[stage]=json.loads(cp.read_text())['rows'] if cp.exists() else (json.loads(progress.read_text())['done'] if progress.exists() else 0)
        scores=r/'full/scoring/reports'/j['id'];audit=scores/'completion_audit.json'
        entry['scoring']='passed' if audit.exists() else ('compatibility_or_infrastructure_failure' if (R/f'{j["id"]}_FAILED.json').exists() else 'pending')
        if audit.exists():
            entry['behavior']=next(x for x in csvrows(scores/'behavior_summary.csv') if x['group']=='all')
            entry['controls']={x['value']:x for x in csvrows(scores/'control_summary.csv') if x['group']=='type'}
        ledger.append(entry)
        lines.append(f'| {j["id"]} | {j["size"]} | {entry["pilot"]} | {entry["native"]} | {entry["full"]}/421504 | {entry["scoring"]} |')
    lines+=['','## 已验收全量结果','','| 模型 | E | R | G | 应变事实 | 未变事实 | 明示目标 |','|---|---:|---:|---:|---:|---:|---:|']
    for e in ledger:
        if e['scoring']!='passed':continue
        b=e['behavior'];c=e['controls']
        lines.append('| '+e['model']+' | '+' | '.join(f'{float(b[k]):.4f}' for k in ['E','R','G'])+' | '+' | '.join(f'{float(c[k]["accuracy"]):.2%}' for k in ['changed_fact','stable_fact','goal'])+' |')
    byid={e['model']:e for e in ledger}
    lines+=['','## 与精确底座比较','','以下为专门模型减底座的差值，等待双方全部验收后才计算。差值本身不等于显著性检验。','']
    for j in jobs:
        if not j['adapter']:continue
        e=byid[j['id']];b=byid[j['id']+'_base']
        if e['scoring']==b['scoring']=='passed':
            delta={k:float(e['behavior'][k])-float(b['behavior'][k]) for k in ['E','R','G']}
            delta.update({k:float(e['controls'][k]['accuracy'])-float(b['controls'][k]['accuracy']) for k in ['changed_fact','stable_fact','goal']})
            lines.append(f'- {j["id"]}：`{json.dumps(delta)}`。')
    lines+=['','完整题目、原始输出、错误重复频率和筛选规则见各模型各阶段的 `失败案例.md`、`failure_cases.json`；先导案例仅一次调用，尚未人工核验。',
        '全部原始文本保存在对应 `responses.jsonl`。兼容性失败记录在模型 FAILED 文件和 logs 中，不计入能力错误。',
        '队列结束后查 `service_restoration.json` 确认两个原服务均为 healthy。']
    save(R/'campaign_ledger.json',ledger);(R/'当前结果.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
