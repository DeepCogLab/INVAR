"""Score independent role comprehension calls. Never merge with behavior scores."""
import argparse,json
from collections import Counter
from pathlib import Path
from evaluate import read_jsonl,save_csv
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('responses',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    controls={x['prompt_id']:x for x in read_jsonl(ROOT/'controls'/'role_comprehension.jsonl')}
    seen=set();rows=[]
    for row in read_jsonl(a.responses):
        p=controls[row['prompt_id']];key=(row['model'],row['prompt_id'],row['replicate'])
        if key in seen:raise ValueError('Duplicate control trial')
        seen.add(key);valid=False;correct=False
        try:
            answer=json.loads(row['response_text'])
            valid=isinstance(answer,dict) and set(answer)=={'answer'} and isinstance(answer['answer'],str)
            correct=valid and answer['answer']==p['expected_answer'] and row.get('status','ok')=='ok'
        except (ValueError,TypeError,KeyError):pass
        rows.append({'model':row['model'],'prompt_id':p['prompt_id'],'variant_id':p['variant_id'],'pair_id':p['pair_id'],'valid_schema':valid,'correct':correct})
    a.out.parent.mkdir(parents=True,exist_ok=True);save_csv(a.out,rows)
    print(json.dumps({'trials':len(rows),'correct':sum(r['correct'] for r in rows),'purpose':'role comprehension only; not social-behavior ground truth'},ensure_ascii=False))
if __name__=='__main__':main()
