"""Synthetic tests are temporary and never enter a real experiment directory."""
import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('invar_v2_score',ROOT/'score_v2.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)

def jl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')

class ScoringTests(unittest.TestCase):
    def test_accuracy_bound_handles_shared_trials_and_perfect_samples(self):
        rows=[dict(quartet_id='q',pair_id='pair',accuracy=1.)]
        boot={'q':dict(accuracy=np.ones(1000),_accuracy_sources=[('p',8)]*4)}
        shared=score.summarized(rows,boot,('accuracy',))[0]
        boot['q']['_accuracy_sources']=[(f'p{i}',8) for i in range(4)]
        distinct=score.summarized(rows,boot,('accuracy',))[0]
        self.assertLess(shared['accuracy_bound95_low'],distinct['accuracy_bound95_low'])
        self.assertLess(distinct['accuracy_bound95_low'],1.)
        self.assertEqual(shared['accuracy_bound95_high'],1.)

    def test_distance_and_shared_draws(self):
        from evaluate import w1
        values={'a':[0,0,.5,1]*64,'b':[0,.2,.2,1]*64,'c':[1.]*256,'d':[0.]*256}
        d=score.Distributions(values)
        for a in values:
            for b in values:
                point,boot,_=d.distance(a,b)
                self.assertAlmostEqual(point,w1(values[a],values[b]))
                self.assertTrue(np.allclose(boot,d.distance(b,a)[1]))
        self.assertTrue(np.all(d.distance('a','a')[1]==0))
        self.assertTrue(np.all(d.distance('c','d')[1]==1))

    def test_end_to_end_and_wrong_answers_stay_wrong(self):
        with tempfile.TemporaryDirectory(prefix='invar_synthetic_score_test_') as temp:
            r=Path(temp);model='synthetic_test'
            action_schema={'type':'object','properties':{'action':{'type':'integer','enum':[0,1]}},'required':['action'],'additionalProperties':False}
            answer_schema={'type':'object','properties':{'answer':{'type':'string','enum':['A','B']}},'required':['answer'],'additionalProperties':False}
            inputs=[];plan=[];raw=[]
            for tag in 'ABCD':
                inputs.append(dict(prompt_id=tag,output_schema=action_schema,task_type='behavior',action_space={'type':'integer','values':[0,1]},outcome={'kind':'probability','target_action':1}))
                for batch in ('main','repeat') if tag=='A' else ('main',):
                    for i in range(256):
                        row=dict(model=model,prompt_id=tag,replicate=i,batch=batch,trial_id=f'{tag}:{batch}:{i}')
                        plan.append(row);raw.append(dict(**row,response_text=json.dumps({'action':int(tag=='C')}),status='ok',finish_reason='stop',config_sha256='synthetic'))
            fq=[];gq=[]
            for typ,prefix in [('changed_fact','F'),('stable_fact','S'),('goal','G')]:
                ids={}
                for i,tag in enumerate('ABCD'):
                    pid=prefix+tag;ids[tag]=pid
                    item=dict(prompt_id=pid,output_schema=action_schema if typ=='goal' else answer_schema,task_type=typ)
                    if typ=='goal':item['expected_actions']=[i//2]
                    else:item['expected_answer']='AB'[i//2]
                    inputs.append(item)
                    for rep in range(8):
                        row=dict(model=model,prompt_id=pid,replicate=rep,batch=typ,trial_id=f'{pid}:{rep}')
                        plan.append(row);raw.append(dict(**row,response_text=json.dumps({'action':0} if typ=='goal' else {'answer':'A'}),status='ok',finish_reason='stop',config_sha256='synthetic'))
                q=dict(quartet_id=prefix,pair_id='pair',task='synthetic_task',variant_id='V1',**ids)
                if typ=='goal':q.update(policy='own',unique_optimum_switch=True);gq.append(q)
                else:q.update(probe_type=typ,factor='synthetic_factor');fq.append(q)
            jl(r/'inputs.jsonl',inputs);jl(r/'main_plan.jsonl',plan);jl(r/'outputs/main'/f'{model}.jsonl',raw)
            jl(r/'INVAR_dataset/data/quartets.jsonl',[dict(quartet_id='behavior',pair_id='pair',source_id='synthetic',task='synthetic_task',role='synthetic_role',variant_id='V1',intervention_kind='synthetic',analysis_tier='synthetic',**{t:t for t in 'ABCD'})])
            jl(r/'fact_quartets.jsonl',fq);jl(r/'goal_quartets.jsonl',gq)
            (r/'frozen_manifest.json').write_text(json.dumps(dict(models=[model],input_sha256=hashlib.sha256((r/'inputs.jsonl').read_bytes()).hexdigest())))
            old_root,old_args=score.R,sys.argv
            try:
                score.R=r;sys.argv=['score','--model',model];score.main()
            finally:score.R=old_root;sys.argv=old_args
            import csv
            with (r/'reports'/model/'behavior_quartets.csv').open(encoding='utf-8-sig') as f:report=list(csv.DictReader(f))[0]
            self.assertEqual(float(report['E']),.5);self.assertEqual(float(report['R']),.5);self.assertEqual(float(report['G']),1.)
            with (r/'reports'/model/'control_summary.csv').open(encoding='utf-8-sig') as f:controls=list(csv.DictReader(f))
            factual=next(x for x in controls if x['group']=='type' and x['value']=='changed_fact')
            self.assertEqual(float(factual['accuracy']),.5);self.assertEqual(float(factual['worst_side']),0)

if __name__=='__main__':unittest.main()
