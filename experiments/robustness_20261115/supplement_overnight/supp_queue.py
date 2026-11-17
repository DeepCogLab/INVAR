"""Wait for the original campaign, then run owned children until 07:00 SG."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from supp_common import *


def idle():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return not any(GPU_UUID in line for line in text.splitlines())


def record_closed_partial(job):
    d=S/'runs'/job;p=d/'responses.jsonl'
    if p.exists() and not (d/'COMPLETE.json').exists():
        with p.open('rb') as f:rows=sum(1 for line in f if line.strip())
        save(d/'PARTIAL_CLOSED.json',dict(status='closed_incomplete_raw_not_scientific_completion',rows=rows,raw_sha256=sha(p),closed_unix=time.time()))


def main():
    lock=(S/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=verify_freeze();deadline=cfg['gpu_deadline_unix']-20;status_path=S/'queue_status.json'
    assert not status_path.exists(),'Never restart or reroll an existing supplementary queue'
    status=dict(status='waiting_original_queue',freeze_sha256=sha(S/'freeze.json'),started_unix=time.time(),jobs=[])
    save(status_path,status)
    while time.time()<deadline:
        original=load(S.parent/'science_queue_status.json')
        assert sha(S.parent/'freeze.json')==cfg['main_freeze_sha256']
        if original['status']=='running':time.sleep(10);continue
        # The original parent retains its exclusive lock through final CPU work.
        probe=(S.parent/'science_queue.lock').open('r')
        try:fcntl.flock(probe,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:probe.close();time.sleep(10);continue
        probe.close()
        if original['status']!='finished':
            status['status']='not_started_original_queue_abnormal';save(status_path,status);return
        break
    logs=S/'queue_logs';logs.mkdir(exist_ok=True)
    status['status']='running';save(status_path,status)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    for job in cfg['job_order']:
        if time.time()>=deadline:status['jobs'].append(dict(job=job,status='not_started_time_limit'));continue
        if not idle():status['status']='blocked_gpu1_in_use';status['jobs'].append(dict(job=job,status='not_started_gpu_in_use'));break
        with (logs/(job+'.log')).open('x') as f:
            begin=time.time();p=subprocess.Popen([sys.executable,str(S/'supp_worker.py'),'--job',job],cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            status['current']=dict(job=job,pid=p.pid,started_unix=begin);save(status_path,status)
            try:code=p.wait(timeout=max(1,deadline-time.time()))
            except subprocess.TimeoutExpired:
                assert p.poll() is None;os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
                record_closed_partial(job)
                status['jobs'].append(dict(job=job,status='time_limit_partial',returncode=code,seconds=time.time()-begin));status.pop('current',None);save(status_path,status);continue
        item=dict(job=job,status='generated' if code==0 and (S/'runs'/job/'COMPLETE.json').exists() else 'failed_no_retry',returncode=code,seconds=time.time()-begin)
        if item['status']=='generated':
            with (logs/(job+'_score.log')).open('x') as f:
                check=subprocess.run([sys.executable,str(S/'supp_score.py'),'--job',job],cwd=S,stdout=f,stderr=subprocess.STDOUT,timeout=600)
            item['score_returncode']=check.returncode
            item['status']='generated_and_scored' if check.returncode==0 else 'generated_score_failed'
        else:record_closed_partial(job)
        status['jobs'].append(item);status.pop('current',None);save(status_path,status)
        for _ in range(12):
            if idle():break
            time.sleep(5)
    if status['status']=='running':status['status']='finished'
    status.pop('current',None);status['finished_unix']=time.time();save(status_path,status)


if __name__=='__main__':main()
