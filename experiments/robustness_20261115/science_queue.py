"""Single-GPU queue. Immutable protocol, owned process groups, explicit failures."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from campaign_common import R, GPU_UUID, load, save, sha


def gpu_idle():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return not any(GPU_UUID in line for line in output.splitlines())


def main():
    lock=(R/'science_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=load(R/'freeze.json')
    for name,h in cfg['files'].items():assert sha(R/name)==h,name
    assert not (R/'science_queue_status.json').exists(), 'Never silently rerun a queue'
    status=dict(status='running',freeze_sha256=sha(R/'freeze.json'),started_unix=time.time(),jobs=[])
    save(R/'science_queue_status.json',status)
    deadline=status['started_unix']+cfg['protocol']['campaign_hours']*3600
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    for job in cfg['protocol']['job_order']:
        if time.time()>=deadline:
            status['jobs'].append(dict(job=job,status='not_started_time_limit'));continue
        if not gpu_idle():
            status['jobs'].append(dict(job=job,status='not_started_gpu_in_use'))
            status['status']='blocked_gpu_in_use';break
        logs=R/'queue_logs';logs.mkdir(exist_ok=True)
        begun=time.time()
        with (logs/(job+'.log')).open('x') as f:
            p=subprocess.Popen([sys.executable,str(R/'science_worker.py'),'--job',job],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            status['current']=dict(job=job,pid=p.pid,started_unix=begun);save(R/'science_queue_status.json',status)
            try:
                code=p.wait(timeout=max(1,deadline-time.time()))
            except subprocess.TimeoutExpired:
                # Only this queue's child process group; never inspect/kill other services.
                assert p.poll() is None
                os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);code=p.wait()
                status['jobs'].append(dict(job=job,status='time_limit_partial',returncode=code,seconds=time.time()-begun))
                continue
        item=dict(job=job,returncode=code,seconds=time.time()-begun)
        if code==0 and (R/'runs'/job/'COMPLETE.json').exists():
            with (logs/(job+'_score.log')).open('x') as f:
                s=subprocess.run([sys.executable,str(R/'score.py'),'--job',job],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
            item['status']='generated_and_scored' if s.returncode==0 else 'generated_score_failed';item['score_returncode']=s.returncode
            if s.returncode==0:
                with (logs/(job+'_analysis.log')).open('x') as f:
                    an=subprocess.run([sys.executable,str(R/'analyze.py'),'--job',job],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
                item['analysis_returncode']=an.returncode
                if an.returncode:item['status']='generated_scored_analysis_failed'
        else:item['status']='worker_failed_no_reroll'
        status['jobs'].append(item);status.pop('current',None);save(R/'science_queue_status.json',status)
        # CUDA engine children normally exit with their parent; give them a bounded grace period.
        for _ in range(12):
            if gpu_idle():break
            time.sleep(5)
    if status['status']=='running':status['status']='finished'
    status.pop('current',None);status['finished_unix']=time.time();save(R/'science_queue_status.json',status)
    subprocess.run([sys.executable,str(R/'analyze.py')],cwd=R,check=True,timeout=1800)


if __name__=='__main__':main()
