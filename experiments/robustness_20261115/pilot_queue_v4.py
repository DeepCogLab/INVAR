"""Sequential pilot queue restricted to the one initially idle physical GPU."""
import fcntl
import json
import os
import subprocess
import sys
import time
from campaign_common import R,MODELS,GPU_UUID,save


def main():
    lock=(R/'engineering_v4_queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    results=[]
    for job in ['mistral7b_v01','sotopia_mistral7b']:
        out=R/'engineering_v4'/job;out.mkdir(parents=True,exist_ok=True)
        if (out/'COMPLETE.json').exists():
            results.append(dict(job=job,status='already_complete'));continue
        process_text=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
        if any(line.split(',')[0].strip()==GPU_UUID for line in process_text.splitlines()):
            save(R/'engineering_v4_queue_status.json',dict(status='blocked_gpu_now_in_use',job=job,completed=results));return
        start=time.time()
        with (out/'worker.log').open('x') as log:
            p=subprocess.run([sys.executable,str(R/'pilot_worker_v4.py'),'--job',job],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        results.append(dict(job=job,status='completed' if p.returncode==0 else 'compatibility_failure',returncode=p.returncode,seconds=time.time()-start))
        save(R/'engineering_v4_queue_status.json',dict(status='running',completed=results,updated_unix=time.time()))
        # Retain failure and continue to independent checkpoints. No second
        # attempt is selected based on output accuracy.
        time.sleep(3)
    save(R/'engineering_v4_queue_status.json',dict(status='completed_all_pilots',completed=results,finished_unix=time.time()))


if __name__=='__main__':main()
