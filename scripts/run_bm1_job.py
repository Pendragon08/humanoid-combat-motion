"""Detached, bounded supervisor: smoke, then one fixed BM1 evaluation arm."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--arm',choices=['baseline','caption_adapter'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    m=json.loads(a.manifest.read_text());job=m['jobs'][a.arm];dest=root/job['directory']
    dest.mkdir(parents=True,exist_ok=False)
    state={'arm':a.arm,'pid':os.getpid(),'gpu':job['gpu'],'status':'starting',
           'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}
    def update(**fields):
        state.update(fields);temp=dest/'status.tmp';temp.write_text(json.dumps(state,indent=2));os.replace(temp,dest/'status.json')
    update()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(job['gpu']);env['PYTHONHASHSEED']=str(m['seed'])
    gpu=subprocess.check_output(['nvidia-smi','-i',str(job['gpu']),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(gpu.strip())>=500:
        update(status='blocked_gpu_busy');return
    for name,digest in m['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            update(status='failed_code_drift',path=name);return
    base=['bash','scripts/run_timotion.sh','scripts/run_bm1_generation_benchmark.py','--arm',a.arm,'--seed',str(m['seed'])]
    if job.get('adapter'):base+=['--adapter',job['adapter']]
    start=time.monotonic()
    for phase in ('smoke','evaluation'):
        command=base+['--output',str(dest/phase),'--repeats',str(1 if phase=='smoke'else m['repetitions'])]
        if phase=='smoke':command+=['--smoke-only']
        update(status='running',phase=phase,command=command)
        with (dest/(phase+'.console.log')).open('wb')as log:
            proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            update(child_pid=proc.pid)
            try:rc=proc.wait(timeout=900 if phase=='smoke'else m['timeout_seconds'])
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                update(status='timed_out',elapsed_s=time.monotonic()-start);return
        marker=dest/phase/('SMOKE_COMPLETE.json'if phase=='smoke'else'EVALUATION_COMPLETE.json')
        if rc or not marker.exists():
            update(status='failed',exit_code=rc,missing_marker=not marker.exists(),elapsed_s=time.monotonic()-start);return
    update(status='complete',elapsed_s=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__=='__main__':main()
