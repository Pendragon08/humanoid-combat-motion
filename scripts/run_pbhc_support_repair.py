"""Run eight fixed-policy repair probes and summarize; no policy training."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_support_repair_20260917'
OUT=ROOT/'runs/pbhc_support_repair_20260917'
MUTEX=threading.Lock()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        assert not (REC/'launch.json').exists(),'Do not duplicate launch'
        assert (OUT/'references/manifest.json').exists()
        original=json.loads((ROOT/'records/pbhc_support_diagnostic_20260917/manifest.json').read_text())
        sources={j['name']:j for j in original['jobs']}
        protocol=json.loads((REC/'protocol.json').read_text());jobs=[]
        for row in protocol['records']:
            source=sources[row['name']];ref=OUT/'references'/f'sample_{row["name"]}_{row["arm"]}_actor0.npz'
            jobs.append(dict(name=row['name'],arm=row['arm'],gpu=source['gpu'],checkpoint=source['checkpoint'],checkpoint_sha256=source['checkpoint_sha256'],
                reference=str(ref.relative_to(ROOT)),reference_sha256=sha(ref),original_reference=source['reference'],
                original_reference_sha256=source['reference_sha256'],output=str((OUT/'evaluation'/f'{row["name"]}_{row["arm"]}').relative_to(ROOT))))
        write(REC/'manifest.json',dict(jobs=jobs,seed=123,num_envs=32,regime='nominal',training_updates=0,
            frozen_code_hashes={p:sha(ROOT/p) for p in ['scripts/evaluate_pbhc_support_diagnostic.py','scripts/run_pbhc_support_repair.py','scripts/summarize_pbhc_support_repair.py']}))
        with (REC/'scheduler.log').open('w') as log:
            child=subprocess.Popen([sys.executable,__file__],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(REC/'launch.json',dict(pid=child.pid,time=time.time()));print('SUPPORT_REPAIR_STARTED',child.pid);return
    lock=(REC/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((REC/'manifest.json').read_text())
    for p,d in manifest['frozen_code_hashes'].items():assert sha(ROOT/p)==d
    state=dict(status='running',jobs={},started=time.time());write(REC/'state.json',state)
    def lane(gpu):
        for job in [j for j in manifest['jobs'] if j['gpu']==gpu]:
            for key in ['reference','checkpoint','original_reference']:assert sha(ROOT/job[key])==job[key+'_sha256']
            dest=ROOT/job['output'];assert not dest.exists(),dest
            memory=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            assert memory<500,('GPU occupied',gpu,memory)
            assert shutil.disk_usage(ROOT).free>12*1024**3
            key=job['name']+'_'+job['arm']
            command=['bash','scripts/run_isaaclab.sh','scripts/evaluate_pbhc_support_diagnostic.py','--checkpoint',job['checkpoint'],
                '--motion_file',job['reference'],'--output',job['output'],'--num_envs','32','--seed','123','--regime','nominal','--headless','--device',f'cuda:{gpu}']
            with MUTEX:state['jobs'][key]={'status':'running','command':command};write(REC/'state.json',state)
            with (REC/(key+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0 and (dest/'metrics.json').exists(),(key,result.returncode)
            metric=json.loads((dest/'metrics.json').read_text());assert metric['reference_sha256']==job['reference_sha256']
            assert metric['checkpoint_sha256']==job['checkpoint_sha256'] and metric['trace_sha256']==sha(dest/'support_trace.npz')
            with MUTEX:state['jobs'][key].update(status='complete',trace_sha256=metric['trace_sha256']);write(REC/'state.json',state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(lane,g) for g in [0,1]]
        for task in tasks:task.result()
    with (REC/'analysis.log').open('w') as log:
        result=subprocess.run(['bash','scripts/run.sh','scripts/summarize_pbhc_support_repair.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0 and (REC/'summary.json').exists()
    state.update(status='complete',finished=time.time());write(REC/'state.json',state)
    write(REC/'COMPLETE.json',dict(jobs=8,training_updates=0,summary_sha256=sha(REC/'summary.json')))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if REC.exists():write(REC/'FAILED.json',dict(error=repr(exc),time=time.time()))
        raise
