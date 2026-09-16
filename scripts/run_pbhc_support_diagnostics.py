"""Four bounded, read-only diagnostic evaluations; no training or mutation of inputs."""
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
REC=ROOT/'records/pbhc_support_diagnostic_20260917'
OUT=ROOT/'runs/pbhc_support_diagnostic_20260917'
LOCK=threading.Lock()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        REC.mkdir(exist_ok=False,parents=True)
        spec=json.loads((ROOT/'env-isaaclab.json').read_text())
        assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]=='072cb4cb'
        old=json.loads((ROOT/'records/pbhc_followup_20260916/manifest.json').read_text())
        jobs=[]
        for source in old['jobs']:
            if source['regime']!='nominal' or not source['checkpoint'].endswith('1999.pt'):continue
            job=source.copy();job['old_output']=job['output'];job['output']=str((OUT/job['name']).relative_to(ROOT))
            jobs.append(job)
        assert len(jobs)==4
        write(REC/'manifest.json',dict(jobs=jobs,scope='Read-only foot support and termination diagnostics; all four development motions; no training',
            seed=123,num_envs=32,env_spec_hash='072cb4cb',
            frozen_code_hashes={p:sha(ROOT/p) for p in ['scripts/evaluate_pbhc_support_diagnostic.py','scripts/run_pbhc_support_diagnostics.py']},
            contact_warning='Simulated net link force; not force truth of the reference demonstration'))
        with (REC/'scheduler.log').open('w') as log:
            child=subprocess.Popen([sys.executable,__file__],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(REC/'launch.json',dict(pid=child.pid,time=time.time()));print('SUPPORT_DIAGNOSTICS_STARTED',child.pid);return
    handle=(REC/'scheduler.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((REC/'manifest.json').read_text())
    for path,digest in manifest['frozen_code_hashes'].items():assert sha(ROOT/path)==digest,path
    state=dict(status='running',jobs={},start=time.time());write(REC/'state.json',state)
    def lane(gpu):
        for job in [j for j in manifest['jobs'] if j['gpu']==gpu]:
            out=ROOT/job['output']
            assert not out.exists(),out
            for key in ['reference','checkpoint']:assert sha(ROOT/job[key])==job[key+'_sha256']
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            assert used<500,('GPU occupied',gpu,used)
            assert shutil.disk_usage(ROOT).free>12*1024**3
            command=['bash','scripts/run_isaaclab.sh','scripts/evaluate_pbhc_support_diagnostic.py',
                '--checkpoint',job['checkpoint'],'--motion_file',job['reference'],'--output',str(out),
                '--num_envs','32','--seed','123','--regime','nominal','--headless','--device',f'cuda:{gpu}']
            with LOCK:state['jobs'][job['name']]={'status':'running','command':command};write(REC/'state.json',state)
            with (REC/(job['name']+'.log')).open('w') as log:
                run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if run.returncode or not (out/'metrics.json').exists():raise RuntimeError(f'{job["name"]}: exit {run.returncode}; inspect saved log')
            metric=json.loads((out/'metrics.json').read_text())
            assert metric['trace_sha256']==sha(out/'support_trace.npz')
            assert metric['checkpoint_sha256']==job['checkpoint_sha256'] and metric['reference_sha256']==job['reference_sha256']
            with LOCK:state['jobs'][job['name']].update(status='complete',trace_sha256=metric['trace_sha256']);write(REC/'state.json',state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,g) for g in [0,1]]
        for future in futures:future.result()
    state.update(status='complete',finish=time.time());write(REC/'state.json',state)
    write(REC/'COMPLETE.json',dict(jobs=4,training_updates=0,completed=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if REC.exists():write(REC/'FAILED.json',dict(error=repr(exc),time=time.time()))
        raise
