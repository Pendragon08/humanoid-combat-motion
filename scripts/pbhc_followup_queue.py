"""Resume-safe dependent diagnostics; never overlap the existing GPU queues."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/pbhc_beyondmimic_20260916'
OLD=ROOT/'records/pbhc_beyondmimic_20260916'
OUT=ROOT/'runs/pbhc_followup_20260916'
REC=ROOT/'records/pbhc_followup_20260916'
MUTEX=threading.Lock()


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    REC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    lock=(REC/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((REC/'manifest.json').read_text())
    state_path=REC/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'phase':'waiting_for_training','jobs':{},'started':time.time()}
    write(state_path,state)
    # Both completion receipts AND parent processes must be settled.
    old_launch=json.loads((OLD/'launch.json').read_text())
    while not all((OLD/f'lane{i}_COMPLETE.json').exists() for i in [0,1]) or any(alive(x['pid']) for x in old_launch):
        for i in [0,1]:
            lane=OLD/f'lane{i}_status.json'
            if lane.exists() and any(x.get('status')=='failed' for x in json.loads(lane.read_text())):
                raise RuntimeError('Predecessor failed; preserve logs and inspect')
        if not any(alive(x['pid']) for x in old_launch) and not all((OLD/f'lane{i}_COMPLETE.json').exists() for i in [0,1]):
            raise RuntimeError('Predecessor exited without completion receipts')
        time.sleep(15)
    for path,value in manifest['frozen_code_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    state['phase']='diagnostics';write(state_path,state)

    def run_lane(lane):
        for job in [x for x in manifest['jobs'] if x['gpu']==lane]:
            key=job['id'];dest=ROOT/job['output'];marker=dest/'metrics.json'
            checkpoint=ROOT/job['checkpoint'];ref=ROOT/job['reference']
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==job['checkpoint_sha256']
            assert hashlib.sha256(ref.read_bytes()).hexdigest()==job['reference_sha256']
            if marker.exists():
                result=json.loads(marker.read_text())
                assert result['checkpoint_sha256']==job['checkpoint_sha256']
                assert result['reference_sha256']==job['reference_sha256']
                assert result['evaluation_regime']==job['regime']
                with MUTEX:state['jobs'][key]={'status':'complete','reused_verified_receipt':True};write(state_path,state)
                continue
            if dest.exists():raise RuntimeError(f'Incomplete output needs inspection: {dest}')
            wait_start=time.time()
            while True:
                memory=int(subprocess.check_output(['nvidia-smi',f'--id={lane}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if memory<500:break
                if time.time()-wait_start>1800:raise RuntimeError('GPU occupied for 30min; stop without pre-emption')
                time.sleep(10)
            if shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError('Disk reserve below12GiB')
            cmd=['bash','scripts/run_isaaclab.sh','scripts/evaluate_pbhc_followup.py',
                 '--checkpoint',str(checkpoint),'--motion_file',str(ref),'--output',str(dest),
                 '--regime',job['regime'],'--num_envs','32','--seed','123','--headless','--device',f'cuda:{lane}']
            with MUTEX:state['jobs'][key]={'status':'running','command':cmd,'started':time.time()};write(state_path,state)
            with (REC/(key+'.log')).open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            ok=result.returncode==0 and marker.exists()
            with MUTEX:
                state['jobs'][key].update(status='complete' if ok else 'failed',exit_code=result.returncode,finished=time.time())
                write(state_path,state)
            if not ok:raise RuntimeError(key+' failed; no automatic blind retry')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_lane,i) for i in [0,1]]
        for future in futures:future.result()
    state['phase']='analysis_and_replay';write(state_path,state)
    for script in ['scripts/summarize_pbhc_followup.py','scripts/render_pbhc_followup.py']:
        with (REC/(Path(script).stem+'.log')).open('w') as log:
            result=subprocess.run(['bash','scripts/run.sh',script],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(script+' failed')
    assert (REC/'summary.json').exists() and (OUT/'videos/COMPLETE.json').exists()
    state['phase']='complete';state['finished']=time.time();write(state_path,state)
    write(REC/'COMPLETE.json',{'diagnostics':len(manifest['jobs']),'report':str(REC/'report.md'),'videos':str(OUT/'videos'),'completed':time.time()})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(REC/'FAILED.json',{'error':repr(exc),'time':time.time()})
        raise
