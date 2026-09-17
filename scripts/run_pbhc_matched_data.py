"""Two GPU lanes, raw then gated, followed immediately by all final evaluations."""
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_matched_data_20260917'
OUT=ROOT/'runs/pbhc_matched_data_20260917'
LOCK=threading.Lock()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def main():
    handle=(REC/'scheduler.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((REC/'manifest.json').read_text());cfg=manifest['config']
    assert not (REC/'state.json').exists(),'Inspect stopped queue; no blind resume'
    for path,digest in manifest['frozen_code_hashes'].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/'records/pbhc_quality_gate_20260917/selection.json')==manifest['selection_sha256']
    state=dict(status='running',started=time.time(),jobs={});write(REC/'state.json',state)
    def run(key,command,gpu,marker):
        assert not marker.exists(),marker
        wait=time.time()
        while True:
            used=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used<500:break
            if time.time()-wait>600:raise RuntimeError('GPU busy for ten minutes; no preemption')
            time.sleep(5)
        assert shutil.disk_usage(ROOT).free>12*1024**3
        cmd=['bash','scripts/run_isaaclab.sh']+command+['--headless','--device',f'cuda:{gpu}']
        with LOCK:state['jobs'][key]=dict(status='running',command=cmd,started=time.time());write(REC/'state.json',state)
        with (REC/(key+'.log')).open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        ok=result.returncode==0 and marker.exists()
        with LOCK:state['jobs'][key].update(status='complete' if ok else 'failed',finished=time.time(),exit_code=result.returncode);write(REC/'state.json',state)
        if not ok:raise RuntimeError(key+' failed; preserve log, no auto retry')
    def lane(gpu):
        for job in [j for j in manifest['jobs'] if j['gpu']==gpu]:
            name=job['name'];arm=job['arm'];output=ROOT/job['training_output'];assert not output.exists()
            assert sha(ROOT/job['bundle'])==job['bundle_sha256']
            for ref in job['references'].values():assert sha(ROOT/ref['path'])==ref['sha256']
            cmd=['scripts/train_pbhc_matched_data.py','--bundle',job['bundle'],'--output',str(output),
                 '--mode','upstream','--num_envs',str(cfg['num_envs']),'--iterations',str(cfg['iterations']),'--seed',str(cfg['training_seed'])]
            if arm=='quality_gated':cmd+=['--match-initialization',str(OUT/'training'/f'{name}_raw'/'initialization.json')]
            run(f'{name}_{arm}_train',cmd,gpu,output/'TRAINING_COMPLETE.json')
            receipt=json.loads((output/(cfg['final_checkpoint']+'.receipt.json')).read_text())
            assert receipt['completed_updates']==cfg['iterations']
            assert receipt['environment_steps']==cfg['iterations']*cfg['num_envs']*cfg['steps_per_env']
            for target in cfg['evaluation_references']:
                for regime in cfg['evaluation_regimes']:
                    key=f'{name}_{arm}_on_{target}_{regime}';dest=OUT/'evaluation'/key
                    evaluator='scripts/evaluate_pbhc_beyondmimic.py' if regime=='randomized' else 'scripts/evaluate_pbhc_followup.py'
                    args=[evaluator,'--checkpoint',str(output/cfg['final_checkpoint']),'--motion_file',job['references'][target]['path'],
                          '--output',str(dest),'--num_envs',str(cfg['evaluation_environments']),'--seed',str(cfg['evaluation_seed'])]
                    if regime!='randomized':args+=['--regime',regime]
                    run(key,args,gpu,dest/'metrics.json')
                    metric=json.loads((dest/'metrics.json').read_text())
                    assert metric['checkpoint_sha256']==sha(output/cfg['final_checkpoint']) and metric['reference_sha256']==job['references'][target]['sha256']
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,g) for g in [0,1]]
            for future in futures:future.result()
        with (REC/'summary.log').open('w') as log:
            result=subprocess.run(['bash','scripts/run.sh','scripts/summarize_pbhc_matched_data.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0 and (REC/'summary.json').exists()
        state.update(status='complete',finished=time.time());write(REC/'state.json',state)
        write(REC/'COMPLETE.json',dict(training_jobs=4,evaluations=16,summary_sha256=sha(REC/'summary.json')))
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));write(REC/'state.json',state);raise

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(REC/'FAILED.json',dict(error=repr(exc),time=time.time()));raise
