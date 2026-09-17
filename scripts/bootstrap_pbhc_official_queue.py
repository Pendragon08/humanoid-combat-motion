"""Wait for bounded environment build, compose official config, then preflight only."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_official_reproduction_20260917'
def write(name,value):
    p=REC/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def main():
    handle=(REC/'bootstrap.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (REC/'bootstrap_state.json').exists(),'Do not duplicate bootstrap'
    write('bootstrap_state.json',dict(status='waiting_for_environment_install',started=time.time()))
    deadline=time.time()+7200
    while True:
        log=(REC/'build_v2.log').read_text(errors='replace')
        if 'PBHC_ENV_INSTALLED_NOT_YET_VALIDATED' in log:break
        if 'Traceback (most recent call last)' in log:raise RuntimeError('Build failed; preserve logs before corrective action')
        if time.time()>deadline:raise TimeoutError('Environment build did not complete in two hours')
        time.sleep(20)
    spec=json.loads((ROOT/'env-pbhc-training.json').read_text())
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
    with (REC/'pip_check.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv-pbhc-training/bin/python'),'-m','pip','check'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    cfg=json.loads((ROOT/'configs/pbhc_official_reproduction_v1.json').read_text())
    args=['bash','scripts/run_pbhc_training.sh','humanoidverse/train_agent.py','--cfg','job','--resolve']+cfg['training_overrides']+[
        'num_envs=4096','algo.config.num_learning_iterations=50000','robot.motion.motion_file='+cfg['motion'],'+device=cuda:0']
    with (REC/'resolved_official_config.yaml').open('w') as out, (REC/'config_compose.log').open('w') as err:
        subprocess.run(args,cwd=ROOT,stdout=out,stderr=err,check=True)
    import yaml
    resolved=yaml.safe_load((REC/'resolved_official_config.yaml').read_text())
    assert resolved['num_envs']==4096 and resolved['algo']['config']['num_learning_iterations']==50000
    assert resolved['algo']['_target_']=='humanoidverse.agents.mh_ppo.mh_ppo.MHPPO'
    assert resolved['rewards']['adaptive_tracking_sigma']['enable'] is True
    assert resolved['robot']['motion']['motion_file']==cfg['motion']
    write('installation_receipt.json',dict(spec_hash=spec_hash,status='installed_configuration_composed_NOT_physics_validated',
        configuration_sha256=hashlib.sha256((REC/'resolved_official_config.yaml').read_bytes()).hexdigest(),time=time.time()))
    write('bootstrap_state.json',dict(status='preflight_queued_after_existing_jobs',spec_hash=spec_hash,time=time.time()))
    with (REC/'preflight_queue.log').open('w') as log:
        subprocess.run(['bash','scripts/run.sh','scripts/run_pbhc_official_reproduction.py','--stage','preflight'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write('bootstrap_state.json',dict(status='awaiting_independent_witness_before_training',spec_hash=spec_hash,time=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('bootstrap_FAILED.json',dict(error=repr(exc),time=time.time()));raise
