"""Continue once the isolated installer finishes; validate rather than assume ready."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/engineai_official_v1'

def main():
    pid=json.loads((REC/'build_archive_process.json').read_text())['pid']
    end=time.time()+1800
    while not (REC/'INSTALL_COMPLETE.json').exists():
        if not Path(f'/proc/{pid}').exists():raise RuntimeError('Installer exited without completion marker')
        if time.time()>end:raise TimeoutError('Installation wait exceeded30min')
        time.sleep(5)
    py=str(ROOT/'.venv-engineai-official/bin/python')
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    env['TMPDIR']=str(ROOT/'cache/engineai-official/tmp')
    # The Lab pin packaging<24 requires a compatible build-only wheel utility.
    subprocess.run([py,'-m','pip','install','--no-cache-dir','wheel==0.45.1'],cwd=ROOT,env=env,check=True)
    check=subprocess.run([py,'-m','pip','check'],cwd=ROOT,env=env,text=True,capture_output=True)
    (REC/'pip_check.txt').write_text(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError('Dependency consistency check failed; inspect pip_check.txt')
    cmd=['bash','scripts/run_engineai_official.sh','scripts/engineai_official_validate.py','--mode','witness','--motion','third_party/engineai_rl_lab/datasets/tracking/pm01/dance.npz','--checkpoint','cache/engineai_overlay/dance.pt','--output','runs/engineai_official_v1/witness_primary','--num_envs','4','--headless','--device','cuda:1']
    with (REC/'witness_primary.log').open('w') as log:
        code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    result=ROOT/'runs/engineai_official_v1/witness_primary/results.json'
    if code or not result.exists():raise RuntimeError('Primary witness failed')
    print('PRIMARY_WITNESS_FINISHED',result.read_text(),flush=True)

if __name__=='__main__':main()
