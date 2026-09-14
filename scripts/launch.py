"""Launch isolated GPU streams after verified archival cleanup, with durable logs."""
from pathlib import Path
import subprocess
import os
from combat.robot import ROOT

assert (ROOT/'audit/tdmpc2_archive/cleanup_verified.json').exists(), 'Requested backup+cleanup must complete first.'
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
for line in gpus.splitlines():
    idx,used,total=[int(x.strip()) for x in line.split(',')]
    if idx in [0,1]:assert used<500,(idx,used)
(ROOT/'runs').mkdir(exist_ok=True)
for seed in [0,1]:
    name=f'combat_pilot_s{seed}'
    sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
    assert '.'+name not in sessions, 'Existing session '+name
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(seed))
    cmd=f'set -o pipefail; bash scripts/run_pipeline.sh {seed} 2>&1 | tee runs/pipeline_seed{seed}.log'
    subprocess.run(['screen','-dmS',name,'bash','-c',cmd],cwd=ROOT,env=env,check=True)
    print('LAUNCHED',name,'GPU',seed)
