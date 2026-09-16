"""Launch preflighted bounded calibration lanes with persistent logs/PID records."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_beyondmimic_20260916'
OUT=ROOT/'runs/pbhc_beyondmimic_20260916'
if (REC/'launch.json').exists():raise FileExistsError('Already launched; inspect recorded PIDs')
assert (OUT/'smoke_v2/TRAINING_COMPLETE.json').exists() and (OUT/'smoke_eval/metrics.json').exists()
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert len(memory.splitlines())==2 and all(int(v)<500 for v in memory.splitlines()),memory
assert shutil.disk_usage(ROOT).free>12*1024**3
files=['scripts/'+n for n in ['prepare_pbhc_beyondmimic.py','convert_full_tracking_references.py',
       'finalize_pbhc_beyondmimic_refs.py','train_pbhc_beyondmimic.py','evaluate_pbhc_beyondmimic.py',
       'run_pbhc_beyondmimic_queue.py','launch_pbhc_beyondmimic.py','run_isaaclab.sh']]
files+=['env-isaaclab.json','records/pbhc_beyondmimic_20260916/protocol.json']
files += [str(p.relative_to(ROOT)) for p in (ROOT/'third_party/whole_body_tracking/source').rglob('*.py')]
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
(REC/'launch_source_hashes.json').write_text(json.dumps(hashes,indent=2))
states=[]
for lane in [0,1]:
    with (REC/f'lane{lane}.log').open('w') as log:
        cmd=[sys.executable,'scripts/run_pbhc_beyondmimic_queue.py','--lane',str(lane)]
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True)
    states.append({'lane':lane,'pid':proc.pid,'command':cmd,'started':time.time()})
    (REC/'launch.json').write_text(json.dumps(states,indent=2))
print('CALIBRATION_QUEUES_LAUNCHED',json.dumps(states))
