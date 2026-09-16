"""Small auditable status/results snapshot; never publishes third-party weights/data."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/pbhc_beyondmimic_20260916'
REC=ROOT/'records/pbhc_beyondmimic_20260916'
snapshot=REC/'snapshot';snapshot.mkdir(parents=True,exist_ok=True)
for name in ['source_manifest.json','conversion_audit.json']:
    shutil.copyfile(OUT/name,snapshot/name)
for stage in ['smoke_v2','smoke_eval']:
    for path in (OUT/stage).glob('*.json'):
        target=snapshot/stage/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
for stage in ['training','evaluation']:
    for path in (OUT/stage).rglob('*.json'):
        target=snapshot/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
for name in ['launch.json','launch_source_hashes.json','lane0_status.json','lane1_status.json','lane0_COMPLETE.json','lane1_COMPLETE.json']:
    path=REC/name
    if path.exists():shutil.copyfile(path,snapshot/name)
states=[]
protocol=json.loads((REC/'protocol.json').read_text())
for lane,names in protocol['lanes'].items():
    for name in names:
        log=REC/(name+'_train.log');steps=0;eta=None;last_iteration=None
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-100000));text=stream.read().decode(errors='replace')
            values=re.findall(r'Total timesteps:\s*(\d+)',text)
            if values:steps=int(values[-1])
            times=re.findall(r'ETA:\s*([^\r\n]+)',text)
            if times:eta=times[-1].strip()
        evaluations=[]
        for path in sorted((OUT/'evaluation'/name).glob('*/metrics.json')):
            m=json.loads(path.read_text());evaluations.append({k:m[k] for k in [
                'checkpoint','training_receipt','completion_fraction','strict_every_frame_pass_fraction',
                'common23_strict_every_frame_pass_fraction','trajectory_mean_pass_fraction','mean_active_errors']})
        states.append({'name':name,'gpu':int(lane),'observed_environment_steps':steps,
                       'observed_updates':steps//(protocol['num_envs']*protocol['steps_per_env']),
                       'target_updates':protocol['iterations'],'training_log_eta':eta,
                       'training_complete':(OUT/'training'/name/'TRAINING_COMPLETE.json').exists(),
                       'evaluations':evaluations})
complete=all((REC/f'lane{i}_COMPLETE.json').exists() for i in [0,1])
report={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'all_complete':complete,
        'scope':'Development-only fixed-budget calibration, not generated-data utility',
        'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
        'jobs':states}
(REC/'status_snapshot.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
