"""Record inspected upstream reproduction inputs without claiming execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1];UP=ROOT/'third_party/PBHC'
REC=ROOT/'records/pbhc_official_reproduction_20260917'
paths=['INSTALL.md','humanoidverse/README.md','humanoidverse/train_agent.py','humanoidverse/sample_eps.py',
       'humanoidverse/ratio_eps.py','humanoidverse/measure_traj.py','humanoidverse/config/exp/motion_tracking.yaml',
       'humanoidverse/config/algo/mh_ppo.yaml','humanoidverse/config/rewards/motion_tracking/main.yaml',
       'humanoidverse/config/obs/motion_tracking/main.yaml','humanoidverse/config/robot/g1/g1_23dof_lock_wrist.yaml',
       'example/motion_data/Horse-stance_punch.pkl','example/pretrained_horse_stance_punch/model_33000.pt',
       'example/pretrained_horse_stance_punch/config.yaml']
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()
assert commit=='ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0'
assert not subprocess.check_output(['git','diff','--name-only'],cwd=UP,text=True).strip()
rows=[dict(path=p,sha256=hashlib.sha256((UP/p).read_bytes()).hexdigest(),bytes=(UP/p).stat().st_size) for p in paths]
output=dict(time=time.time(),upstream_commit=commit,upstream_tracked_diff_empty=True,files=rows,
    official_repository='https://github.com/TeleHuman/PBHC',paper='https://arxiv.org/html/2506.12851v1',
    observed_differences=['Previously used BeyondMimic/IsaacLab, not PBHC/IsaacGym training',
        'Original 23DOF pkl required; previous converted 29DOF NPZ is a different interface',
        'Official README 4096 environments and 50000 updates; config default one million must be overridden',
        'Official error/smoothness and ELR metrics differ from custom every-frame strict pass',
        'Paper three seeds and 1000 rollouts; present first stage one seed, final 1000 rollouts',
        'Checkpoint iter/name can lag actual final updates; observed-step receipt needed'],
    current_stage='Environment installation and future preflight queue; no full PBHC training yet',
    source_status='Official release containing both papers; explicit first-paper single-motion config selected')
(REC/'source_audit.json').write_text(json.dumps(output,indent=2))
print('PBHC_SOURCE_AUDIT_CAPTURED',len(rows))
