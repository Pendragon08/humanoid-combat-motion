"""Freeze a twelve-job post-training diagnostic manifest, with finite scope."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/pbhc_beyondmimic_20260916'
OLD=ROOT/'records/pbhc_beyondmimic_20260916'
REC=ROOT/'records/pbhc_followup_20260916'
if (REC/'manifest.json').exists():raise FileExistsError('Manifest exists; resume scheduler rather than re-launch')
assert all((OLD/f'lane{i}_COMPLETE.json').exists() for i in [0,1])
REC.mkdir(parents=True,exist_ok=True)
protocol=json.loads((OLD/'protocol.json').read_text());jobs=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for lane,names in protocol['lanes'].items():
    for name in names:
        ref=BASE/'references_full'/('sample_'+name+'_actor0.npz')
        for regime,checkpoint in [('no_push','model_1999.pt'),('nominal','model_1999.pt'),('nominal','model_1000.pt')]:
            cp=BASE/'training'/name/checkpoint
            assert cp.exists() and Path(str(cp)+'.receipt.json').exists()
            id=f'{name}_{regime}_{checkpoint[:-3]}'
            jobs.append(dict(id=id,gpu=int(lane),name=name,regime=regime,
                checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=sha(cp),
                reference=str(ref.relative_to(ROOT)),reference_sha256=sha(ref),
                output='runs/pbhc_followup_20260916/evaluation/'+id))
scripts=['evaluate_pbhc_followup.py','pbhc_followup_queue.py','summarize_pbhc_followup.py','render_pbhc_followup.py']
manifest={'scope':'Post-training disturbance and convergence diagnostics; no new policy training or altered success thresholds',
          'parent':'pbhc_beyondmimic_20260916','jobs':jobs,'num_environments':32,'seed':123,
          'no_push':'External pushes disabled; original material/COM/joint-default/reset randomization retained',
          'nominal':'No pushes and no startup/reset perturbations; reference initial velocities retained',
          'interpretation':'Diagnostic ablation, not substitute for primary randomized results; 32 nominal environments are not independent motion sources',
          'future_data_utility':'Requires validated tracker and independent data protocol; not automatically launched solely because process exits0',
          'frozen_code_hashes':{'scripts/'+p:sha(ROOT/'scripts'/p) for p in scripts}}
(REC/'manifest.json').write_text(json.dumps(manifest,indent=2))
with (REC/'scheduler.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,'scripts/pbhc_followup_queue.py'],cwd=ROOT,stdin=subprocess.DEVNULL,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(REC/'launch.json').write_text(json.dumps({'pid':proc.pid,'time':time.time(),'jobs':len(jobs)},indent=2))
print('FOLLOWUP_QUEUE_STARTED',proc.pid,len(jobs))
