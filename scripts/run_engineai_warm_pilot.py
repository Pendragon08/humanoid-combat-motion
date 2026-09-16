"""Matched public-prior adaptation after scratch calibration proved inadequate."""
import json
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/engineai_single_v1';REC=ROOT/'records/engineai_single_v1'
def main():
    if (REC/'warm_queue_started.json').exists():raise FileExistsError('Existing queue')
    (REC/'warm_queue_started.json').write_text(json.dumps({'seed':0,'iterations':100,'environments':256,'steps_per_run':614400,'prior':'official public PM01 dance actor+critic','learning_rate':.001,'reset_std':.3,'claim':'matched reference-fit pilot; no held-out utility','time':time.time()},indent=2))
    records=[]
    for arm in ['raw_ik','joint_constrained']:
        for mode in ['train','eval']:
            out=RUN/f'warm_{arm}_{mode}'
            cmd=['bash','scripts/run_engineai.sh','scripts/engineai_single_tracking.py','--mode',mode,'--motion',str(RUN/'tracking_references_50hz'/f'sample_000_actor0_{arm}.npz'),'--output',str(out),'--warm-start','cache/engineai_overlay/dance.pt','--num_envs','256' if mode=='train' else '32','--iterations','100','--seed','0','--headless','--device','cuda:1']
            if mode=='eval':cmd+=['--checkpoint',str(RUN/f'warm_{arm}_train/model_99.pt')]
            start=time.time()
            with (REC/f'warm_{arm}_{mode}.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            expected=out/('COMPLETE.json' if mode=='train' else 'results.json')
            ok=code==0 and expected.exists() and not (out.parent/(out.name+'_FAILED.txt')).exists()
            records.append({'arm':arm,'mode':mode,'exit_code':code,'complete':ok,'seconds':time.time()-start,'command':cmd})
            (REC/'warm_queue_status.json').write_text(json.dumps(records,indent=2))
            if not ok:raise RuntimeError(f'Failed {arm} {mode}')
    (REC/'WARM_PILOT_COMPLETE.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
