"""Bounded native-runner comparisons, after documented environment validation."""
from pathlib import Path
import json
import subprocess
import time
import shutil

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/engineai_official_v1'
RUN=ROOT/'runs/engineai_official_v1'

def main():
    if (REC/'queue_started.json').exists():raise FileExistsError('Do not duplicate existing queue')
    if not (RUN/'witness_independent_v2/results.json').exists():raise RuntimeError('Independent witness required')
    if shutil.disk_usage(ROOT).free/2**30<60:raise RuntimeError('Disk reserve below60GiB')
    (REC/'queue_started.json').write_text(json.dumps({'time':time.time(),'gpu':1,'budget_per_train':614400,'purpose':'native runner calibration and reference-fit; not data utility'},indent=2))
    jobs=[('official_full_eval','eval','third_party/engineai_rl_lab/datasets/tracking/pm01/dance.npz','cache/engineai_overlay/dance.pt')]
    for arm in ['raw_ik','joint_constrained']:
        motion=f'runs/engineai_single_v1/tracking_references_50hz/sample_000_actor0_{arm}.npz'
        jobs.extend([(f'{arm}_prior_eval','eval',motion,'cache/engineai_overlay/dance.pt'),(f'{arm}_train','train',motion,'cache/engineai_overlay/dance.pt'),(f'{arm}_eval','eval',motion,str(RUN/f'{arm}_train/model_99.pt'))])
    records=[]
    for name,mode,motion,checkpoint in jobs:
        out=RUN/name
        cmd=['bash','scripts/run_engineai_official.sh','scripts/engineai_official_validate.py','--mode',mode,'--motion',motion,'--checkpoint',checkpoint,'--output',str(out),'--num_envs','256' if mode=='train' else '32','--iterations','100','--seed','0','--headless','--device','cuda:1']
        start=time.time()
        with (REC/f'{name}.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        expected=out/('COMPLETE.json' if mode=='train' else 'results.json')
        ok=code==0 and expected.exists() and not (RUN/(name+'_FAILED.txt')).exists()
        records.append({'name':name,'exit_code':code,'complete':ok,'seconds':time.time()-start,'command':cmd})
        (REC/'queue_status.json').write_text(json.dumps(records,indent=2))
        if not ok:raise RuntimeError(f'Failed {name}')
        if name=='official_full_eval' and json.loads(expected.read_text())['completion_fraction']<.9:
            raise RuntimeError('Official example completion<90%; stop before generated-motion fitting')
    (REC/'PILOT_COMPLETE.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
