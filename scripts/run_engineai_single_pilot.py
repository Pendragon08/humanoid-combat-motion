"""Sequential bounded PM01 pilot on GPU1, with explicit completion checks."""
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/engineai_single_v1'
LOG=ROOT/'records/engineai_single_v1'
def main():
    if (LOG/'queue_started.json').exists():raise FileExistsError('Do not duplicate existing pilot queue')
    (LOG/'queue_started.json').write_text(json.dumps({'time':time.time(),'gpu':1,'steps_per_training_run':300*256*24,'jobs':['official','raw_ik','joint_constrained']}))
    jobs=[('official',ROOT/'third_party/engineai_rl_lab/datasets/tracking/pm01/dance.npz')]+[(arm,RUN/'tracking_references'/f'sample_000_actor0_{arm}.npz') for arm in ['raw_ik','joint_constrained']]
    records=[]
    for name,motion in jobs:
        for mode in ['train','eval']:
            out=RUN/f'{name}_{mode}'
            cmd=['bash','scripts/run_engineai.sh','scripts/engineai_single_tracking.py','--mode',mode,'--motion',str(motion),'--output',str(out),'--num_envs','256' if mode=='train' else '32','--iterations','300','--seed','0','--headless','--device','cuda:1']
            if mode=='eval':
                checkpoints=list((RUN/f'{name}_train').glob('model_*.pt'))
                checkpoint=max(checkpoints,key=lambda x:int(x.stem.split('_')[-1]))
                cmd+=['--checkpoint',str(checkpoint)]
            start=time.time()
            with (LOG/f'{name}_{mode}.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            expected=out/('COMPLETE.json' if mode=='train' else 'results.json')
            ok=code==0 and expected.exists() and not (out.parent/(out.name+'_FAILED.txt')).exists()
            records.append({'job':name,'mode':mode,'exit_code':code,'complete':ok,'seconds':time.time()-start,'command':cmd})
            (LOG/'queue_status.json').write_text(json.dumps(records,indent=2))
            if not ok:raise RuntimeError(f'Failed {name} {mode}; inspect preserved log before retry')
    (LOG/'PILOT_COMPLETE.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
