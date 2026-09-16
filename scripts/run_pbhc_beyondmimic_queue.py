"""Two disjoint GPU lanes; bounded full-motion calibration, never full utility jobs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/pbhc_beyondmimic_20260916'
RECORD=ROOT/'records/pbhc_beyondmimic_20260916'


def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',type=int,choices=[0,1],required=True);a=p.parse_args()
    RECORD.mkdir(parents=True,exist_ok=True)
    lock=(RECORD/f'lane{a.lane}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (OUT/'smoke_v2/TRAINING_COMPLETE.json').exists()
    assert (OUT/'smoke_eval/metrics.json').exists()
    audit=json.loads((OUT/'conversion_audit.json').read_text())
    assert len(audit['records'])==4
    protocol=json.loads((RECORD/'protocol.json').read_text())
    if (RECORD/f'lane{a.lane}_status.json').exists():raise FileExistsError('Lane already started; inspect rather than duplicate')
    names=protocol['lanes'][str(a.lane)]
    states=[]
    for name in names:
        output=OUT/'training'/name
        if output.exists():raise FileExistsError(output)
        if shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError('Disk reserve below 12 GiB')
        ref=OUT/'references_full'/('sample_'+name+'_actor0.npz')
        expected=next(r['reference_sha256'] for r in audit['records'] if r['name']==name)
        assert hashlib.sha256(ref.read_bytes()).hexdigest()==expected
        commands=[('train',['scripts/train_pbhc_beyondmimic.py','--bundle',str(OUT/'bundles'/(name+'.npz')),
                            '--output',str(output),'--mode','upstream','--num_envs',str(protocol['num_envs']),
                            '--iterations',str(protocol['iterations']),'--seed','0'])]
        for checkpoint in protocol['evaluate_checkpoints']:
            commands.append(('eval_'+checkpoint,['scripts/evaluate_pbhc_beyondmimic.py',
                '--checkpoint',str(output/checkpoint),'--motion_file',str(ref),
                '--output',str(OUT/'evaluation'/name/checkpoint.removesuffix('.pt')),
                '--num_envs','32','--seed','123']))
        for label,args in commands:
            cmd=['bash','scripts/run_isaaclab.sh']+args+['--headless','--device',f'cuda:{a.lane}']
            item=dict(motion=name,stage=label,command=cmd,status='running',started=time.time())
            states.append(item)
            status=RECORD/f'lane{a.lane}_status.json'
            status.write_text(json.dumps(states,indent=2))
            with (RECORD/f'{name}_{label}.log').open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            item.update(exit_code=result.returncode,finished=time.time(),status='complete' if result.returncode==0 else 'failed')
            status.write_text(json.dumps(states,indent=2))
            if result.returncode:raise RuntimeError(f'{name} {label} failed; no automatic retry')
            marker=output/'TRAINING_COMPLETE.json' if label=='train' else Path(args[args.index('--output')+1])/'metrics.json'
            if not marker.exists():raise RuntimeError('Missing actual result receipt')
    (RECORD/f'lane{a.lane}_COMPLETE.json').write_text(json.dumps({'motions':names,'completed':time.time()}))


if __name__=='__main__':main()
