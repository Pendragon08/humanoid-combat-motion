"""Sequential seeds on one assigned GPU, with resumable completed-run checks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['remos','intergen'],required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    pilot=root/'runs/remocap_v2'/('pilot_'+a.method)/'TRAINING_COMPLETE.json'
    if not pilot.exists() or not json.loads(pilot.read_text())['training_pass']:
        raise RuntimeError('Pilot gate not satisfied')
    record=root/'records/remocap_20260916'
    for seed in [0,1,2]:
        output=root/'runs/remocap_v2'/f'{a.method}_seed{seed}'
        done=output/'TRAINING_COMPLETE.json'
        if done.exists():
            result=json.loads(done.read_text())
            if result['steps']!=5000 or result['seed']!=seed or not result['training_pass']:
                raise RuntimeError('Existing run does not meet declared training gate')
            continue
        with (record/f'{a.method}_seed{seed}.log').open('w') as f:
            subprocess.run([sys.executable,'-u','-m','scripts.train_remocap',
                '--data',str(root/'data/derived/remocap_body_v2'),'--output',str(output),
                '--method',a.method,'--seed',str(seed),'--steps','5000'],cwd=root,
                stdout=f,stderr=subprocess.STDOUT,check=True)
        result=json.loads(done.read_text())
        (record/f'{a.method}_seed{seed}_training.json').write_text(json.dumps(result,indent=2))
        if not result['training_pass']:
            raise RuntimeError('Training gate failed; do not automatically spend on more seeds')
        print(json.dumps(result),flush=True)
    (record/f'{a.method}_TRAINING_QUEUE_COMPLETE.json').write_text(json.dumps({'method':a.method,'seeds':[0,1,2]}))


if __name__=='__main__':main()
