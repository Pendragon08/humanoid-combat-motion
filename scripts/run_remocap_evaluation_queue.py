"""Wait for this method's declared training queue, then evaluate frozen models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['remos','intergen'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    record=root/'records/remocap_20260916';begin=time.monotonic()
    while not (record/f'{a.method}_TRAINING_QUEUE_COMPLETE.json').exists():
        log=record/f'queue_{a.method}.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text():
            raise RuntimeError('Training queue failed; evaluation cancelled')
        if time.monotonic()-begin>10800:raise TimeoutError('Training completion not observed within 3 hours')
        time.sleep(30)
    for seed in [0,1,2]:
        run=root/'runs/remocap_v2'/f'{a.method}_seed{seed}'
        output=root/'runs/remocap_v2'/f'{a.method}_seed{seed}_evaluation'
        done=output/'EVALUATION_COMPLETE.json'
        if not done.exists():
            with (record/f'{a.method}_seed{seed}_evaluation.log').open('w') as f:
                subprocess.run([sys.executable,'-u','-m','scripts.evaluate_remocap',
                    '--data',str(root/'data/derived/remocap_body_v2'),'--run',str(run),
                    '--output',str(output)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
        (record/f'{a.method}_seed{seed}_evaluation.json').write_text(done.read_text())
        print('EVALUATED',a.method,seed,flush=True)
    (record/f'{a.method}_EVALUATION_QUEUE_COMPLETE.json').write_text(json.dumps({'method':a.method,'seeds':[0,1,2]}))


if __name__=='__main__':main()
