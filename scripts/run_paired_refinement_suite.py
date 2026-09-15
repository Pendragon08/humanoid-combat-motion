"""Fixed 48-case paired refinement with matched export control and frozen policy."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from combat.twist_tracking import ROOT

def case(arm,i):
    name=f'{arm}_{i:03}';base=ROOT/'runs/paired_refinement_v2'/name
    logdir=ROOT/'records/paired_refinement_v2';logdir.mkdir(parents=True,exist_ok=True)
    args=[]
    for actor in range(2):args+=['--reference',str(ROOT/f'runs/timotion_{arm}_24_seed2026_tracking_refs/sample_{i:03}_actor{actor}.npz')]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def run(script,opts,log):
        with log.open('w') as f:
            proc=subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*opts],stdout=f,stderr=subprocess.STDOUT,env=env)
        if proc.returncode:raise RuntimeError(f'{name}: see {log}')
    if not (base/'refinement/refinement.json').exists():
        run('refine_twist_pair.py',args+['--output',str(base/'refinement')],logdir/f'{name}_refine.log')
    for variant,ref in [('original_export',base/'refinement/original_export'),('corrected',base/'refinement')]:
        if not (base/variant/'metrics.json').exists():
            run('evaluate_twist.py',['--reference',str(ref/'actor0.npz'),'--reference',str(ref/'actor1.npz'),
                '--output',str(base/variant),'--device','cuda:0'],logdir/f'{name}_{variant}.log')
    print('PAIRED_REFINEMENT_CASE_COMPLETE',name,flush=True)
    return name

def main():
    cases=[(arm,i) for arm in ('generic','adapted') for i in range(24)]
    (ROOT/'records/paired_refinement_v2').mkdir(parents=True,exist_ok=True)
    (ROOT/'records/paired_refinement_v2/protocol.json').write_text(json.dumps({'cases':cases,'workers':3,
        'arms':['original_export','corrected'],'selection':'all48existing pairs; implementation pilot adapted000; development analysis',
        'reference_timing_unchanged':True,'controller_frozen':True},indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for _ in pool.map(lambda x:case(*x),cases):pass
    print('PAIRED_REFINEMENT_SUITE_COMPLETE',len(cases),flush=True)

if __name__=='__main__':main()
