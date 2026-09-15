"""Predeclared diagnostic panel: 8 old single references + all 48 existing pairs."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
from combat.twist_tracking import ROOT

def main():
    cases=[('dual_standing',['--dual-standing','--seconds','5','--video'])]
    for group,indices in [('kungfu_train20',[0,8,10,14]),('kungfu_test4',[0,1,2,3])]:
        for i in indices:
            cases.append((f'{group}_{i:03}', ['--reference',f'runs/{group}_tracking_refs/sample_{i:03}_actor0.npz']))
    for arm in ('generic','adapted'):
        for i in range(24):
            opts=[]
            for actor in range(2):opts+=['--reference',f'runs/timotion_{arm}_24_seed2026_tracking_refs/sample_{i:03}_actor{actor}.npz']
            if i==0:opts+=['--video']
            cases.append((f'dual_{arm}_{i:03}',opts))
    root=ROOT/'runs/twist_suite_v2';root.mkdir(exist_ok=True)
    rec=ROOT/'records/twist';rec.mkdir(exist_ok=True)
    (rec/'suite_protocol.json').write_text(json.dumps({'cases':cases,'seed':123,'policy_frozen':True,
        'same_scene_dual':True,'selection':'All existing 24 generic + 24 adapted pairs; old diagnostic 4 train + 4 development clips',
        'no_reference_repairs':True,'no_hyperparameter_search':True},indent=2))
    for name,opts in cases:
        out=root/name
        if (out/'metrics.json').exists():
            saved=json.loads((out/'metrics.json').read_text())
            for path,digest in saved['code_sha256'].items():
                if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError(f'{name}: code changed since existing result; use a new output directory')
            continue
        with (rec/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/evaluate_twist.py'),*opts,'--device','cuda:0','--output',str(out)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see log')
        row=json.loads((out/'metrics.json').read_text())
        print('TWIST_CASE_COMPLETE',name,'no_fall',row['no_fall_complete'],'strict',row['strict_complete'],flush=True)
    print('TWIST_SUITE_COMPLETE',len(cases),flush=True)

if __name__=='__main__':main()
