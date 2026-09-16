"""Resolve from saved wheels, fetching only missing requirements from official PyPI."""
from pathlib import Path
import os
import re
import json
import hashlib
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    py=str(ROOT/'.venv-engineai-official/bin/python');rec=ROOT/'records/engineai_official_v1'
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['TMPDIR']=str(ROOT/'cache/engineai-official/tmp')
    wheels=ROOT/'cache/engineai-official/wheels';src=ROOT/'third_party/IsaacLab-engineai-official/source'
    cmd=[py,'-m','pip','install','--no-index','--no-build-isolation','--find-links',str(wheels),'-c',str(ROOT/'cache/engineai-official/constraints.txt')]
    for name in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']:cmd+=['-e',str(src/name)]
    cmd+=['rsl-rl-lib==5.0.1','onnxscript>=0.5','tensorboard','numpy==1.26.4','pin==2.7.0','hpp-fcl==2.4.4','wheel==0.45.1']
    attempt=int(time.time())
    for i in range(30):
        p=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (rec/f'cached_resolve_{attempt}_{i:02d}.log').write_text(p.stdout);print(p.stdout[-1800:],flush=True)
        if p.returncode==0:
            spec=json.loads((ROOT/'configs/engineai_official_env.json').read_text())
            sha=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            (rec/'INSTALL_COMPLETE.json').write_text(json.dumps({'spec_sha256':sha,'free_gib_after':shutil.disk_usage(ROOT).free/2**30,'status':'installed; physics witness still required'},indent=2));return
        missing=re.findall(r'No matching distribution found for ([^\n]+)',p.stdout)
        if not missing:raise RuntimeError('Resolution failed without missing requirement; inspect cached log')
        requirement=missing[-1].split(' (from')[0].strip()
        if len(requirement)>200 or requirement.startswith('-'):raise ValueError(requirement)
        print('FETCH ONLY',requirement,flush=True)
        subprocess.run([py,'-m','pip','download','--no-cache-dir','--no-deps','--dest',str(wheels),'--index-url','https://pypi.org/simple',requirement],cwd=ROOT,env=env,check=True)
    raise RuntimeError('Missing dependency limit reached')

if __name__=='__main__':main()
