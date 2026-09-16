"""Compatibility panel: fixed TWIST on all seven official PBHC references."""
import json
import subprocess
import sys
import time
from pathlib import Path
from combat.twist_tracking import ROOT

panel=ROOT/'runs/pbhc_seed_panel_20260916'
out=ROOT/'runs/pbhc_seed_twist_20260916'
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
inventory=json.loads((panel/'inventory.json').read_text())
protocol={'scope':'Cross-controller compatibility; not PBHC reproduction or synthetic-data utility',
          'selection':'All seven official bundled motions','seed':123,'policy':'Unchanged official pretrained TWIST',
          'reference_conversion':'23DOF to25DOF by name with neutral wrist rolls,30Hz to50Hz,full duration',
          'all_sources_development_only':True,'training_steps':0,'cases':[x['name'] for x in inventory['motions']]}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
states=[]
for item in inventory['motions']:
    start=time.time()
    cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_twist.py'),'--reference',item['target_reference'],
         '--device','cuda:1','--output',str(out/item['name'])]
    with (out/(item['name']+'.log')).open('w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    states.append({'name':item['name'],'exit':p.returncode,'seconds':time.time()-start,'command':cmd})
    (out/'queue.json').write_text(json.dumps(states,indent=2))
    if p.returncode:raise RuntimeError(item['name'])
    print('PBHC_TWIST_CASE',item['name'],'complete',flush=True)
(out/'COMPLETE.json').write_text(json.dumps({'completed':len(states)}))
