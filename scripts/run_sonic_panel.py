import json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'runs/sonic_probe_20260916';protocol=out/'protocol.json'
    if protocol.exists():raise FileExistsError(protocol)
    inventory=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())
    names=['Hooks_punch','Horse-stance_punch','Roundhouse_kick','Side_kick'];cases=[x for x in inventory['motions'] if x['name'] in names]
    assert len(cases)==4
    protocol.write_text(json.dumps({'selection':'All4 existing complete combat development examples, unchanged','training':False,'backend':'HumanTracker SONIC observation/action adaptation','cases':cases},indent=2))
    states=[]
    for case in cases:
        ref=Path(case['target_reference']);assert hashlib.sha256(ref.read_bytes()).hexdigest()==case['target_sha256']
        dest=out/case['name'];cmd=[sys.executable,str(ROOT/'scripts/evaluate_sonic_probe.py'),'--reference',str(ref),'--output',str(dest)]
        begin=time.time()
        with (ROOT/'records/tracker_upgrade_20260916'/('sonic_'+case['name']+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        states.append({'name':case['name'],'exit':r.returncode,'wall_s':time.time()-begin,'command':cmd})
        (out/'queue.json').write_text(json.dumps(states,indent=2))
        if r.returncode:raise RuntimeError(case['name'])
    (out/'COMPLETE.json').write_text(json.dumps({'completed':len(states)}));print('SONIC_PANEL_COMPLETE')
if __name__=='__main__':main()
