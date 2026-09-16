"""Predeclared two deployment configurations on all four complete seed motions."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['official','benchmark','cross20k50','cross25k100'],required=True);a=p.parse_args()
    ckpt,hz={'official':('20k',100),'benchmark':('25k',50),'cross20k50':('20k',50),'cross25k100':('25k',100)}[a.variant]
    out=ROOT/'runs/twist2_upgrade_20260916'
    inventory=json.loads((ROOT/'runs/pbhc_seed_panel_20260916/inventory.json').read_text())
    names=['Hooks_punch','Horse-stance_punch','Roundhouse_kick','Side_kick']
    cases=[x for x in inventory['motions'] if x['name'] in names]
    assert len(cases)==4
    protocol=out/f'protocol_{a.variant}.json'
    if protocol.exists():raise FileExistsError(protocol)
    protocol.write_text(json.dumps({'variant':a.variant,'checkpoint':ckpt,'hz':hz,'selection':'All4 previously declared combat development examples, no outcome selection',
        'followup_design':'Cross configurations added after first2 results to distinguish checkpoint from frequency; diagnostic, not confirmatory selection' if a.variant.startswith('cross') else 'Original2 configurations','cases':cases},indent=2))
    rows=[]
    for case in cases:
        ref=Path(case['target_reference']);assert hashlib.sha256(ref.read_bytes()).hexdigest()==case['target_sha256']
        target=out/f"{case['name']}_{ckpt}_{hz}"
        command=[sys.executable,str(ROOT/'scripts/evaluate_twist2_official.py'),'--reference',str(ref),'--checkpoint',ckpt,'--hz',str(hz),'--output',str(target)]
        begin=time.time()
        with (ROOT/'records/tracker_upgrade_20260916'/f"{case['name']}_{ckpt}_{hz}.log").open('w') as f:
            status=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        rows.append({'name':case['name'],'exit':status.returncode,'wall_s':time.time()-begin,'output':str(target),'command':command})
        (out/f'queue_{a.variant}.json').write_text(json.dumps(rows,indent=2))
        if status.returncode:raise RuntimeError(case['name'])
    (out/f'COMPLETE_{a.variant}.json').write_text(json.dumps({'completed':len(rows)}))
    print('TWIST2_PANEL_COMPLETE',a.variant,flush=True)
if __name__=='__main__':main()
