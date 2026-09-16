"""Export small auditable records separately from source-derived binary motions."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from combat.twist_tracking import ROOT

def main():
    run_names=['pbhc_official_20260916','pbhc_seed_panel_20260916',
               'pbhc_seed_twist_20260916','pbhc_refinement_pilot_20260916']
    destination=ROOT/'records/pbhc_20260916'
    inventory=[]
    with zipfile.ZipFile(destination/'audit_records.zip','w',zipfile.ZIP_DEFLATED) as audit, \
         zipfile.ZipFile(destination/'full_evidence.zip','w',zipfile.ZIP_DEFLATED) as full:
        for name in run_names:
            for path in sorted((ROOT/'runs'/name).rglob('*')):
                if not path.is_file() or path.suffix not in ['.json','.csv','.npz','.yaml']:continue
                rel=str(path.relative_to(ROOT/'runs'))
                inventory.append({'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size,
                                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                full.write(path,rel)
                if path.suffix in ['.json','.csv','.yaml']:audit.write(path,rel)
        audit.writestr('evidence_manifest.json',json.dumps(inventory,indent=2))
        full.writestr('evidence_manifest.json',json.dumps(inventory,indent=2))
    (destination/'environment_packages.json').write_text(subprocess.check_output(
        [str(ROOT/'.venv-pbhc-inference/bin/python'),'-m','pip','list','--format=json'],text=True))
    print(json.dumps({'files':len(inventory),'archives':[
        {'path':str(destination/n),'bytes':(destination/n).stat().st_size}
        for n in ['audit_records.zip','full_evidence.zip']]}))

if __name__=='__main__':main()
