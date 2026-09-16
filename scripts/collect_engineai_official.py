"""Save native-version evidence and checkpoints' hashes, not third-party weights."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/engineai_official_v1'
RUN=ROOT/'runs/engineai_official_v1'

def main():
    REC.mkdir(parents=True,exist_ok=True)
    for p in RUN.glob('*/*.json'):shutil.copy2(p,REC/(p.parent.name+'_'+p.name))
    for p in RUN.glob('*_FAILED.txt'):shutil.copy2(p,REC/p.name)
    source=ROOT/'third_party/IsaacLab-engineai-official/SOURCE.json'
    if source.exists():shutil.copy2(source,REC/'isaaclab_source.json')
    py=ROOT/'.venv-engineai-official/bin/python'
    if py.exists():
        freeze=subprocess.run([str(py),'-m','pip','freeze'],capture_output=True,text=True)
        (REC/'pip_freeze.txt').write_text(freeze.stdout)
    storage={name:{'total_gib':shutil.disk_usage(path).total/2**30,'free_gib':shutil.disk_usage(path).free/2**30} for name,path in [('system','/'),('data','/data')]}
    (REC/'storage_after.json').write_text(json.dumps(storage,indent=2))
    manifest=[]
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.pt','.npz']:
            manifest.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    (REC/'artifacts.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(ROOT/'cache/engineai_official_records.tar.gz','w:gz') as tar:
        for p in REC.iterdir():
            if p.is_file():tar.add(p,arcname=str(p.relative_to(ROOT)))
    print('OFFICIAL_EVIDENCE_SAVED',len(manifest))

if __name__=='__main__':main()
