"""Archive private rollout/reference arrays locally; commit only compact evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_support_repair_20260917'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert (REC/'verification.json').exists()
    archive=ROOT/'runs/pbhc_support_pilot_artifacts_20260917.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    files=[]
    for stage in ['pbhc_support_diagnostic_20260917','pbhc_support_repair_20260917']:
        rec=ROOT/'records'/stage;manifest=json.loads((rec/'manifest.json').read_text())
        for job in manifest['jobs']:
            src=ROOT/job['output']/'metrics.json'
            dest=rec/'metrics'/Path(job['output']).name/'metrics.json';dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists();shutil.copy2(src,dest)
        for base in [ROOT/'runs'/stage,rec]:
            files.extend(p for p in base.rglob('*') if p.is_file() and p.suffix not in ['.lock','.tmp'])
    shutil.copy2(ROOT/'runs/pbhc_support_repair_20260917/references/manifest.json',REC/'conversion_manifest.json')
    files.append(REC/'conversion_manifest.json')
    inventory={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    receipt=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=inventory,
                 privacy='Raw reference/rollout arrays stay in private archive; no checkpoints or original dataset bundle included. Compact reports/metrics can be committed.')
    (REC/'archive_manifest.json').write_text(json.dumps(receipt,indent=2))
    print('SUPPORT_ARCHIVE',archive,archive.stat().st_size,receipt['sha256'])

if __name__=='__main__':main()
