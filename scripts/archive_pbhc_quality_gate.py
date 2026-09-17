"""Post-run provenance verification and private trace archive; not a pre-freeze."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'records/pbhc_quality_gate_20260917'
OUT=ROOT/'runs/pbhc_quality_gate_20260917'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert (REC/'COMPLETE.json').exists()
    target=REC/'postrun_provenance.json'
    assert not target.exists()
    manifest=json.loads((REC/'manifest.json').read_text())
    paths=[];checks=[]
    for row in manifest['records']:
        for key,digest in [('source','source_sha256'),('source_contact_path','source_contact_sha256')]:
            path=Path(row[key]);value=sha(path);assert value==row[digest]
            paths.append(path);checks.append(dict(path=str(path.relative_to(ROOT)),sha256=value,matched_record=digest))
        src=OUT/'evaluation'/f'{row["name"]}_quality_gated'/'metrics.json'
        dest=REC/'evaluation'/row['name'];dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest/'metrics.json')
    urdf=ROOT/'third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf'
    assert sha(urdf)==json.loads((ROOT/'records/pbhc_kinematic_audit_20260917/audit.json').read_text())['urdf_sha256']
    paths.append(urdf);checks.append(dict(path=str(urdf.relative_to(ROOT)),sha256=sha(urdf),matched_record='prior kinematic audit'))
    for rel in ['configs/pbhc_quality_gate_v1.json','runs/pbhc_seed_panel_20260916/inventory.json',
                'runs/pbhc_refinement_pilot_20260916/protocol.json',
                'scripts/audit_pbhc_reference_kinematics.py','scripts/analyze_pbhc_support_diagnostics.py',
                'scripts/audit_pbhc_temporal_quality.py','scripts/prepare_pbhc_quality_gate.py',
                'scripts/select_pbhc_quality_candidates.py','scripts/verify_pbhc_quality_gate.py']:
        path=ROOT/rel;paths.append(path)
        checks.append(dict(path=rel,sha256=sha(path),matched_record='post-run snapshot; not all pre-frozen'))
    assert sha(ROOT/'configs/pbhc_quality_gate_v1.json')==manifest['config_sha256']
    assert sha(ROOT/'scripts/prepare_pbhc_quality_gate.py')==manifest['script_sha256']
    target.write_text(json.dumps(dict(scope='Post-run checks, does not retroactively establish complete pre-execution freeze',
        checks=checks,limitation='Imported helpers, inventory and coarse source-window metadata were not all hash-enforced during original selection; snapshots preserve current bytes only'),indent=2))
    report=REC/'report.md'
    report.write_text(report.read_text().replace('最大可接受修正幅度','预定离散候选中的最大通过幅度'))
    archive=OUT/'quality_gate_artifacts.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for folder in ['candidate','selected','backtracking_selected','references','evaluation']:
            tar.add(OUT/folder,arcname=str((OUT/folder).relative_to(ROOT)))
        tar.add(REC,arcname=str(REC.relative_to(ROOT)))
        for path in paths:tar.add(path,arcname=str(path.relative_to(ROOT)))
    (REC/'archive.json').write_text(json.dumps(dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
        scope='Private source/reference and evaluation trace archive; no weights; do not put third-party data in Git'),indent=2))
    print('QUALITY_GATE_ARCHIVED',archive.stat().st_size,flush=True)

if __name__=='__main__':main()
