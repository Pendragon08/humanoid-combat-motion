"""Freeze the completed pilot as a local downloadable bundle, excluding environments/caches."""
from pathlib import Path
import hashlib,json,zipfile,importlib.metadata
from combat.robot import ROOT

def main():
    for seed in [0,1]:
        done=ROOT/f'runs/pipeline_seed{seed}.complete.json'
        assert json.loads(done.read_text())['status']=='PASS'
    assert (ROOT/'deliverables/humanoid_combat_motion_pilot_report.html').exists()
    packages=['torch','numpy','scipy','matplotlib','Pillow','imageio','imageio-ffmpeg','mujoco','osqp','glfw','PyOpenGL','etils']
    (ROOT/'audit/installed_versions.json').write_text(json.dumps({p:importlib.metadata.version(p) for p in packages},indent=2))
    selected=[]
    for folder in ['combat','scripts','tests','data','assets','runs','audit']:
        for path in (ROOT/folder).rglob('*'):
            if not path.is_file() or path.is_symlink():continue
            rel=path.relative_to(ROOT)
            if '__pycache__' in rel.parts or 'cache' in rel.parts:continue
            if 'tdmpc2_archive' in rel.parts and path.name!='cleanup_verified.json':continue
            if path.name.endswith('.partial.json'):continue
            selected.append(path)
    selected += [ROOT/'README.md',ROOT/'env-spec.json',ROOT/'.gitignore']
    files={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(selected))}
    manifest=ROOT/'deliverables/code_bundle_manifest.json'
    manifest.write_text(json.dumps({'files':files,'file_count':len(files),'scope':'Pilot code, raw reference data, pinned robot assets, both training checkpoints, logs, evaluations and audit. No environment, cache or retired-project backup. No learned robot imitation policy yet.'},indent=2))
    target=ROOT/'deliverables/humanoid_combat_motion_pilot_code_and_results.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(set(selected)):z.write(p,'humanoid-combat-motion/'+str(p.relative_to(ROOT)))
        z.write(manifest,'humanoid-combat-motion/code_bundle_manifest.json')
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    result={'name':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'file_count':len(files),'zip_crc_check':'PASS'}
    (ROOT/'deliverables/code_bundle_verification.json').write_text(json.dumps(result,indent=2))
    print('PACKAGE_COMPLETE',json.dumps(result))

if __name__=='__main__':main()
