"""Collect small evidence without model weights or third-party datasets."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/engineai_single_v1';REC=ROOT/'records/engineai_single_v1'
def main():
    mapping={'references_manifest.json':RUN/'references_50hz/manifest.json','export_manifest.json':RUN/'tracking_references_50hz/results.json','pd_results.json':RUN/'pd_diagnostic_50hz/results.json'}
    for name in ['official','raw_ik','joint_constrained']:
        mapping[name+'_eval.json']=RUN/f'{name}_eval/results.json'
        mapping[name+'_train_complete.json']=RUN/f'{name}_train/COMPLETE.json'
    mapping['official_pretrained_eval.json']=RUN/'official_pretrained_eval/results.json'
    mapping['official_pretrained_full_eval.json']=RUN/'official_pretrained_full_eval/results.json'
    for arm in ['raw_ik','joint_constrained']:
        mapping[f'warm_{arm}_eval.json']=RUN/f'warm_{arm}_eval/results.json'
        mapping[f'warm_{arm}_train_complete.json']=RUN/f'warm_{arm}_train/COMPLETE.json'
    for dst,src in mapping.items():
        if src.exists():shutil.copy2(src,REC/dst)
    for p in RUN.glob('*/protocol.json'):
        shutil.copy2(p,REC/(p.parent.name+'_protocol.json'))
    manifest=[]
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.npz','.pt','.mp4']:
            manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (REC/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(ROOT/'cache/engineai_evidence_latest.tar.gz','w:gz') as tar:
        for p in REC.iterdir():
            if p.is_file():tar.add(p,arcname=str(p.relative_to(ROOT)))
        for p in (RUN/'media').glob('*'):
            if p.is_file():tar.add(p,arcname='reports/media/engineai_single/'+p.name)
        for p in RUN.glob('*eval/rollout_env0.npz'):tar.add(p,arcname=str(p.relative_to(ROOT)))
    print('ENGINEAI_EVIDENCE_COLLECTED',len(manifest))
if __name__=='__main__':main()
