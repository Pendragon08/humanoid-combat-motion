"""Copy small owned experiment records, excluding raw data and third-party weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    allowed={'metrics.json','evaluation.json','provenance.json','manifest.json','config.json',
             'TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json','evaluation.log','loss.jsonl'}
    copied=[]
    patterns=['timotion_*','g1_tracking_eval_*','dual_g1_*']
    for pattern in patterns:
        for run in sorted((root/'runs').glob(pattern)):
            if not run.is_dir():continue
            for path in sorted(run.iterdir()):
                if path.name not in allowed or not path.is_file():continue
                dest=out/run.name/path.name;dest.parent.mkdir(exist_ok=True)
                shutil.copy2(path,dest)
                copied.append({'origin':str(path.relative_to(root)),'file':str(dest.relative_to(out)),
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for group in ('interhuman_manifest','interhuman_validated_manifest'):
        for name in ('summary.json','missing.json','excluded.json'):
            path=root/'records/paired_v1'/group/name
            if path.exists():
                dest=out/group/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
    (out/'INDEX.json').write_text(json.dumps({'records':copied,'excludes':'raw datasets, checkpoints, credentials'},indent=2))
    print('RECORDS_SNAPSHOT',len(copied),out)


if __name__=='__main__':main()
