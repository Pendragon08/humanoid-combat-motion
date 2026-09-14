"""Copy reproducible small records and compress logs; exclude motions/weights."""
import gzip
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1];destination=root/'records/step7/snapshot';destination.mkdir(parents=True,exist_ok=True)
names={'config.json','TRAINING_COMPLETE.json','metrics.json','manifest.json','source_manifest.json','evaluation.json','report.json','protocol.json','generic.json','contact.json','env.yaml','agent.yaml','history_seed0.json','history_seed1.json','history_seed2.json'}
entries=[]
for directory in sorted((root/'runs').glob('step7_*')):
    if not directory.is_dir():continue
    for path in sorted(directory.iterdir()):
        if path.name not in names or not path.is_file():continue
        if path.stat().st_size>8*1024**2:raise ValueError('Unexpectedly large record')
        data=path.read_bytes();out=destination/directory.name/path.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
        entries.append({'source':str(path.relative_to(root)),'saved':str(out.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest()})
for path in sorted((root/'records/step7').glob('*.log')):
    data=path.read_bytes();out=destination/'logs'/(path.name+'.gz');out.parent.mkdir(parents=True,exist_ok=True)
    # This is a consistent prefix snapshot if the task is still writing.
    out.write_bytes(gzip.compress(data,mtime=0))
    entries.append({'source':str(path.relative_to(root)),'saved':str(out.relative_to(root)),'uncompressed_sha256':hashlib.sha256(data).hexdigest(),'snapshot_bytes':len(data)})
summary=root/'records/step7/summary.json'
(destination/'INDEX.json').write_text(json.dumps({'files':entries,'summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest() if summary.exists() else None,'raw_data_and_weights_included':False},indent=2))
print('STEP7_RECORDS_SAVED',len(entries),flush=True)
