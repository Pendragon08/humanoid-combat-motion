import argparse
import json
from pathlib import Path
from combat.reference_bundle import build_bundle

p=argparse.ArgumentParser();p.add_argument('--entries',type=Path)
p.add_argument('--source-manifest',type=Path);p.add_argument('--reference-dir',type=Path)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.entries:
    entries=json.loads(a.entries.read_text())
else:
    if not a.source_manifest or not a.reference_dir:raise ValueError('Supply entries or source manifest plus reference directory')
    source=json.loads(a.source_manifest.read_text())
    entries=[{'path':str(a.reference_dir/(r['sample']+'_actor0.npz')),'clip_id':r['sample'],
              'source_id':r['source']['id'],'source_group':r['source']['source_group'],'split':r['source']['split'],
              'kind':r['source']['kind']} for r in source['records']]
print(json.dumps(build_bundle(entries,a.output),ensure_ascii=False),flush=True)
