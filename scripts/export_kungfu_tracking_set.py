"""Select source-disjoint public video-derived G1 references before evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','test'],required=True)
p.add_argument('--limit',type=int,default=20);p.add_argument('--frames',type=int,default=150)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
if a.output.exists():raise FileExistsError(a.output)
manifest=json.loads((root/'data/manifest.json').read_text())
selected=sorted([r for r in manifest['motions'] if r['split']==a.split],key=lambda r:int(r['id']))[:a.limit]
names=[j.get('name') for j in ET.parse(root/'assets/gmr_source_order.xml').findall('.//worldbody//joint') if j.get('type','hinge')!='free']
if len(names)!=29:raise ValueError('Unexpected joint order')
a.output.mkdir(parents=True);records=[]
for i,entry in enumerate(selected):
    path=root/entry['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        q=z['qpos'][:a.frames].copy();fps=float(z['fps'])
    if len(q)<20 or q.shape[1]!=36 or not np.isfinite(q).all():raise ValueError('Invalid source')
    norm=np.linalg.norm(q[:,3:7],axis=-1)
    if norm.min()<.5 or norm.max()>1.5:raise ValueError('Invalid root quaternion')
    q[:,3:7]/=norm[:,None]
    dest=a.output/f'sample_{i:03d}.npz'
    np.savez_compressed(dest,qpos=q,fps=fps,joint_names=np.array(names))
    records.append({'sample':dest.stem,'source':entry,'frames':len(q),'start':0,
                    'quaternion_normalized':True,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
(a.output/'source_manifest.json').write_text(json.dumps({'records':records,'selection':'sorted numeric source ID, prefix crop, no outcome selection','split':a.split},ensure_ascii=False,indent=2),encoding='utf-8')
print('KUNGFU_REFERENCE_EXPORT',a.split,len(records),flush=True)
