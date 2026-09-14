"""Two separated copies of a learned training reference as an interface control.

This deliberately synthetic non-contact control is not paired human combat data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
with np.load(a.reference,allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
a.output.mkdir(parents=True);records=[]
for actor,shift in enumerate([-1.5,1.5]):
    arrays={k:v.copy() for k,v in source.items()};arrays['body_pos_w'][...,0]+=shift
    path=a.output/f'actor{actor}.npz';np.savez_compressed(path,**arrays)
    records.append({'actor':actor,'x_translation_m':shift,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(a.output/'manifest.json').write_text(json.dumps({'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
    'records':records,'purpose':'non-contact positive-control attempt for dual-policy interface',
    'combat':False,'independent_demonstrations':False},indent=2))
