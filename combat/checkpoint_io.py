"""Verified transport of an official model's arrays, without optimizer state."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def validated_arrays(path):
    path=Path(path)
    manifest=json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if sha256(path)!=manifest['export_sha256']:raise ValueError('Transport archive hash mismatch')
    arrays={}
    with np.load(path,allow_pickle=False) as data:
        expected={r['key'] for r in manifest['parameters']}
        if set(data.files)!=expected:raise ValueError('Missing or extra parameter arrays')
        for record in manifest['parameters']:
            a=data[record['key']]
            if a.dtype!=np.dtype('<f4') or list(a.shape)!=record['shape'] or hashlib.sha256(a.tobytes()).hexdigest()!=record['sha256']:
                raise ValueError('Parameter integrity failure: '+record['key'])
            arrays[record['key']]=a
    return arrays,manifest['source_sha256']


def load_model_state(path):
    import torch
    path=Path(path)
    if path.suffix=='.npz':
        arrays,original_sha=validated_arrays(path)
        return {k:torch.from_numpy(v) for k,v in arrays.items()},original_sha
    if path.suffix in ('.partial','.assembling'):raise ValueError('Incomplete checkpoint')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    return {k.removeprefix('model.'):v for k,v in checkpoint['state_dict'].items()},sha256(path)
