"""Validated storage for separate reference clips; never interpolate boundaries."""
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')


def build_bundle(entries, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not entries:raise ValueError('Empty bundle')
    ids=[e['clip_id'] for e in entries]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate clip IDs')
    if any(e['split']!='train' for e in entries):raise ValueError('Training bundle requires train-only sources')
    arrays={k:[] for k in FIELDS};starts=[];ends=[];records=[];offset=0;schema=None
    for entry in entries:
        path=Path(entry['path']).resolve(strict=True)
        with np.load(path,allow_pickle=False) as z:
            current=(float(np.asarray(z['fps']).item()),tuple(z['joint_names']),tuple(z['body_names']))
            if schema is None:schema=current
            if current!=schema:raise ValueError('FPS or robot name/order mismatch')
            n=len(z['joint_pos'])
            if n<4:raise ValueError('Clip too short')
            for k in FIELDS:
                x=z[k]
                if len(x)!=n or not np.isfinite(x).all():raise ValueError('Invalid reference arrays')
                arrays[k].append(x.copy())
            starts.append(offset);offset+=n;ends.append(offset)
        records.append(dict(entry,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=n))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**{k:np.concatenate(v) for k,v in arrays.items()},fps=schema[0],
        joint_names=np.array(schema[1]),body_names=np.array(schema[2]),clip_starts=np.array(starts),
        clip_ends=np.array(ends),clip_ids=np.array(ids),source_ids=np.array([e['source_id'] for e in entries]))
    record={'clips':records,'frames':offset,'fps':schema[0],'boundary_interpolation':False,
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record
