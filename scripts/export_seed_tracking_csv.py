"""Convert a verified train-source G1 qpos crop to the official tracker CSV ABI.

Input: GMR named joint order, root quaternion wxyz. Output: named tracker order,
root quaternion xyzw. This is a seed-controller integration run, not paired data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

NAMES=[f'{side}_{name}_joint' for side in ('left','right') for name in
       ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
NAMES += ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
NAMES += [f'{side}_{name}_joint' for side in ('left','right') for name in
          ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--start',type=int,default=0)
    p.add_argument('--frames',type=int,default=120);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'data/manifest.json').read_text(encoding='utf-8'))
    matches=[r for r in manifest['motions'] if r['id']==a.id]
    if len(matches)!=1 or matches[0]['split']!='train':raise ValueError('Use a unique training-source seed')
    entry=matches[0];source=root/entry['path']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Source checksum mismatch')
    z=np.load(source,allow_pickle=False);q=z['qpos'];fps=float(z['fps'])
    if a.start<0 or a.frames<20 or a.start+a.frames>len(q):raise ValueError('Invalid crop')
    names=[j.get('name') for j in ET.parse(root/'assets/gmr_source_order.xml').findall('.//worldbody//joint') if j.get('type','hinge')!='free']
    if len(names)!=29 or set(names)!=set(NAMES):raise ValueError('Unexpected source joint schema')
    q=q[a.start:a.start+a.frames].copy()
    if q.shape[1]!=36 or not np.isfinite(q).all():raise ValueError('Invalid G1 motion')
    norms=np.linalg.norm(q[:,3:7],axis=-1)
    # Public smoothed qpos files contain non-unit quaternions. Preserve the
    # represented orientation by normalization; reject degenerate/extreme norms.
    if norms.min()<.5 or norms.max()>1.5:raise ValueError('Invalid root quaternions')
    q[:,3:7]/=norms[:,None]
    data=np.concatenate([q[:,:3],q[:,[4,5,6,3]],q[:,7:][:,[names.index(n) for n in NAMES]]],axis=-1)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savetxt(a.output,data,delimiter=',',fmt='%.9g')
    record={'source':entry,'source_fps':fps,'start':a.start,'frames':a.frames,'joint_names':NAMES,
            'input_quaternion_norm_range':[float(norms.min()),float(norms.max())],
            'output_quaternion':'xyzw','input_quaternion':'wxyz','csv_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),
            'purpose':'seed learned-tracker integration; not paired or synthetic utility experiment'}
    a.output.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'csv':str(a.output),'fps':fps,'frames':a.frames}),flush=True)


if __name__=='__main__':main()
