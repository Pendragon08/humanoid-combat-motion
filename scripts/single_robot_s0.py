"""Predeclared single-robot S0 source/interface pilot. No policy training.

Keep all source frames; compare prefix cropping and an initial-ground-offset
repair. The repair is an engineering control, not a reproduction of PBHC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from combat.twist_tracking import ROOT, scene, Controller

OUT = ROOT / 'runs/single_robot_s0_20260916'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def ground_depth(model, data):
    return max([0.] + [-float(data.contact[k].dist) for k in range(data.ncon)
        if model.geom_bodyid[data.contact[k].geom1] == 0 or model.geom_bodyid[data.contact[k].geom2] == 0])

def arrays(model, q, names, bodyids):
    data = mujoco.MjData(model)
    pos, quat = [], []
    for row in q:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        pos.append(data.xpos[bodyids].copy())
        quat.append(data.xquat[bodyids].copy())
    pos, quat = np.array(pos), np.array(quat)
    angular = np.zeros_like(pos)
    for b in range(len(bodyids)):
        rot = Rotation.from_quat(quat[:, b, [1,2,3,0]])
        # Spatial/world angular velocity, consistent with body_lin_vel_w.
        angular[:-1,b] = (rot[1:] * rot[:-1].inv()).as_rotvec() * 50
        angular[-1,b] = angular[-2,b]
    return dict(fps=np.array(50.), joint_names=np.array(names),
        body_names=np.array([model.body(i).name[3:] for i in bodyids]),
        joint_pos=q[:,7:], joint_vel=np.gradient(q[:,7:],.02,axis=0),
        body_pos_w=pos, body_quat_w=quat,
        body_lin_vel_w=np.gradient(pos,.02,axis=0), body_ang_vel_w=angular)

def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    spec=json.loads((ROOT/'env-twist-inference.json').read_text())
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
    assert digest=='86ed9bff',digest
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    model,_=scene(1)
    ctrl=Controller(model,0,None,'cpu')
    source_names=[e.attrib['name'] for e in ET.parse(ROOT/'assets/gmr_source_order.xml').findall('.//worldbody//joint')
                  if e.attrib.get('type','hinge')!='free']
    assert len(source_names)==29
    mapping=[source_names.index(n) for n in ctrl.names]
    bodyids=[i for i in range(model.nbody) if model.body(i).name.startswith('a0_')]
    rows=[]; chosen=[]; groups=set()
    for item in manifest['motions']:
        path=ROOT/item['path']
        assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:
            q=np.array(z['qpos'],dtype=float);fps=float(z['fps'])
        assert q.shape[1]==36 and np.isfinite(q).all() and fps==50
        norms=np.linalg.norm(q[:,3:7],axis=1)
        assert np.min(norms)>.5 and np.max(norms)<1.5
        row={**item,'frames':len(q),'seconds':(len(q)-1)/fps,
             'quaternion_norm_max_error':float(abs(norms-1).max())}
        rows.append(row)
        if item['split']=='train' and item['source_group'] not in groups:
            groups.add(item['source_group'])
            if len(chosen)<6:chosen.append((item,q))
    OUT.mkdir(parents=True)
    (OUT/'references').mkdir();(OUT/'logs').mkdir()
    save_json(OUT/'source_audit.json',{'motions':rows,'title_groups_not_independent_recording_guarantee':True})
    cases=[{'id':'standing','options':['--standing','--seconds','5']}]
    source_records=[]
    for item,raw in chosen:
        q=np.c_[raw[:,:7],raw[:,7:][:,mapping]]
        q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
        q[:,:2]-=q[0,:2].copy()
        data=mujoco.MjData(model);data.qpos[:]=q[0];mujoco.mj_forward(model,data)
        penetration=ground_depth(model,data)
        offset=penetration+.002 if penetration>.001 else 0.
        fixed=q.copy();fixed[:,2]+=offset
        variants={'full':q,'prefix150':q[:150].copy(),'initial_ground_offset':fixed}
        for name,value in variants.items():
            output=OUT/'references'/f'{item["id"]}_{name}.npz'
            np.savez_compressed(output,**arrays(model,value,ctrl.names,bodyids))
            cases.append({'id':f'{item["id"]}_{name}','source_id':item['id'],
                          'arm':name,'options':['--reference',str(output)]})
        limits=model.jnt_range[ctrl.jids]
        source_records.append({'id':item['id'],'source_group':item['source_group'],
            'split':item['split'],'raw_sha256':item['sha256'],'full_frames':len(q),
            'full_seconds':(len(q)-1)/50,'prefix_seconds':(min(150,len(q))-1)/50,
            'initial_ground_penetration_m':penetration,'constant_height_offset_m':offset,
            'joint_limit_violation_frame_fraction':float(np.any((q[:,7:]<limits[:,0])|(q[:,7:]>limits[:,1]),axis=1).mean()),
            'dropped_joints':[n for n in source_names if n not in ctrl.names],
            'raw_peak_joint_speed_rad_s':float(abs(np.gradient(q[:,7:],.02,axis=0)).max())})
        # Equal duration, joints, rotations, derivatives: offset changes height only.
        assert np.array_equal(fixed[:,3:],q[:,3:])
        assert np.allclose(np.diff(fixed[:,:3],axis=0),np.diff(q[:,:3],axis=0))
    protocol={'stage':'S0 engineering acceptance; not S1-S4 or official PBHC reproduction',
        'selection':'First training motion of first six title groups in frozen manifest order; no score-based choice',
        'arms':['full','prefix150','initial_ground_offset'],
        'ground_offset':'If initial ground penetration >1mm, raise every frame by penetration+2mm; no other motion change',
        'labels':'Whole martial-arts routines; no verified jab/hook/kick labels or phase/contact truth',
        'reference_conversion':'Original 50Hz retained, wxyz quaternion normalized, initial xy origin removed; official TWIST 25DOF FK, finite-difference velocities, no speed smoothing',
        'evaluation':'Unchanged evaluate_twist.py frozen policy, nominal deterministic physics; full-sequence metrics including post-fall errors',
        'strict_definition':'Every frame joint MAE<0.3rad AND root error<0.15m AND tracked body error<0.15m AND pelvis height>=0.35m',
        'cropping_caveat':'prefix vs full has unequal duration and is not an improvement claim',
        'source_partition':'Training-only pilot; val/test not used for tuning or claimed independent generalization',
        'progression_gate':'Standing witness passes AND at least 5/6 unmodified full motions strict-pass before S4; semantic source audit independently required',
        'training_steps':0,'seed':123,'env_spec_sha256_8':digest,'sources':source_records,
        'code_sha256':{p:sha(ROOT/p) for p in ['scripts/single_robot_s0.py','scripts/evaluate_twist.py','combat/twist_tracking.py']},
        'cases':cases,'limitations':['Policy pretraining overlap unknown','25DOF model omits four wrist joints','Not real robot','A failed frozen policy does not prove a motion is physically impossible']}
    save_json(OUT/'protocol.json',protocol)
    print('S0_PREPARED',len(rows),len(cases),flush=True)

def run(shard):
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['code_sha256'].items():assert sha(ROOT/path)==digest
    states=[]
    for i,case in enumerate(protocol['cases']):
        if i%2!=shard:continue
        output=OUT/'results'/case['id']
        if output.exists():raise FileExistsError(output)
        state={**case,'state':'running','started_at':time.time(),'device':f'cuda:{shard}'}
        states.append(state);save_json(OUT/f'queue_{shard}.json',states)
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_twist.py'),*case['options'],
             '--device',f'cuda:{shard}','--output',str(output)]
        state['command']=cmd
        with (OUT/'logs'/f'{case["id"]}.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        state['elapsed_s']=time.time()-state['started_at'];state['returncode']=result.returncode
        state['state']='completed' if result.returncode==0 and (output/'metrics.json').is_file() else 'failed'
        save_json(OUT/f'queue_{shard}.json',states)
        print('S0_CASE',case['id'],state['state'],round(state['elapsed_s'],2),flush=True)
        if state['state']=='failed':raise RuntimeError(case['id'])
    save_json(OUT/f'COMPLETE_{shard}.json',{'cases':len(states),'finished_at':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--shard',type=int,choices=[0,1],default=0)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    else:run(a.shard)
