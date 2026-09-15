"""Refine paired existing references and export named TWIST-compatible NPZ."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from combat.twist_tracking import ROOT,Controller,reference,scene
from combat.paired_refinement import refine,difference

def export(model,qs,fps,out,actor,ctrl):
    data=mujoco.MjData(model);bids=[i for i in range(1,model.nbody) if model.body(i).name.startswith(ctrl.prefix)]
    pos=[];quat=[]
    for q in qs:
        data.qpos[:]=q;mujoco.mj_forward(model,data);pos.append(data.xpos[bids].copy());quat.append(data.xquat[bids].copy())
    pos=np.asarray(pos);quat=np.asarray(quat)
    # Keep quaternion hemisphere continuous before differentiating rotations.
    for t in range(1,len(quat)):
        flip=np.sum(quat[t]*quat[t-1],axis=-1)<0;quat[t,flip]*=-1
    lin=np.gradient(pos,1/fps,axis=0);ang=np.zeros_like(lin)
    for b in range(len(bids)):
        r=Rotation.from_quat(quat[:,b,[1,2,3,0]])
        a=(r[1:]*r[:-1].inv()).as_rotvec()*fps
        ang[:-1,b]=a;ang[-1,b]=a[-1]
    joint=qs[:,ctrl.qids]
    np.savez_compressed(out/f'actor{actor}.npz',fps=50,joint_names=np.array(ctrl.names),
        body_names=np.array([model.body(i).name[len(ctrl.prefix):] for i in bids]),
        joint_pos=joint,joint_vel=np.gradient(joint,1/fps,axis=0),body_pos_w=pos,body_quat_w=quat,
        body_lin_vel_w=lin,body_ang_vel_w=ang)

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int)
    p.add_argument('--config',type=Path,default=ROOT/'configs/paired_refinement_v1.json');a=p.parse_args()
    assert len(a.reference)==2
    if a.output.exists():raise FileExistsError(a.output)
    model,_=scene(2);controllers=[Controller(model,i,None,'cpu') for i in range(2)]
    refs=[reference(path,c) for path,c in zip(a.reference,controllers)]
    assert len(refs[0]['qpos'])==len(refs[1]['qpos'])
    raw=np.concatenate([r['qpos'] for r in refs],axis=1)
    if a.max_frames:raw=raw[:a.max_frames]
    params=json.loads(a.config.read_text(encoding='utf-8'))
    corrected,info=refine(model,raw,50,params)
    a.output.mkdir(parents=True)
    (a.output/'original_export').mkdir()
    for i,c in enumerate(controllers):
        export(model,corrected,50,a.output,i,c)
        export(model,raw,50,a.output/'original_export',i,c)
    correction=np.array([difference(model,x,y) for x,y in zip(raw,corrected)])
    np.savez_compressed(a.output/'geometry_audit.npz',original_qpos=raw,corrected_qpos=corrected,correction=correction,
        **{k:v for k,v in info.items() if k!='solver_status'})
    pair_before=raw[:,:3]-raw[:,32:35];pair_after=corrected[:,:3]-corrected[:,32:35]
    result={'frames':len(raw),'fps':50,'duration_changed':False,'config':params,
        'inputs':[{'path':str(x),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in a.reference],
        'code_sha256':hashlib.sha256((ROOT/'combat/paired_refinement.py').read_bytes()).hexdigest(),
        'penetration_kinds':['ground','self','interrobot'],
        'max_penetration_before_m':info['penetration_before'].max(0).tolist(),
        'max_penetration_after_m':info['penetration_after'].max(0).tolist(),
        'frames_over_1cm_before':(info['penetration_before'].max(1)>.01).sum().item(),
        'frames_over_1cm_after':(info['penetration_after'].max(1)>.01).sum().item(),
        'mean_joint_change_rad':float(np.abs(correction[:,np.r_[6:31,37:62]]).mean()),
        'max_joint_change_rad':float(np.abs(correction[:,np.r_[6:31,37:62]]).max()),
        'max_pair_root_relation_change_m':float(np.linalg.norm(pair_after-pair_before,axis=1).max()),
        'support_frame_fraction':float(info['support_mask'].mean()),
        'max_joint_speed_rad_s':float(np.abs(np.diff(corrected[:,np.r_[7:32,39:64]],axis=0)*50).max()),
        'dynamic_feasibility_verified':False,'limitations':['Soft geometry constraints can remain violated.','Support is a velocity/height heuristic, not labeled contact.','Joint change is bounded; no semantic correctness certificate.']}
    (a.output/'refinement.json').write_text(json.dumps(result,indent=2))
    print('PAIRED_REFINEMENT_COMPLETE '+json.dumps(result),flush=True)

if __name__=='__main__':main()
