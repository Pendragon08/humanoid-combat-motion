"""Explicit position-only G1 IK baseline using pinned GMR assets and Mink.

Does NOT claim to reproduce GMR's orientation-based SMPL-X retargeting. TIMotion
rotation-channel conventions are not assumed. Both actors retain a common world
frame. Joint-position limits alone do not certify dynamic executability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation

TARGETS={'pelvis':(0,20.),'torso_link':(9,2.),
 'left_knee_link':(4,2.),'right_knee_link':(5,2.),
 'left_ankle_roll_link':(7,8.),'right_ankle_roll_link':(8,8.),
 'left_elbow_link':(18,2.),'right_elbow_link':(19,2.),
 'left_wrist_yaw_link':(20,5.),'right_wrist_yaw_link':(21,5.)}


def anatomical_rotation(points):
    left=points[1]-points[2];up=points[9]-points[0]
    up=up/max(np.linalg.norm(up),1e-8)
    forward=np.cross(left,up)
    if np.linalg.norm(forward)<1e-6:raise ValueError('Degenerate anatomical frame')
    forward/=np.linalg.norm(forward);left=np.cross(up,forward)
    return Rotation.from_matrix(np.stack([forward,left,up],axis=1)).as_quat(scalar_first=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scale',type=float,default=.75)
    p.add_argument('--iterations',type=int,default=8)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not .3<a.scale<1.5 or a.iterations<1:raise ValueError('Invalid IK setting')
    root=Path(__file__).resolve().parents[1]
    xml=root/'third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml'
    model=mujoco.MjModel.from_xml_path(str(xml))
    names=[model.joint(i).name for i in range(1,model.njnt)]
    tasks=[mink.FrameTask(name,'body',position_cost=weight,
             orientation_cost=1. if name in ('pelvis','torso_link') else 0.,lm_damping=1.)
             for name,(idx,weight) in TARGETS.items()]
    limits=[mink.ConfigurationLimit(model)]
    a.output.mkdir(parents=True);records=[]
    for path in sorted(a.input.glob('sample_*.npz')):
        z=np.load(path,allow_pickle=False);x=z['positions'].astype(np.float64);fps=float(z['fps'])
        if x.ndim!=4 or x.shape[1:]!=(2,22,3) or not np.isfinite(x).all():raise ValueError('Invalid paired positions')
        # Fixed right-handed rotation: human (x,y,z) -> world (x,-z,y).
        target=x[...,[0,2,1]].copy();target[...,1]*=-1;target*=a.scale
        # One constant floor offset shared by both actors, never per-frame snapping.
        floor=float(target[:,:, [7,8,10,11],2].min());target[...,2]-=floor
        qpos=np.empty((len(x),2,model.nq));errors=[]
        for actor in range(2):
            config=mink.Configuration(model)
            initial=config.data.qpos.copy();initial[:3]=target[0,actor,0]
            initial[3:7]=anatomical_rotation(target[0,actor]);config.update(initial)
            for t in range(len(x)):
                rotation=anatomical_rotation(target[t,actor])
                for task,(name,(joint,weight)) in zip(tasks,TARGETS.items()):
                    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rotation),target[t,actor,joint]))
                for _ in range(a.iterations):
                    velocity=mink.solve_ik(config,tasks,1/fps,'daqp',damping=.05,limits=limits)
                    config.integrate_inplace(velocity,1/fps)
                qpos[t,actor]=config.data.qpos
                errors.extend(float(np.linalg.norm(config.get_transform_frame_to_world(name,'body').translation()-target[t,actor,idx])) for name,(idx,_) in TARGETS.items())
        if not np.isfinite(qpos).all():raise FloatingPointError('Nonfinite IK')
        velocity=np.diff(qpos[...,7:],axis=0)*fps
        lower=model.jnt_range[1:,0];upper=model.jnt_range[1:,1]
        violation=np.maximum(lower-qpos[...,7:],0)+np.maximum(qpos[...,7:]-upper,0)
        dest=a.output/path.name
        np.savez_compressed(dest,qpos=qpos,fps=fps,joint_names=np.asarray(names),target_positions=target)
        row={'sample':path.stem,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'mean_body_target_error_m':float(np.mean(errors)),'max_joint_limit_violation_rad':float(violation.max()),
             'max_joint_speed_rad_s':float(np.abs(velocity).max()),'common_floor_offset_m':floor,
             'frames':len(x),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        records.append(row);print(json.dumps(row),flush=True)
    if not records:raise ValueError('No paired samples')
    (a.output/'manifest.json').write_text(json.dumps({'method':'position-only Mink IK with GMR G1 assets; not default GMR',
        'scale':a.scale,'scale_source':'fixed research setting, not measured human height',
        'iterations_per_frame':a.iterations,'gmr_commit':'bb1bbe40774794fceb2a7c579a3464a28e68c844',
        'rotation_channels_used':False,'root_orientation':'anatomical hip/spine frame',
        'shared_world_preserved':True,'learned_tracking':False,'dynamic_feasibility_verified':False,
        'limitations':['twist ambiguity','no pair collision constraints','no ground force balance constraint',
                       'iterative IK time is solver time, not a joint-velocity certificate'],
        'samples':records},indent=2),encoding='utf-8')


if __name__=='__main__':main()
