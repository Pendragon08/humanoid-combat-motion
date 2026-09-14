"""Export explicit G1 kinematics for a future tracking-policy adapter.

Not certified drop-in compatible with BeyondMimic/Isaac Lab; map body and joint names first.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from combat.robot import Robot

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--case',type=int,default=0);p.add_argument('--arm',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    z=np.load(a.input,allow_pickle=False);q=z['qpos'][a.case,a.arm];fps=float(z['fps']);robot=Robot();pos=[];quat=[]
    for pose in q:
        d=robot.forward(pose);pos.append(d.xpos[1:].copy());quat.append(d.xquat[1:].copy())
    pos=np.stack(pos);quat=np.stack(quat);ang=np.zeros_like(pos)
    for b in range(pos.shape[1]):
        r=Rotation.from_quat(quat[:,b][:,[1,2,3,0]]);ang[:-1,b]=(r[1:]*r[:-1].inv()).as_rotvec()*fps;ang[-1,b]=ang[-2,b]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,fps=np.array(fps),qpos=q,joint_pos=q[:,7:],joint_vel=robot.velocities(q,fps)[:,6:],body_pos_w=pos,body_quat_w=quat,body_lin_vel_w=np.gradient(pos,1/fps,axis=0),body_ang_vel_w=ang,joint_names=np.array(robot.names),body_names=np.array([robot.model.body(i).name for i in range(1,robot.model.nbody)]),quaternion_order=np.array('wxyz'),format_version=np.array('combat_reference_v1'))
    print('REFERENCE_EXPORTED',a.output,'CHECK TARGET BODY ORDER BEFORE TRAINING')

if __name__=='__main__':main()
