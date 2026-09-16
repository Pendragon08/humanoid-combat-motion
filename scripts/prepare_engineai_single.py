"""Transfer both actors of paired diffusion to solo PM01 references, with provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import mink
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation,Slerp
from scripts.retarget_paired_positions import anatomical_rotation

ROOT=Path(__file__).resolve().parents[1]
TARGETS={'LINK_BASE':(0,20.),'LINK_TORSO_YAW':(9,2.),'LINK_KNEE_PITCH_L':(4,2.),'LINK_KNEE_PITCH_R':(5,2.),'LINK_ANKLE_ROLL_L':(7,8.),'LINK_ANKLE_ROLL_R':(8,8.),'LINK_ELBOW_PITCH_L':(18,2.),'LINK_ELBOW_PITCH_R':(19,2.),'hand_L':(20,5.),'hand_R':(21,5.)}

def joint_project(q,limits,velocity,fps,acc=100.):
    n=len(q);eye=sparse.eye(n,format='csc')
    d1=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc')
    d2=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    A=sparse.vstack([eye,d1,d2],format='csc');P=2*(eye+.5*d2.T@d2);out=q.copy()
    for j in range(q.shape[1]):
        lo=np.r_[np.full(n,limits[j,0]),np.full(n-1,-velocity[j]/fps),np.full(n-2,-acc/fps**2)]
        hi=np.r_[np.full(n,limits[j,1]),np.full(n-1,velocity[j]/fps),np.full(n-2,acc/fps**2)]
        solver=osqp.OSQP();solver.setup(P=P,q=-2*q[:,j],A=A,l=lo,u=hi,verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=20000)
        result=solver.solve()
        if result.info.status_val not in [1,2]:raise RuntimeError(result.info.status)
        out[:,j]=result.x
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    model=mujoco.MjModel.from_xml_path(str(ROOT/'cache/engineai_overlay/pm01_kinematic.xml'))
    names=[model.joint(i).name for i in range(1,model.njnt)]
    if len(names)!=24:raise ValueError(names)
    limits=model.jnt_range[1:].copy()
    velocity=np.array([26.3 if ('HIP_PITCH' in n or 'HIP_ROLL' in n or 'KNEE' in n) else 35.2 for n in names])
    tasks=[mink.FrameTask(name,'site' if name.startswith('hand') else 'body',position_cost=w,orientation_cost=1. if name in ['LINK_BASE','LINK_TORSO_YAW'] else 0.,lm_damping=1.) for name,(idx,w) in TARGETS.items()]
    rows=[];manifest=json.loads((a.input/'manifest.json').read_text());a.output.mkdir(parents=True)
    for sample in range(3):
        path=a.input/f'sample_{sample:03d}.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        entry=next(r for r in manifest['samples'] if r['sample_id']==path.stem)
        if entry['sha256']!=digest:raise ValueError('source hash mismatch')
        with np.load(path,allow_pickle=False) as z:x=z['positions'].astype(float);fps=float(z['fps'])
        if x.shape[1:]!=(2,22,3) or not np.isfinite(x).all():raise ValueError('bad paired input')
        for actor in range(2):
            human=x[:,actor];target=human[:, :, [0,2,1]].copy();target[:,:,1]*=-1
            # Robot/human median hip-to-ankle scale, measured from kinematics.
            config=mink.Configuration(model);q0=config.data.qpos.copy()
            q0[3:7]=[1,0,0,0];config.update(q0)
            robot_leg=np.mean([np.linalg.norm(config.data.xpos[model.body('LINK_BASE').id]-config.data.xpos[model.body(f'LINK_ANKLE_ROLL_{s}').id]) for s in ['L','R']])
            human_leg=np.median(np.linalg.norm(target[:,0,None]-target[:,[7,8]],axis=-1));scale=robot_leg/human_leg
            target*=scale;target[:,:,:2]-=target[0,0,:2].copy()
            # Constant initial floor alignment only; no per-frame root teleport.
            target[:,:,2]+=.05-target[0,[7,8],2].min()
            qs=[];errs=[]
            q0[:3]=target[0,0];q0[3:7]=anatomical_rotation(target[0]);config.update(q0)
            for frame in target:
                rot=anatomical_rotation(frame)
                for task,(name,(idx,w)) in zip(tasks,TARGETS.items()):task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot),frame[idx]))
                for _ in range(10):
                    v=mink.solve_ik(config,tasks,1/fps,'daqp',damping=.05,limits=[mink.ConfigurationLimit(model)])
                    config.integrate_inplace(v,1/fps)
                qs.append(config.data.qpos.copy())
                errs.append(np.mean([np.linalg.norm(config.get_transform_frame_to_world(n,'site' if n.startswith('hand') else 'body').translation()-frame[j]) for n,(j,w) in TARGETS.items()]))
            source_q=np.stack(qs)
            # Enforce temporal limits at the actual 50Hz control rate, AFTER
            # interpolation: a 30Hz acceleration certificate does not survive
            # piecewise-linear resampling automatically.
            source_t=np.arange(len(source_q))/fps;output_t=np.arange(0,source_t[-1]+1e-9,.02)
            raw=np.stack([np.interp(output_t,source_t,source_q[:,i]) for i in range(model.nq)],-1)
            raw[:,3:7]=Slerp(source_t,Rotation.from_quat(source_q[:,[4,5,6,3]]))(output_t).as_quat()[:,[3,0,1,2]]
            outfps=50.
            constrained=raw.copy();constrained[:,7:]=joint_project(raw[:,7:],limits,velocity,outfps)
            for arm,q in [('raw_ik',raw),('joint_constrained',constrained)]:
                dest=a.output/f'{path.stem}_actor{actor}_{arm}.npz'
                np.savez_compressed(dest,qpos=q,fps=outfps,joint_names=np.array(names),source_target_positions=target,source_fps=fps,source_actor=actor,quaternion_order='wxyz')
                vel=np.diff(q[:,7:],axis=0)*outfps;acc=np.diff(q[:,7:],n=2,axis=0)*outfps**2
                rows.append({'file':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_sha256':digest,'actor':actor,'arm':arm,'text':entry['text'],'role':'index only; attacker/defender semantics not verified','scale':float(scale),'ik_body_error_m':float(np.mean(errs)),'joint_limit_fraction':float(np.mean((q[:,7:]<limits[:,0]-1e-4)|(q[:,7:]>limits[:,1]+1e-4))),'speed_violation_fraction':float(np.mean(np.abs(vel)>velocity+1e-3)),'acceleration_screen_violation_fraction':float(np.mean(np.abs(acc)>100.1)),'peak_speed_rad_s':float(np.max(np.abs(vel))),'joint_change_rms_rad':float(np.sqrt(np.mean((q[:,7:]-raw[:,7:])**2))),'dynamic_feasibility':False})
                print(json.dumps(rows[-1]),flush=True)
    (a.output/'manifest.json').write_text(json.dumps({'mode':'paired-generator actor extraction, no new generator training','model_sha256':hashlib.sha256((ROOT/'cache/engineai_overlay/pm01_kinematic.xml').read_bytes()).hexdigest(),'joint_names':names,'samples':rows,'contact_scope':'solo shadowboxing; no opponent contact labels','acceleration_limit':'100 rad/s2 research setting'},indent=2))

if __name__=='__main__':main()
