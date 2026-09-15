"""Bounded nominal TWIST single/dual MuJoCo deployment, never a kinematic replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from combat.twist_tracking import ROOT, UPSTREAM, Controller, reference, scene

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, action='append', default=[])
    p.add_argument('--standing', action='store_true'); p.add_argument('--dual-standing',action='store_true')
    p.add_argument('--seconds',type=float,default=5);p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--video',action='store_true')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    assert (len(a.reference) in (1,2)) or a.standing or a.dual_standing
    torch.manual_seed(123);np.random.seed(123);torch.set_num_threads(4)
    actors=2 if a.dual_standing else max(1,len(a.reference))
    model,xml=scene(actors);data=mujoco.MjData(model)
    weights=UPSTREAM/'assets/twist_general_motion_tracker.pt'
    policy=torch.jit.load(str(weights),map_location=a.device).eval()
    controllers=[Controller(model,i,policy,a.device) for i in range(actors)]
    refs=[reference(a.reference[i] if a.reference else None,c,a.seconds) for i,c in enumerate(controllers)]
    if a.dual_standing:
        # Engineering control only; not claimed as a generated combat pair.
        refs[1]['qpos'][:,1]=1.5
    n=min(len(r['qpos']) for r in refs)-1
    for c,r in zip(controllers,refs):
        data.qpos[c.rootq:c.rootq+32]=r['qpos'][0]
        data.qvel[c.rootv:c.rootv+31]=r['velocity'][0]
    mujoco.mj_forward(model,data)
    a.output.mkdir(parents=True)
    (a.output/'scene.xml').write_text(xml)
    renderer=writer=None
    if a.video:
        import imageio
        renderer=mujoco.Renderer(model,height=544,width=960)
        writer=imageio.get_writer(a.output/'rollout.mp4',fps=50)
        cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-15;cam.distance=3.5 if actors==2 else 2.7
    poses=[];errors=[];root_errors=[];contacts=[];falls=[];strict=[];body_errors=[]
    refdata=mujoco.MjData(model)
    tracked=('pelvis','torso_link','left_ankle_roll_link','right_ankle_roll_link',
             'left_elbow_link','right_elbow_link','left_wrist_roll_link','right_wrist_roll_link')
    bid=[[model.body(c.prefix+b).id for b in tracked] for c in controllers]
    collision_geoms=[sum(bool(model.geom_contype[g] or model.geom_conaffinity[g]) for g in range(model.ngeom)
                         if model.body(model.geom_bodyid[g]).name.startswith(c.prefix)) for c in controllers]
    pair_contact_steps=0;pair_force_peak=0.;initial_penetration=0.;initial_contacts=[];max_applied_torque=0.
    for k in range(data.ncon):
        co=data.contact[k];initial_penetration=max(initial_penetration,-float(co.dist))
        if co.dist < -.001:
            b1=model.body(model.geom_bodyid[co.geom1]).name;b2=model.body(model.geom_bodyid[co.geom2]).name
            kind='ground' if model.geom_bodyid[co.geom1]==0 or model.geom_bodyid[co.geom2]==0 else ('interrobot' if b1[:3]!=b2[:3] else 'self')
            initial_contacts.append({'body1':b1,'body2':b2,'kind':kind,'penetration_m':-float(co.dist)})
    for t in range(n):
        # Official high-level default queries one control frame into the future.
        for c,r in zip(controllers,refs): c.target(data,r['mimic'][min(t+1,n)])
        tick_force=0.;tick_count=0
        for substep in range(20):
            for c in controllers:c.torque(data)
            max_applied_torque=max(max_applied_torque,float(np.abs(data.ctrl).max()))
            mujoco.mj_step(model,data)
            for k in range(data.ncon):
                co=data.contact[k]
                b1=model.body(model.geom_bodyid[co.geom1]).name
                b2=model.body(model.geom_bodyid[co.geom2]).name
                if actors==2 and ((b1.startswith('a0_') and b2.startswith('a1_')) or (b1.startswith('a1_') and b2.startswith('a0_'))):
                    f=np.zeros(6);mujoco.mj_contactForce(model,data,k,f)
                    tick_force=max(tick_force,float(np.linalg.norm(f[:3])));tick_count+=1
        q=np.array([data.qpos[c.rootq:c.rootq+32].copy() for c in controllers]);poses.append(q)
        target=np.array([r['qpos'][t+1] for r in refs])
        je=np.mean(abs(q[:,7:]-target[:,7:]),axis=1);re=np.linalg.norm(q[:,:3]-target[:,:3],axis=1)
        for c,r in zip(controllers,refs):refdata.qpos[c.rootq:c.rootq+32]=r['qpos'][t+1]
        mujoco.mj_forward(model,refdata)
        be=np.array([np.linalg.norm(data.xpos[b]-refdata.xpos[b],axis=1).mean() for b in bid])
        fall=q[:,2]<.35
        errors.append(je);root_errors.append(re);body_errors.append(be);falls.append(fall)
        strict.append((je<.3)&(re<.15)&(be<.15)&~fall)
        contacts.append([tick_count,tick_force]);pair_contact_steps+=int(tick_count>0);pair_force_peak=max(pair_force_peak,tick_force)
        if writer:
            cam.lookat[:]=q[:,:3].mean(axis=0);cam.lookat[2]=.7
            renderer.update_scene(data,camera=cam);writer.append_data(renderer.render())
        if not np.isfinite(data.qpos).all():raise RuntimeError('Nonfinite physics')
    if writer:writer.close();renderer.close()
    errors=np.asarray(errors);strict=np.asarray(strict);falls=np.asarray(falls)
    survival=np.logical_and.accumulate(strict,axis=0)
    fallfree=np.logical_and.accumulate(~falls,axis=0)
    firstfall=[float((np.flatnonzero(falls[:,i])[0]+1)*.02) if falls[:,i].any() else None for i in range(actors)]
    results={'actors':actors,'frames':n,'duration_s':n*.02,'seed':123,'nominal_deterministic':True,
        'policy':'official TWIST pretrained TorchScript','checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'upstream_commit':'42d8c134739eee51f28d7cc0ff72a86728afb8dc','simulator':'MuJoCo '+mujoco.__version__,
        'torch':torch.__version__,'device':a.device,'training_steps_this_run':0,
        'max_commanded_torque_nm':max_applied_torque,
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'combat/twist_tracking.py',ROOT/'scripts/evaluate_twist.py')},
        'joint_names':controllers[0].names,'dropped_reference_joints':[r['dropped_joints'] for r in refs],
        'references':[{'path':r['source'],'sha256':hashlib.sha256(a.reference[i].read_bytes()).hexdigest() if a.reference else None} for i,r in enumerate(refs)],
        'joint_mae_rad':errors.mean(axis=0).tolist(),'root_error_m':np.mean(root_errors,axis=0).tolist(),
        'tracked_body_error_m':np.mean(body_errors,axis=0).tolist(),
        'no_fall_complete':(~falls.any(axis=0)).tolist(),'strict_complete':strict.all(axis=0).tolist(),
        'first_fall_s':firstfall,'joint_mae_before_fall_rad':[float(errors[fallfree[:,i],i].mean()) if fallfree[:,i].any() else None for i in range(actors)],
        'strict_prefix_s':(survival.sum(axis=0)*.02).tolist(),
        'collision_geoms_per_actor':collision_geoms,'initial_max_penetration_m':initial_penetration,
        'initial_penetrating_contacts':sorted(initial_contacts,key=lambda x:-x['penetration_m'])[:20],
        'interrobot_contact_control_frames':pair_contact_steps,'interrobot_peak_contact_force_n':pair_force_peak,
        'caveats':['Nominal deployment diagnostic; not same-simulator randomized BeyondMimic comparison.',
                   'Unqualified mean errors cover the full rollout including any post-fall frames; first-fall and strict prefix separately reported.',
                   'Four wrist pitch/yaw reference DOFs omitted in official 25-DOF model; metrics cover retained joints.',
                   'Any interrobot contact is not semantic attack/block success; no force/contact correctness claim.',
                   'Pretrained data overlap unknown; not a zero-shot generalization or synthetic-data utility result.']}
    np.savez_compressed(a.output/'rollout.npz',qpos=np.asarray(poses),reference=np.stack([r['qpos'][1:n+1] for r in refs],axis=1),
        joint_error=errors,root_error=np.array(root_errors),body_error=np.array(body_errors),falls=falls,strict=strict,contacts=np.array(contacts),fps=50)
    (a.output/'metrics.json').write_text(json.dumps(results,indent=2))
    print('TWIST_PHYSICS_WITNESS '+json.dumps(results),flush=True)

if __name__=='__main__':main()
