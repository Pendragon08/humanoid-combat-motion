"""PM01 official USD/actuators in Isaac Lab: witness, FK export, actual PD diagnostic.

Reference export sets states and is NOT execution. Diagnostic steps physics with
a free root and finite PD actuators; failure is not universal untrackability.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','export','diagnostic'],required=True)
p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a,sync_loads=False,multi_gpu=False);app=launcher.app
import json
import hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.utils import configclass
from isaacsim.core.api.objects import GroundPlane
from engineai_rl_lab.tasks.tracking.robots.pm01 import PM01_CYLINDER_CFG

ROOT=Path(__file__).resolve().parents[1]
@configclass
class Scene(InteractiveSceneCfg):
    robot=PM01_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')

def main():
    if a.output.exists():raise FileExistsError(a.output)
    torch.manual_seed(0)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=a.device))
    GroundPlane('/World/Ground',size=30,z_position=0)
    cfg=Scene(num_envs=1,env_spacing=3)
    cfg.robot.spawn.usd_path=str(ROOT/'third_party/engineai_rl_lab/source/engineai_rl_lab/engineai_rl_lab/assets/pm01/serial_pm01_edu.usd')
    scene=InteractiveScene(cfg);sim.reset();robot=scene['robot']
    if len(robot.joint_names)!=24:raise ValueError(robot.joint_names)
    if a.mode=='witness':
        for _ in range(200):
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim();sim.step(render=False);scene.update(.005)
        net=torch.nn.Linear(24,8,device=a.device);loss=net(robot.data.joint_pos.detach()).square().mean();loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(robot.data.root_state_w).all():raise ValueError('Nonfinite witness')
        result={'witness':'PM01_ISAAC_PHYSICS_WITNESS','physics_steps':200,'joints':robot.joint_names,'bodies':robot.body_names,'root_position':robot.data.root_pos_w.cpu().tolist(),'gpu':torch.cuda.get_device_name(),'loss':loss.item(),'learned_tracking':False,'actuators':'official PM01_CYLINDER_CFG','environment':'Isaac Lab 2.1 compatibility overlay, not official 2.3.2 reproduction'}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    a.output.mkdir(parents=True);rows=[]
    paths=sorted(a.input.glob('sample_*_*.npz'))
    if not paths:raise ValueError('no input references')
    for path in paths:
        with np.load(path,allow_pickle=False) as z:q=z['qpos'];fps=float(z['fps']);names=z['joint_names'].tolist()
        indices=[names.index(n) for n in robot.joint_names]
        times=np.arange(len(q))/fps;newtimes=np.arange(0,times[-1]+1e-9,.02)
        pos=np.stack([np.interp(newtimes,times,q[:,i]) for i in range(3)],-1)
        quat=Slerp(times,Rotation.from_quat(q[:,[4,5,6,3]]))(newtimes).as_quat()[:,[3,0,1,2]]
        joints=np.stack([np.interp(newtimes,times,q[:,7+j]) for j in indices],-1)
        vel=np.gradient(joints,.02,axis=0);rootvel=np.gradient(pos,.02,axis=0)
        rots=Rotation.from_quat(quat[:,[1,2,3,0]]);rootang=np.zeros_like(pos);rootang[:-1]=(rots[1:]*rots[:-1].inv()).as_rotvec()/.02;rootang[-1]=rootang[-2]
        refroot=np.c_[pos,quat,rootvel,rootang]
        logs={k:[] for k in ['body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
        actual=[];actualq=[];torques=[]
        scene.reset()
        for t in range(len(joints)):
            if a.mode=='export' or t==0:
                robot.write_root_state_to_sim(torch.tensor(refroot[t:t+1],device=a.device,dtype=torch.float32))
                robot.write_joint_state_to_sim(torch.tensor(joints[t:t+1],device=a.device,dtype=torch.float32),torch.tensor(vel[t:t+1],device=a.device,dtype=torch.float32))
                sim.forward();scene.update(.005)
            if a.mode=='diagnostic':
                robot.set_joint_position_target(torch.tensor(joints[t:t+1],device=a.device,dtype=torch.float32))
                for _ in range(4):scene.write_data_to_sim();sim.step(render=False);scene.update(.005)
                actual.append(robot.data.root_pos_w[0].cpu().numpy().copy());actualq.append(robot.data.joint_pos[0].cpu().numpy().copy());torques.append(robot.data.applied_torque[0].cpu().numpy().copy())
            for key in logs:logs[key].append(getattr(robot.data,key)[0].cpu().numpy().copy())
        arrays={k:np.stack(v) for k,v in logs.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite motion')
        dest=a.output/path.name
        if a.mode=='export':
            np.savez_compressed(dest,fps=50,joint_pos=joints,joint_vel=vel,joint_names=np.array(robot.joint_names),body_names=np.array(robot.body_names),**arrays)
            row={'file':path.name,'frames':len(joints),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'state_replay_only':True,'physics_execution':False}
        else:
            actual=np.array(actual);actualq=np.array(actualq);error=np.sqrt(np.mean((actualq-joints)**2));rooterror=np.sqrt(np.mean(np.sum((actual-pos)**2,axis=-1)))
            fallen=actual[:,2]<.45;up=Rotation.from_quat(arrays['body_quat_w'][:,robot.body_names.index('LINK_BASE'),[1,2,3,0]]).as_matrix()[:,2,2]
            fallen |= up<.5
            np.savez_compressed(dest,actual_joint_pos=actualq,reference_joint_pos=joints,actual_root_pos=actual,reference_root_pos=pos,applied_torque=np.array(torques),**arrays)
            row={'file':path.name,'joint_rmse_rad':float(error),'root_position_rmse_m':float(rooterror),'fell':bool(fallen.any()),'first_failure_frame':int(np.argmax(fallen)) if fallen.any() else None,'frames':len(joints),'peak_abs_torque_Nm':float(np.abs(torques).max()),'physics_execution':True,'learned_tracking':False,'thresholds':'engineering diagnostic: root z<0.45m or upright cosine<0.5; not a standard combat success definition'}
        rows.append(row);print(json.dumps(row),flush=True)
    (a.output/'results.json').write_text(json.dumps({'mode':a.mode,'samples':rows,'environment':'official PM01 USD and actuator config on Isaac Lab 2.1 compatibility runtime','claim':'no imitation utility established'},indent=2))

try:main()
except BaseException:
    import traceback
    failure=traceback.format_exc();print(failure,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    (a.output.parent/(a.output.name+'_FAILED.txt')).write_text(failure)
    raise
finally:sim_utils.SimulationContext.clear_instance();app.close(wait_for_replicator=False)
