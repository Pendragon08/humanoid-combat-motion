"""Batch paired named-qpos references through upstream interpolation and Isaac FK.

Forward kinematics is not a physical rollout. Both actor coordinates remain in
the shared input world; no independent recentering or ground snapping is done.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--output-fps',type=int,default=50)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app
import ast
import hashlib
import json
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat,quat_conjugate,quat_mul,quat_slerp
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

root=Path(__file__).resolve().parents[1]
source=root/'third_party/whole_body_tracking/scripts/csv_to_npz.py'
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionLoader')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    robot=G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')


def main():
    if a.output.exists():raise FileExistsError(a.output)
    cfg=ReplaySceneCfg(num_envs=1,env_spacing=2.)
    cfg.robot.spawn.usd_dir=str(root/'cache/usd/g1_tracking')
    sim=SimulationContext(sim_utils.SimulationCfg(device=a.device,dt=1/a.output_fps))
    scene=InteractiveScene(cfg);sim.reset();robot=scene['robot']
    a.output.mkdir(parents=True);records=[]
    for path in sorted(a.input.glob('sample_*.npz')):
        z=np.load(path,allow_pickle=False);q=z['qpos'];fps=float(z['fps'])
        if q.ndim!=3 or q.shape[1:]!=(2,36):raise ValueError('Expected paired G1 qpos')
        joint_indexes=robot.find_joints(list(z['joint_names']),preserve_order=True)[0]
        if len(joint_indexes)!=29:raise ValueError('Incomplete joint mapping')
        for actor in range(2):
            data=torch.tensor(q[:,actor],device=sim.device,dtype=torch.float32)
            if not torch.allclose(data[:,3:7].norm(dim=-1),torch.ones(len(data),device=sim.device),atol=1e-4):raise ValueError('Non-unit quaternion')
            motion=MotionLoader.__new__(MotionLoader)
            motion.input_fps=fps;motion.output_fps=a.output_fps;motion.input_dt=1/fps;motion.output_dt=1/a.output_fps
            motion.device=sim.device;motion.input_frames=len(data);motion.duration=(len(data)-1)/fps
            motion.motion_base_poss_input=data[:,:3];motion.motion_base_rots_input=data[:,3:7];motion.motion_dof_poss_input=data[:,7:]
            motion._interpolate_motion();motion._compute_velocities()
            keys=('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')
            log={k:[] for k in keys}
            for t in range(motion.output_frames):
                state=torch.cat([motion.motion_base_poss[t],motion.motion_base_rots[t],motion.motion_base_lin_vels[t],motion.motion_base_ang_vels[t]])[None]
                robot.write_root_state_to_sim(state)
                jp=robot.data.default_joint_pos.clone();jv=robot.data.default_joint_vel.clone()
                jp[:,joint_indexes]=motion.motion_dof_poss[t];jv[:,joint_indexes]=motion.motion_dof_vels[t]
                robot.write_joint_state_to_sim(jp,jv)
                sim.render();scene.update(sim.get_physics_dt())
                for key in keys:log[key].append(getattr(robot.data,key)[0].cpu().numpy().copy())
            arrays={k:np.stack(v) for k,v in log.items()}
            if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Invalid FK output')
            dest=a.output/f'{path.stem}_actor{actor}.npz'
            np.savez_compressed(dest,fps=np.array([a.output_fps]),joint_names=np.array(robot.joint_names),body_names=np.array(robot.body_names),**arrays)
            records.append({'sample':path.stem,'actor':actor,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'output':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'frames':motion.output_frames})
            print('PAIRED_REFERENCE_CONVERTED '+json.dumps(records[-1]),flush=True)
    (a.output/'manifest.json').write_text(json.dumps({'records':records,'output_fps':a.output_fps,
        'upstream_interpolation_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'simulation_steps':0,'shared_world_preserved':True,'purpose':'reference conversion; not dynamic validation'},indent=2))


try:main()
finally:
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)
