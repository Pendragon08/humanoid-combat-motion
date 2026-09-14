"""Two physical G1 articulations in each Isaac environment, frozen learned policy.

Initial diagnostic only: the controller was trained on one different seed motion.
Failures reflect both reference quality and controller coverage. Do not use these
labels as an unconditional certificate that a motion is impossible to execute.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--reference-a',type=Path,required=True)
p.add_argument('--reference-b',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4)
p.add_argument('--seed',type=int,default=123)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app
import copy
import hashlib
import json
import numpy as np
import torch
import gymnasium as gym
from isaaclab.managers import SceneEntityCfg,ObservationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


class FixedStart(MotionCommand):
    def _adaptive_sampling(self,env_ids):self.time_steps[env_ids]=0


class SinglePolicyView:
    """Expose original 29-action ABI to the unchanged upstream checkpoint loader."""
    def __init__(self,env):self.env=env;self.num_actions=29;self.num_envs=env.num_envs
    def get_observations(self):
        obs=self.env.observation_manager.compute()
        return obs['policy'],{'observations':obs}


def main():
    if a.output.exists():raise FileExistsError(a.output)
    refs=[a.reference_a.resolve(strict=True),a.reference_b.resolve(strict=True)]
    checkpoint=a.checkpoint.resolve(strict=True);root=Path(__file__).resolve().parents[1]
    cfg=load_cfg_from_registry('Tracking-Flat-G1-v0','env_cfg_entry_point')
    agent=load_cfg_from_registry('Tracking-Flat-G1-v0','rsl_rl_cfg_entry_point')
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs;cfg.scene.env_spacing=6.;cfg.sim.device=a.device
    cfg.scene.robot.spawn.usd_dir=str(root/'cache/usd/g1_tracking')
    cfg.scene.opponent=copy.deepcopy(cfg.scene.robot);cfg.scene.opponent.prim_path='{ENV_REGEX_NS}/Opponent'
    cfg.scene.opponent_contacts=copy.deepcopy(cfg.scene.contact_forces)
    cfg.scene.opponent_contacts.prim_path='{ENV_REGEX_NS}/Opponent/.*'
    cfg.commands.opponent=copy.deepcopy(cfg.commands.motion)
    cfg.actions.opponent_joint_pos=copy.deepcopy(cfg.actions.joint_pos);cfg.actions.opponent_joint_pos.asset_name='opponent'
    for cmd,asset,ref in zip((cfg.commands.motion,cfg.commands.opponent),('robot','opponent'),refs):
        cmd.asset_name=asset;cmd.motion_file=str(ref);cmd.class_type=FixedStart;cmd.debug_vis=False
        cmd.pose_range={};cmd.velocity_range={};cmd.joint_position_range=(0.,0.)
    # Nominal two-body diagnostic, no asymmetric startup or push perturbations.
    for name in ('physics_material','add_joint_default_pos','base_com','push_robot'):setattr(cfg.events,name,None)
    cfg.observations.policy.enable_corruption=False
    cfg.observations.policy.actions.params={'action_name':'joint_pos'}
    cfg.observations.critic.actions.params={'action_name':'joint_pos'}
    cfg.observations.opponent_policy=copy.deepcopy(cfg.observations.policy)
    for term in vars(cfg.observations.opponent_policy).values():
        if not isinstance(term,ObservationTermCfg):continue
        if 'command_name' in term.params:term.params['command_name']='opponent'
    for name in ('base_lin_vel','base_ang_vel','joint_pos','joint_vel'):
        term=getattr(cfg.observations.opponent_policy,name,None)
        if term is not None:term.params['asset_cfg']=SceneEntityCfg('opponent')
    cfg.observations.opponent_policy.actions.params={'action_name':'opponent_joint_pos'}
    for name in ('anchor_pos','anchor_ori','ee_body_pos'):
        term=copy.deepcopy(getattr(cfg.terminations,name));term.params['command_name']='opponent'
        if 'asset_cfg' in term.params:term.params['asset_cfg']=SceneEntityCfg('opponent')
        setattr(cfg.terminations,'opponent_'+name,term)
    sensor_names=[]
    for actor,robot,other in [(0,'Robot','Opponent'),(1,'Opponent','Robot')]:
        for side in ('left','right'):
            name=f'pair_contact_{actor}_{side}';sensor_names.append(name)
            setattr(cfg.scene,name,ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/{robot}/{side}_wrist_yaw_link',
                filter_prim_paths_expr=[f'{{ENV_REGEX_NS}}/{other}/torso_link'],history_length=1,update_period=0.))
    wrapped=gym.make('Tracking-Flat-G1-v0',cfg=cfg);env=wrapped.unwrapped
    try:
        obs,_=env.reset()
        commands=[env.command_manager.get_term(n) for n in ('motion','opponent')]
        for cmd in commands:
            cmd.time_steps.sub_(1);cmd._update_command()
        obs=env.observation_manager.compute()
        if obs['policy'].shape!=obs['opponent_policy'].shape:raise ValueError('Actor observation mismatch')
        runner=OnPolicyRunner(SinglePolicyView(env),agent.to_dict(),log_dir=None,device=a.device)
        runner.load(str(checkpoint),load_optimizer=False);policy=runner.get_inference_policy(device=a.device)
        if env.action_manager.total_action_dim!=58:raise ValueError('Expected two independent 29-joint action terms')
        if list(env.action_manager.active_terms)!=['joint_pos','opponent_joint_pos']:raise ValueError('Unexpected action order')
        counts=[cmd.motion.time_step_total for cmd in commands]
        if counts[0]!=counts[1]:raise ValueError('Unsynchronized pair')
        if any(abs(float(np.asarray(cmd.motion.fps).item())-1/env.step_dt)>1e-3 for cmd in commands):raise ValueError('FPS mismatch')
        horizon=counts[0]-2;alive=torch.ones(a.num_envs,dtype=torch.bool,device=a.device)
        failure=torch.full((a.num_envs,),horizon,device=a.device,dtype=torch.long)
        poses=[];forces=[];masks=[];errors=[];failure_reasons={}
        original_compute=env.termination_manager.compute
        def audit_termination():
            result=original_compute()
            for name in env.termination_manager.active_terms:
                failure_reasons[name]=failure_reasons.get(name,0)+int((env.termination_manager.get_term(name)&alive).sum())
            return result
        env.termination_manager.compute=audit_termination
        for t in range(horizon):
            with torch.inference_mode():
                pair=[];err=[]
                for cmd in commands:
                    robot=cmd.robot
                    pair.append(torch.cat([robot.data.root_pos_w-env.scene.env_origins,robot.data.root_quat_w,robot.data.joint_pos],dim=-1))
                    err.append((robot.data.joint_pos-cmd.joint_pos).abs().mean(-1))
                poses.append(torch.stack(pair,dim=1).cpu().numpy());masks.append(alive.cpu().numpy().copy())
                errors.append(torch.stack(err,dim=1).cpu().numpy())
                contact=[]
                for name in sensor_names:
                    matrix=env.scene[name].data.force_matrix_w
                    if matrix is None:raise RuntimeError('Filtered physical contact forces unavailable')
                    contact.append(torch.linalg.vector_norm(matrix,dim=-1).flatten(1).sum(-1))
                forces.append(torch.stack(contact,dim=-1).cpu().numpy())
                action=torch.cat([policy(obs['policy']),policy(obs['opponent_policy'])],dim=-1)
                obs,reward,terminated,truncated,extra=env.step(action)
                done=terminated|truncated;new=alive&done;failure[new]=t+1;alive&=~done
                if not alive.any():break
        a.output.mkdir(parents=True)
        np.savez_compressed(a.output/'rollout.npz',qpos=np.asarray(poses),alive=np.asarray(masks),
            wrist_torso_contact_force_n=np.asarray(forces),joint_error_rad=np.asarray(errors),fps=1/env.step_dt,
            joint_names=np.asarray(commands[0].robot.joint_names))
        result={'robots_per_environment':2,'environments':a.num_envs,'learned_tracking':True,
            'physics_simulator':'Isaac Sim 4.5 / Isaac Lab 2.1','nominal_reset':True,
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'reference_sha256':[hashlib.sha256(ref.read_bytes()).hexdigest() for ref in refs],
            'completion_fraction':float(alive.float().mean()),'first_failure_steps':failure.cpu().tolist(),
            'horizon_steps':horizon,'recorded_steps':len(poses),'termination_counts':failure_reasons,
            'maximum_recorded_wrist_torso_force_n':float(np.max(forces)),
            'contact_scope':'four directed wrist-body / opponent-torso physical force sensors only; not full collision audit',
            'critic_training_labels_approved':False,
            'claim':'two-robot frozen-controller diagnostic; single-seed controller coverage confounds feasibility'}
        (a.output/'metrics.json').write_text(json.dumps(result,indent=2));print('DUAL_G1_EVALUATION '+json.dumps(result),flush=True)
    finally:env.close()


try:main()
finally:
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)
