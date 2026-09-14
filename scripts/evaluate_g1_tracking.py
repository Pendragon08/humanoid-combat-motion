"""Fixed-start, censored-on-first-failure evaluation of an actual PPO policy.

This initial diagnostic uses one training reference, not held-out generalization.
Resets performed internally after failure are excluded from all reported metrics.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--motion_file',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--seed',type=int,default=123)
p.add_argument('--task',default='Tracking-Flat-G1-v0')
p.add_argument('--held-out',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
launcher=AppLauncher(a,sync_loads=False,multi_gpu=False);app=launcher.app

import hashlib
import json
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.sim import SimulationContext
import whole_body_tracking.tasks  # Registers pinned upstream task.
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


class FixedStartMotionCommand(MotionCommand):
    def _adaptive_sampling(self,env_ids):
        self.time_steps[env_ids]=0


def main():
    if a.output.exists():raise FileExistsError(a.output)
    checkpoint=a.checkpoint.resolve(strict=True);motion_path=a.motion_file.resolve(strict=True)
    cfg=load_cfg_from_registry(a.task,'env_cfg_entry_point')
    agent=load_cfg_from_registry(a.task,'rsl_rl_cfg_entry_point')
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device
    cfg.scene.robot.spawn.usd_dir=str(Path('cache/usd/g1_tracking').resolve())
    cfg.commands.motion.motion_file=str(motion_path)
    cfg.commands.motion.class_type=FixedStartMotionCommand
    cfg.commands.motion.debug_vis=False
    cfg.observations.policy.enable_corruption=False
    # Keep training morphology/material/reset perturbations and pushes. This is
    # a reproducible randomized-start diagnostic, not a pristine nominal test.
    agent.device=a.device;agent.seed=a.seed;agent.logger='tensorboard'
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    wrapped=gym.make(a.task,cfg=cfg);raw=wrapped.unwrapped;env=RslRlVecEnvWrapper(wrapped)
    try:
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=a.device)
        runner.load(str(checkpoint),load_optimizer=False)
        policy=runner.get_inference_policy(device=a.device)
        obs,_=env.reset()
        cmd=raw.command_manager.get_term('motion');robot=raw.scene['robot']
        # Upstream reset initializes qpos but leaves body_pos_relative_w zero on
        # the first reset; terminations are evaluated before the first command
        # update. Prime the upstream alignment cache at frame0 WITHOUT advancing
        # physics or reference time. Otherwise every episode fails spuriously.
        initial_times=cmd.time_steps.clone()
        cmd.time_steps.sub_(1);cmd._update_command()
        if not torch.equal(cmd.time_steps,initial_times):raise RuntimeError('Reference priming advanced time')
        obs,_=env.get_observations()
        reference_fps=float(np.asarray(cmd.motion.fps).item())
        if abs(reference_fps-1/raw.step_dt)>1e-3:
            raise ValueError('Reference FPS must equal control rate; no implicit time warp')
        horizon=cmd.motion.time_step_total-2
        if horizon<10:raise ValueError('Reference too short')
        alive=torch.ones(a.num_envs,dtype=torch.bool,device=a.device)
        first_failure=torch.full((a.num_envs,),horizon,dtype=torch.long,device=a.device)
        totals={k:0.0 for k in ('joint_mae_rad','root_error_m','anchor_error_m','body_error_m','aligned_body_error_m','reward')}
        per_env_sums={k:torch.zeros(a.num_envs,device=a.device) for k in ('joint_mae_rad','root_error_m','body_error_m')}
        per_env_counts=torch.zeros(a.num_envs,device=a.device)
        counts=0;poses=[];alive_record=[];actions_record=[];reference=[]
        first_step_diagnostic={}
        original_termination_compute=raw.termination_manager.compute
        def audited_termination_compute():
            result=original_termination_compute()
            if not first_step_diagnostic:
                first_step_diagnostic.update({
                    'termination_counts':{name:int(raw.termination_manager.get_term(name).sum()) for name in raw.termination_manager.active_terms},
                    'maximum_body_position_error_m':float(torch.linalg.vector_norm(cmd.robot_body_pos_w-cmd.body_pos_w,dim=-1).max()),
                    'maximum_joint_speed_rad_s':float(robot.data.joint_vel.abs().max()),
                    'maximum_root_speed_m_s':float(torch.linalg.vector_norm(robot.data.root_lin_vel_w,dim=-1).max()),
                    'root_height_range_m':[float(robot.data.root_pos_w[:,2].min()),float(robot.data.root_pos_w[:,2].max())],
                    'first_env_body_z_errors_m':(cmd.robot_body_pos_w[0,:,2]-cmd.body_pos_w[0,:,2]).cpu().tolist(),
                    'tracked_body_names':list(cmd.cfg.body_names)})
            return result
        raw.termination_manager.compute=audited_termination_compute
        for t in range(horizon):
            with torch.inference_mode():
                # Record before step: post-step observations can already be reset.
                q=torch.cat([robot.data.root_pos_w-raw.scene.env_origins,
                             robot.data.root_quat_w,robot.data.joint_pos],dim=-1)
                poses.append(q[0].cpu().numpy().copy());alive_record.append(bool(alive[0]))
                reference.append(torch.cat([cmd.motion._body_pos_w[t,0],
                    cmd.motion._body_quat_w[t,0],cmd.motion.joint_pos[t]]).cpu().numpy())
                n=int(alive.sum());counts+=n
                per_env_counts+=alive
                errors={
                    'joint_mae_rad':(robot.data.joint_pos-cmd.joint_pos).abs().mean(-1),
                    'root_error_m':torch.linalg.vector_norm(robot.data.root_pos_w-cmd.body_pos_w[:,0],dim=-1),
                    'anchor_error_m':torch.linalg.vector_norm(cmd.robot_anchor_pos_w-cmd.anchor_pos_w,dim=-1),
                    'body_error_m':torch.linalg.vector_norm(cmd.robot_body_pos_w-cmd.body_pos_w,dim=-1).mean(-1),
                    'aligned_body_error_m':torch.linalg.vector_norm(cmd.robot_body_pos_w-cmd.body_pos_relative_w,dim=-1).mean(-1)}
                for key,value in errors.items():totals[key]+=float(value[alive].sum())
                for key in per_env_sums:per_env_sums[key]+=errors[key]*alive
                actions=policy(obs);actions_record.append(actions[0].cpu().numpy())
                obs,reward,done,extra=env.step(actions)
                totals['reward']+=float(reward[alive].sum())
                failed=alive & done.bool();first_failure[failed]=t+1;alive &= ~failed
        arrays={'qpos':np.asarray(poses),'reference_qpos':np.asarray(reference),
                'actions':np.asarray(actions_record),'alive':np.asarray(alive_record),'fps':reference_fps,
                'joint_names':np.asarray(robot.joint_names)}
        a.output.mkdir(parents=True)
        np.savez_compressed(a.output/'rollout_env0.npz',**arrays)
        result={'evaluation':'fixed_start_held_out_reference' if a.held_out else 'fixed_start_training_reference_diagnostic','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'motion_sha256':hashlib.sha256(motion_path.read_bytes()).hexdigest(),'checkpoint':str(checkpoint),
                'seed':a.seed,'environments':a.num_envs,'control_dt':raw.step_dt,'horizon_steps':horizon,
                'completion_fraction':float(alive.float().mean()),'first_failure_steps':first_failure.cpu().tolist(),
                'completion_definition':'survived reference horizon under original primarily height/orientation termination rules; not accurate 3D tracking success',
                'per_environment_mean_errors':{k:(v/per_env_counts.clamp(min=1)).cpu().tolist() for k,v in per_env_sums.items()},
                'strict_tracking_pass_fraction':float((alive & (per_env_sums['body_error_m']/per_env_counts.clamp(min=1)<.15)
                    & (per_env_sums['root_error_m']/per_env_counts.clamp(min=1)<.15)
                    & (per_env_sums['joint_mae_rad']/per_env_counts.clamp(min=1)<.3)).float().mean()),
                'strict_thresholds':{'mean_body_position_error_m':.15,'mean_root_position_error_m':.15,'mean_joint_error_rad':.3,
                    'scope':'declared research diagnostic thresholds, not original BeyondMimic success definition'},
                'valid_active_env_steps':counts,'mean_alive_error':{k:v/max(counts,1) for k,v in totals.items()},
                'first_step_diagnostic':first_step_diagnostic,
                'learned_tracking':True,'held_out_motion':a.held_out,'robots_per_environment':1,
                'observation_noise':False,'training_reset_and_dynamics_randomization':True,
                'initial_reference_alignment_cache_primed':True,
                'metric_caveat':'Errors averaged only before first termination; always interpret together with completion fraction. Termination is tracking failure, not necessarily a fall.',
                'claim':'single-reference controller feasibility diagnostic, not paired physics labels or synthetic-data utility'}
        (a.output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('G1_TRACKING_EVALUATION '+json.dumps(result),flush=True)
    finally:
        env.close()


try:
    main()
finally:
    SimulationContext.clear_instance()
    app.close(wait_for_replicator=False)
