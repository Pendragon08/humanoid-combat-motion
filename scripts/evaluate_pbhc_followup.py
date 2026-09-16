"""Full-horizon development evaluation, recording physics BEFORE automatic reset.

Diagnostic disturbance ablation; original randomized results remain primary.
Every-frame strict metrics are separate from trajectory-mean historical metrics.
No root alignment in scoring. Original training randomization remains enabled.
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
p.add_argument('--regime',choices=['no_push','nominal'],required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(); launcher=AppLauncher(a,sync_loads=False,multi_gpu=False); app=launcher.app
import hashlib
import json
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.sim import SimulationContext
import whole_body_tracking.tasks
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


class FixedStartMotionCommand(MotionCommand):
    def _adaptive_sampling(self,env_ids):
        self.time_steps[env_ids]=0


def main():
    if a.output.exists():raise FileExistsError(a.output)
    cfg=load_cfg_from_registry('Tracking-Flat-G1-v0','env_cfg_entry_point')
    agent=load_cfg_from_registry('Tracking-Flat-G1-v0','rsl_rl_cfg_entry_point')
    cfg.seed=a.seed; cfg.scene.num_envs=a.num_envs; cfg.sim.device=a.device
    cfg.scene.robot.spawn.usd_dir=str(Path('cache/usd/g1_tracking').resolve())
    cfg.commands.motion.motion_file=str(a.motion_file.resolve(strict=True))
    cfg.commands.motion.class_type=FixedStartMotionCommand; cfg.commands.motion.debug_vis=False
    cfg.observations.policy.enable_corruption=False
    cfg.events.push_robot=None
    if a.regime=='nominal':
        cfg.events.physics_material=None
        cfg.events.add_joint_default_pos=None
        cfg.events.base_com=None
        cfg.commands.motion.pose_range={k:(0.,0.) for k in cfg.commands.motion.pose_range}
        cfg.commands.motion.velocity_range={k:(0.,0.) for k in cfg.commands.motion.velocity_range}
        cfg.commands.motion.joint_position_range=(0.,0.)
    assert cfg.observations.policy.motion_anchor_pos_b is not None
    agent.device=a.device; agent.seed=a.seed; agent.logger='tensorboard'
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    wrapped=gym.make('Tracking-Flat-G1-v0',cfg=cfg); raw=wrapped.unwrapped
    env=RslRlVecEnvWrapper(wrapped)
    try:
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=a.device)
        runner.load(str(a.checkpoint.resolve(strict=True)),load_optimizer=False)
        policy=runner.get_inference_policy(device=a.device)
        obs,_=env.reset(); cmd=raw.command_manager.get_term('motion'); robot=raw.scene['robot']
        cmd.time_steps.sub_(1); cmd._update_command()
        assert torch.all(cmd.time_steps==0)
        obs,_=env.get_observations()
        fps=float(np.asarray(cmd.motion.fps).item()); assert abs(fps-1/raw.step_dt)<1e-5
        horizon=cmd.motion.time_step_total-1
        tracked=['pelvis','torso_link','left_ankle_roll_link','right_ankle_roll_link',
                 'left_elbow_link','right_elbow_link','left_wrist_roll_link','right_wrist_roll_link']
        bids=torch.tensor([robot.body_names.index(n) for n in tracked],device=a.device)
        core=torch.tensor([i for i,n in enumerate(robot.joint_names) if 'wrist' not in n],device=a.device)
        assert len(core)==23 and robot.body_names[0]=='pelvis'
        alive=torch.ones(a.num_envs,dtype=torch.bool,device=a.device)
        strict=alive.clone(); common_strict=alive.clone(); fall_free=alive.clone()
        sums=torch.zeros((a.num_envs,4),device=a.device); counts=torch.zeros(a.num_envs,device=a.device)
        first_failure=torch.full((a.num_envs,),horizon,device=a.device,dtype=torch.long)
        first_strict=first_failure.clone(); records=[]; poses=[]; refs=[]; masks=[]
        old_compute=raw.termination_manager.compute
        clock=[0]
        def measure_before_reset():
            result=old_compute()
            t=clock[0]+1
            # Manager computes termination before advancing motion commands.
            if not torch.all(cmd.time_steps[alive]==t-1):raise RuntimeError('Reference clock mismatch')
            ref_root=cmd.motion._body_pos_w[t,0]+raw.scene.env_origins
            ref_bodies=cmd.motion._body_pos_w[t,bids][None]+raw.scene.env_origins[:,None,:]
            joint=(robot.data.joint_pos-cmd.motion.joint_pos[t]).abs().mean(-1)
            common=(robot.data.joint_pos[:,core]-cmd.motion.joint_pos[t,core]).abs().mean(-1)
            root=torch.linalg.vector_norm(robot.data.root_pos_w-ref_root,dim=-1)
            body=torch.linalg.vector_norm(robot.data.body_pos_w[:,bids]-ref_bodies,dim=-1).mean(-1)
            height=robot.data.root_pos_w[:,2]-raw.scene.env_origins[:,2]
            errors=torch.stack([joint,common,root,body],dim=-1)
            if not torch.isfinite(errors[alive]).all():raise ValueError('Nonfinite active rollout')
            sums.add_(errors*alive[:,None]); counts.add_(alive)
            passed=(joint<.3)&(root<.15)&(body<.15)&(height>=.35)&~result.bool()
            cp=(common<.3)&(root<.15)&(body<.15)&(height>=.35)&~result.bool()
            just_failed=strict&~passed
            first_strict[just_failed]=t
            strict.logical_and_(passed); common_strict.logical_and_(cp)
            fall_free.logical_and_((height>=.35)|~alive)
            poses.append(torch.cat([robot.data.root_pos_w[0]-raw.scene.env_origins[0],
                         robot.data.root_quat_w[0],robot.data.joint_pos[0]]).cpu().numpy().copy())
            refs.append(torch.cat([cmd.motion._body_pos_w[t,0],cmd.motion._body_quat_w[t,0],
                                   cmd.motion.joint_pos[t]]).cpu().numpy().copy())
            records.append(errors.cpu().numpy().copy()); masks.append(alive.cpu().numpy().copy())
            failed=alive&result.bool(); first_failure[failed]=t; alive.logical_and_(~result.bool())
            return result
        raw.termination_manager.compute=measure_before_reset
        for t in range(horizon):
            clock[0]=t
            with torch.inference_mode():obs,_,_,_=env.step(policy(obs))
        assert len(records)==horizon
        means=sums/counts.clamp(min=1)[:,None]
        mean_pass=alive&(means[:,0]<.3)&(means[:,2]<.15)&(means[:,3]<.15)
        a.output.mkdir(parents=True)
        np.savez_compressed(a.output/'rollout.npz',qpos=poses,reference=refs,errors=records,alive=masks,
                            fps=fps,joint_names=np.array(robot.joint_names),tracked_body_names=np.array(tracked))
        receipt_path=Path(str(a.checkpoint)+'.receipt.json')
        result=dict(scope='Single-motion training-reference calibration; no generalization/data-utility claim',
                    checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                    training_receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None,
                    reference_sha256=hashlib.sha256(a.motion_file.read_bytes()).hexdigest(),
                    seed=a.seed,environments=a.num_envs,frames=horizon,duration_s=horizon/fps,
                    completion_fraction=float(alive.float().mean()),
                    strict_every_frame_pass_fraction=float(strict.float().mean()),
                    common23_strict_every_frame_pass_fraction=float(common_strict.float().mean()),
                    trajectory_mean_pass_fraction=float(mean_pass.float().mean()),
                    per_environment_mean_errors=means.cpu().tolist(),
                    error_columns=['joint29_rad','joint23_rad','root_m','body8_m'],
                    first_failure_steps=first_failure.cpu().tolist(), first_strict_failure_steps=first_strict.cpu().tolist(),
                    no_observed_height_fall_fraction=float(fall_free.float().mean()),
                    mean_active_errors=(sums.sum(0)/counts.sum().clamp(min=1)).cpu().tolist(),
                    position_feedback=True,observation_noise=False,training_randomization=(a.regime!='nominal'),
                    evaluation_regime=a.regime,external_pushes=False,
                    startup_and_reset_randomization=(a.regime!='nominal'),
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    thresholds={'joint_mae_rad':.3,'root_error_m':.15,'body8_error_m':.15,'minimum_root_height_m':.35},
                    caveats=['Error metrics include first termination frame; later resets excluded.',
                             'No observed height fall is censored by other terminations; use completion.',
                             'Diagnostic ablation of disturbances; nominal results do not replace original randomized results. No head-to-head ranking against MuJoCo policies.',
                             'Body8 metric differs from historical BeyondMimic body14; both every-frame and mean definitions labeled.'])
        (a.output/'metrics.json').write_text(json.dumps(result,indent=2))
        print('PBHC_BEYONDMIMIC_EVAL_COMPLETE',json.dumps(result),flush=True)
    finally:env.close()


try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
