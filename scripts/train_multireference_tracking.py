"""Train the pinned PPO tracker on a boundary-safe reference bundle."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--num_envs',type=int,default=512);p.add_argument('--seed',type=int,default=0)
p.add_argument('--checkpoint',type=Path);p.add_argument('--smoke-steps',type=int,default=0)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a,sync_loads=False,multi_gpu=False);app=launcher.app
import hashlib
import json
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.sim import SimulationContext
from isaaclab.utils.io import dump_yaml
import whole_body_tracking.tasks
from combat.multireference_tracking import configure_multireference


def main():
    if a.output.exists():raise FileExistsError(a.output)
    cfg=load_cfg_from_registry('Tracking-Flat-G1-v0','env_cfg_entry_point')
    agent=load_cfg_from_registry('Tracking-Flat-G1-v0','rsl_rl_cfg_entry_point')
    cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device=a.device
    cfg.scene.robot.spawn.usd_dir=str(Path('cache/usd/g1_tracking').resolve())
    cfg.commands.motion.motion_file=str(a.bundle.resolve(strict=True));cfg.commands.motion.debug_vis=False
    configure_multireference(cfg)
    agent.seed=a.seed;agent.device=a.device;agent.logger='tensorboard';agent.save_interval=500
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    raw=gym.make('Tracking-Flat-G1-v0',cfg=cfg);env=RslRlVecEnvWrapper(raw)
    a.output.mkdir(parents=True)
    try:
        dump_yaml(str(a.output/'env.yaml'),cfg);dump_yaml(str(a.output/'agent.yaml'),agent)
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(a.output),device=a.device)
        if a.checkpoint:runner.load(str(a.checkpoint.resolve(strict=True)),load_optimizer=True)
        command=raw.unwrapped.command_manager.get_term('motion')
        record={'seed':a.seed,'environments':a.num_envs,'iterations':a.iterations,
            'bundle_sha256':hashlib.sha256(a.bundle.read_bytes()).hexdigest(),'clip_ids':command.clip_ids,
            'initial_checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,
            'sampling':'uniform clip, uniform valid frame; explicit clip timeout; no interpolation across clips',
            'algorithm':'upstream RSL-RL PPO, original BeyondMimic observations/rewards',
            'global_anchor_soft_reward':True,'paired_world_geometry_guaranteed':False,'evaluation_not_training':False}
        (a.output/'config.json').write_text(json.dumps(record,indent=2))
        if a.smoke_steps:
            obs,_=env.reset();policy=runner.get_inference_policy(device=a.device)
            clip_timeouts=0
            for _ in range(a.smoke_steps):
                with torch.inference_mode():obs,_,_,_=env.step(policy(obs))
                if not torch.isfinite(obs).all():raise RuntimeError('Nonfinite observation')
                if torch.any(command.time_steps<command.clip_starts[command.selected_clip]) or torch.any(command.time_steps>=command.clip_ends[command.selected_clip]):raise RuntimeError('Clip boundary violation')
                clip_timeouts+=int(raw.unwrapped.termination_manager.get_term('reference_clip_end').sum())
            result={'steps':a.smoke_steps,'clip_timeouts':clip_timeouts,'sample_counts':command.sample_counts.cpu().tolist(),
                    'boundary_violations':0,'finite_observations':True,'learned_training':False}
            (a.output/'smoke.json').write_text(json.dumps(result,indent=2));print('MULTIREFERENCE_SMOKE',json.dumps(result),flush=True)
        else:
            runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
            (a.output/'TRAINING_COMPLETE.json').write_text(json.dumps({'iterations_requested':a.iterations,'sample_counts':command.sample_counts.cpu().tolist()},indent=2))
    finally:env.close()


try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    print('MULTIREFERENCE_FAILED',flush=True)
    raise
finally:
    SimulationContext.clear_instance();app.close(wait_for_replicator=False)
