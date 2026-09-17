"""Matched data-arm calibration; upstream training unchanged, initialization audited."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['upstream','uniform','adaptive'],required=True)
p.add_argument('--iterations',type=int,default=1000);p.add_argument('--num_envs',type=int,default=512)
p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke-steps',type=int,default=0)
p.add_argument('--match-initialization',type=Path)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();launcher=AppLauncher(a,sync_loads=False,multi_gpu=False);app=launcher.app
import hashlib
import time
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
from combat.diagnostic_tracking import FailureRateMotionCommand

def main():
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.bundle,allow_pickle=False) as z:
        ids=list(z['clip_ids']);fps=float(z['fps'])
    if a.mode=='upstream' and len(ids)!=1:raise ValueError('Unmodified upstream is single-reference only')
    cfg=load_cfg_from_registry('Tracking-Flat-G1-v0','env_cfg_entry_point')
    agent=load_cfg_from_registry('Tracking-Flat-G1-v0','rsl_rl_cfg_entry_point')
    cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device=a.device
    cfg.scene.robot.spawn.usd_dir=str(Path('cache/usd/g1_tracking').resolve())
    cfg.commands.motion.motion_file=str(a.bundle.resolve(strict=True));cfg.commands.motion.debug_vis=False
    if a.mode!='upstream':configure_multireference(cfg)
    if a.mode=='adaptive':cfg.commands.motion.class_type=FailureRateMotionCommand
    if abs(cfg.sim.dt*cfg.decimation-1/fps)>1e-9:raise ValueError('Reference/control time mismatch')
    agent.seed=a.seed;agent.device=a.device;agent.logger='tensorboard';agent.save_interval=500
    if cfg.observations.policy.motion_anchor_pos_b is None or cfg.observations.policy.base_lin_vel is None:
        raise ValueError('Position-feedback baseline required')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    raw=gym.make('Tracking-Flat-G1-v0',cfg=cfg);env=RslRlVecEnvWrapper(raw)
    a.output.mkdir(parents=True)
    command=raw.unwrapped.command_manager.get_term('motion')
    exposure=torch.zeros(len(ids),device=a.device,dtype=torch.long)
    termination_totals={}
    original_step=env.step
    def counted_step(action):
        clips=command.selected_clip.clone() if hasattr(command,'selected_clip') else torch.zeros(a.num_envs,device=a.device,dtype=torch.long)
        result=original_step(action)
        exposure.add_(torch.bincount(clips,minlength=len(ids)))
        manager=raw.unwrapped.termination_manager
        for name in manager.active_terms:
            termination_totals[name]=termination_totals.get(name,0)+int(manager.get_term(name).sum())
        return result
    env.step=counted_step
    try:
        dump_yaml(str(a.output/'env.yaml'),cfg);dump_yaml(str(a.output/'agent.yaml'),agent)
        config={'mode':a.mode,'seed':a.seed,'iterations':a.iterations,'environments':a.num_envs,
                'steps_per_env':agent.num_steps_per_env,'clip_ids':ids,'fps':fps,
                'bundle_sha256':hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
                'initialization':'random; no checkpoint',
                'scope':'Matched-budget two-source development data-arm calibration; not independent test or full synthetic data utility',
                'position_feedback':True, 'task':'Tracking-Flat-G1-v0','environment_spec_unchanged':True,
                'sampling':'upstream single-reference' if a.mode=='upstream' else a.mode+' clip; uniform valid frame',
                'adaptive_parameters':{'ema':.1,'initial_failure_rate':.5,'uniform_mixture':.2,'weight_formula':'.5+clamp(failure_rate,0,1)'},
                'upstream_vs_uniform_caveat':'also changes reset alignment and reference-end handling; not pure sampling ablation'}
        (a.output/'config.json').write_text(json.dumps(config,indent=2))
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(a.output),device=a.device)
        def state_digest(state):
            digest=hashlib.sha256()
            for key,value in sorted(state.items()):
                tensor=value.detach().cpu().contiguous()
                digest.update(key.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
            return digest.hexdigest()
        initial=dict(actor_critic_sha256=state_digest(runner.alg.policy.state_dict()),
                     seed=a.seed,random_initialization=True)
        if hasattr(runner,'obs_normalizer'):
            initial['obs_normalizer_sha256']=state_digest(runner.obs_normalizer.state_dict())
        if hasattr(runner,'privileged_obs_normalizer'):
            initial['privileged_obs_normalizer_sha256']=state_digest(runner.privileged_obs_normalizer.state_dict())
        if a.match_initialization:
            expected=json.loads(a.match_initialization.read_text())
            for key in [k for k in initial if k.endswith('_sha256')]:
                if initial[key]!=expected[key]:raise RuntimeError('Initial parameter mismatch: '+key)
            initial['matched_against']=str(a.match_initialization)
        (a.output/'initialization.json').write_text(json.dumps(initial,indent=2))
        # Upstream initializes logger_type inside learn(); initial snapshot comes before it.
        runner.logger_type='tensorboard'
        original_save=runner.save
        start_time=time.time()
        def recorded_save(path,infos=None):
            original_save(path,infos)
            record={'checkpoint':str(path),'environment_steps':int(exposure.sum()),
                    'completed_updates':int(exposure.sum())//(a.num_envs*agent.num_steps_per_env),
                    'elapsed_s':time.time()-start_time,'position_feedback':True,
                    'bundle_sha256':config['bundle_sha256'], 'seed':a.seed}
            Path(str(path)+'.receipt.json').write_text(json.dumps(record,indent=2))
            (a.output/'progress.json').write_text(json.dumps(record,indent=2))
        runner.save=recorded_save
        runner.save(str(a.output/'initial_random.pt'))
        if a.smoke_steps:
            obs,_=env.reset();policy=runner.get_inference_policy(device=a.device)
            for _ in range(a.smoke_steps):
                with torch.inference_mode():obs,_,_,_=env.step(policy(obs))
                if not torch.isfinite(obs).all():raise ValueError('Nonfinite observation')
                if hasattr(command,'clip_ends'):
                    if torch.any(command.time_steps<command.clip_starts[command.selected_clip]) or torch.any(command.time_steps>=command.clip_ends[command.selected_clip]):raise ValueError('Boundary crossing')
            marker='SMOKE_COMPLETE.json'
        else:
            runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
            marker='TRAINING_COMPLETE.json'
        expected=a.num_envs*(a.smoke_steps if a.smoke_steps else a.iterations*agent.num_steps_per_env)
        if int(exposure.sum())!=expected:raise ValueError('Exposure count mismatch')
        stats={'environment_steps':int(exposure.sum()),'per_clip_steps':exposure.cpu().tolist(),
               'reset_counts':command.sample_counts.cpu().tolist() if hasattr(command,'sample_counts') else None,
               'termination_counts':termination_totals,'termination_counts_may_overlap':True,
               'failure_rates':command.failure_rates.cpu().tolist() if hasattr(command,'failure_rates') else None}
        (a.output/marker).write_text(json.dumps(stats,indent=2))
        print('TRACKING_DIAGNOSTIC_COMPLETE',a.mode,marker,flush=True)
    finally:env.close()

try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
