"""Bounded PM01 official-task PPO compatibility pilot; no new control algorithm."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['smoke','train','eval'],required=True);p.add_argument('--checkpoint',type=Path)
p.add_argument('--official-actor',action='store_true',help='Load public newer-schema deterministic actor, without optimizer/critic transfer')
p.add_argument('--warm-start',type=Path,help='Official public actor+critic; reset optimizer and noise, disable observation normalization')
p.add_argument('--iterations',type=int,default=300);p.add_argument('--num_envs',type=int,default=256);p.add_argument('--seed',type=int,default=0)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a,sync_loads=False,multi_gpu=False);app=launcher.app
import json
import hashlib
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import math as math_utils
# Isaac Lab 2.3 renamed this rotation helper; preserve inverse unit-quaternion
# rotation rather than changing the termination rule.
if not hasattr(math_utils,'quat_apply_inverse'):
    math_utils.quat_apply_inverse=lambda q,v: math_utils.quat_apply(math_utils.quat_conjugate(q),v)
from engineai_rl_lab.tasks.tracking.config.pm01.flat_env_cfg import PM01FlatWoStateEstimationEnvCfg
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand

ROOT=Path(__file__).resolve().parents[1]
class FixedStart(MotionCommand):
    def _adaptive_sampling(self,env_ids):self.time_steps[env_ids]=0

def main():
    if a.output.exists():raise FileExistsError(a.output)
    cfg=PM01FlatWoStateEstimationEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=a.seed
    cfg.scene.robot.spawn.usd_path=str(ROOT/'third_party/engineai_rl_lab/source/engineai_rl_lab/engineai_rl_lab/assets/pm01/serial_pm01_edu.usd')
    cfg.scene.terrain.visual_material=None;cfg.scene.contact_forces.debug_vis=False
    cfg.commands.motion.motion_file=str(a.motion.resolve(strict=True));cfg.commands.motion.debug_vis=False
    if a.mode=='eval':
        cfg.commands.motion.class_type=FixedStart;cfg.observations.policy.enable_corruption=False
        with np.load(a.motion,allow_pickle=False) as z:
            # A training episode timeout is not a physical tracking failure.
            # Permit the full reference during fixed-start evaluation.
            cfg.episode_length_s=max(cfg.episode_length_s,len(z['joint_pos'])/float(np.asarray(z['fps']).item())+1.)
    # Match official architecture/PPO settings using the installed RSL-RL 2.3
    # ActorCritic schema. This is explicitly not the upstream newer runner.
    agent={'seed':a.seed,'num_steps_per_env':24,'save_interval':100,'empirical_normalization':True,'logger':'tensorboard',
        'policy':{'class_name':'ActorCritic','init_noise_std':1.,'actor_hidden_dims':[512,256,128],'critic_hidden_dims':[512,256,128],'activation':'elu'},
        'algorithm':{'class_name':'PPO','value_loss_coef':1.,'use_clipped_value_loss':True,'clip_param':.2,'entropy_coef':.005,'num_learning_epochs':5,'num_mini_batches':4,'learning_rate':.001,'schedule':'adaptive','gamma':.99,'lam':.95,'desired_kl':.01,'max_grad_norm':1.}}
    if a.warm_start:agent['empirical_normalization']=False
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    raw=ManagerBasedRLEnv(cfg);env=RslRlVecEnvWrapper(raw)
    a.output.mkdir(parents=True)
    try:
        cmd=raw.command_manager.get_term('motion');robot=raw.scene['robot']
        if abs(float(np.asarray(cmd.motion.fps).item())-1/raw.step_dt)>1e-3:raise ValueError('FPS mismatch')
        if cmd.motion.joint_names and cmd.motion.joint_names!=robot.joint_names:raise ValueError('Joint name order mismatch')
        runner=OnPolicyRunner(env,agent,log_dir=str(a.output) if a.mode=='train' else None,device=a.device)
        if a.warm_start:
            public=torch.load(a.warm_start,map_location=a.device,weights_only=True)
            transferred={}
            for prefix,source in [('actor','actor_state_dict'),('critic','critic_state_dict')]:
                for key,value in public[source].items():
                    if key.startswith('mlp.'):transferred[prefix+'.'+key[4:]]=value
                    elif key!='distribution.std_param':raise ValueError('Unknown pretrained field')
            current=runner.alg.policy.state_dict()
            if any(k not in current or current[k].shape!=v.shape for k,v in transferred.items()):raise ValueError('Warm-start architecture mismatch')
            current.update(transferred)
            # Same moderate exploration in both arms; optimizer starts fresh.
            if 'std' in current:current['std']=torch.full_like(current['std'],.3)
            runner.alg.policy.load_state_dict(current,strict=True)
        provenance={'robot':'PM01','mode':a.mode,'reference':str(a.motion),'reference_sha256':hashlib.sha256(a.motion.read_bytes()).hexdigest(),'iterations':a.iterations,'environments':a.num_envs,'seed':a.seed,'runtime':'Isaac Lab2.1/RSL-RL2.3 compatibility','agent':agent,'claim':'single-reference feasibility; not heldout data utility'}
        provenance['warm_start_sha256']=hashlib.sha256(a.warm_start.read_bytes()).hexdigest() if a.warm_start else None
        provenance['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (a.output/'protocol.json').write_text(json.dumps(provenance,indent=2))
        if a.mode=='train':
            runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
            (a.output/'COMPLETE.json').write_text(json.dumps({'iterations':a.iterations,'environment_steps':a.iterations*24*a.num_envs}))
        else:
            if a.mode=='eval':
                if a.checkpoint is None:raise ValueError('checkpoint required')
                if not a.official_actor:runner.load(str(a.checkpoint.resolve(strict=True)),load_optimizer=False)
            policy=runner.get_inference_policy(device=a.device);obs,_=env.reset()
            if a.official_actor:
                if a.mode!='eval':raise ValueError('official actor only for evaluation')
                public=torch.load(a.checkpoint,map_location=a.device,weights_only=True)['actor_state_dict']
                allowed={f'mlp.{i}.{suffix}' for i in [0,2,4,6] for suffix in ['weight','bias']}|{'distribution.std_param'}
                if set(public)!=allowed:raise ValueError('Unexpected public actor fields; normalization cannot be guessed')
                layers=[]
                for i in [0,2,4,6]:
                    w=public[f'mlp.{i}.weight'];layer=torch.nn.Linear(w.shape[1],w.shape[0],device=a.device)
                    layer.load_state_dict({'weight':w,'bias':public[f'mlp.{i}.bias']});layers.append(layer)
                    if i!=6:layers.append(torch.nn.ELU())
                policy=torch.nn.Sequential(*layers).eval()
                if obs.shape[-1]!=public['mlp.0.weight'].shape[1]:raise ValueError('Official observation dimension mismatch')
            cmd.time_steps.sub_(1);cmd._update_command();obs,_=env.get_observations()
            horizon=cmd.motion.time_step_total-2 if a.mode=='eval' else 32
            alive=torch.ones(a.num_envs,dtype=torch.bool,device=a.device)
            first_failure=torch.full((a.num_envs,),horizon,device=a.device,dtype=torch.long)
            counts=torch.zeros(a.num_envs,device=a.device);sums={k:torch.zeros_like(counts) for k in ['joint_mae_rad','root_error_m','body_error_m']}
            poses=[];living=[]
            for t in range(horizon):
                with torch.inference_mode():
                    errors={'joint_mae_rad':(robot.data.joint_pos-cmd.joint_pos).abs().mean(-1),'root_error_m':torch.linalg.vector_norm(robot.data.root_pos_w-cmd.body_pos_w[:,0],dim=-1),'body_error_m':torch.linalg.vector_norm(cmd.robot_body_pos_w-cmd.body_pos_w,dim=-1).mean(-1)}
                    counts+=alive
                    for k,v in errors.items():sums[k]+=v*alive
                    poses.append(torch.cat([robot.data.root_pos_w[0]-raw.scene.env_origins[0],robot.data.root_quat_w[0],robot.data.joint_pos[0]]).cpu().numpy().copy());living.append(bool(alive[0]))
                    obs,reward,done,_=env.step(policy(obs))
                    if not torch.isfinite(obs).all():raise FloatingPointError('nonfinite rollout')
                    failed=alive & done.bool();first_failure[failed]=t+1;alive &= ~failed
            errors={k:(v/counts.clamp(min=1)).cpu().tolist() for k,v in sums.items()}
            strict=alive & (sums['joint_mae_rad']/counts.clamp(min=1)<.3) & (sums['root_error_m']/counts.clamp(min=1)<.15) & (sums['body_error_m']/counts.clamp(min=1)<.15)
            result={'physics_steps_per_env':horizon*4,'completion_fraction':float(alive.float().mean()),'strict_tracking_fraction':float(strict.float().mean()),'first_failure_steps':first_failure.cpu().tolist(),'mean_alive_errors':{k:float(v.sum()/counts.sum().clamp(min=1)) for k,v in sums.items()},'per_env_errors':errors,'horizon':horizon,'learned_policy':a.mode=='eval','official_pretrained_actor':a.official_actor,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() if a.checkpoint else None,'randomized_dynamics_and_resets':True,'threshold_source':'research diagnostic, not official universal standard','first_failure_censored':True,'claim':'training-reference fit only; no unseen generalization or data utility'}
            np.savez_compressed(a.output/'rollout_env0.npz',qpos=np.array(poses),alive=np.array(living),fps=50,joint_names=np.array(robot.joint_names))
            (a.output/'results.json').write_text(json.dumps(result,indent=2));print('ENGINEAI_TRACKING_RESULT',json.dumps(result),flush=True)
    finally:env.close()

try:main()
except BaseException:
    import traceback
    failure=traceback.format_exc();print(failure,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    (a.output.parent/(a.output.name+'_FAILED.txt')).write_text(failure)
    raise
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
