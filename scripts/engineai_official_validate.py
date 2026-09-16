"""Bounded evaluation using pinned upstream Lab and native RSL-RL 5 runner.

No compatibility task patch or checkpoint architecture transplant. Rendering-only
settings and evaluation time/start rules are disclosed separately from training.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['witness','eval','train'],required=True)
p.add_argument('--motion',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--seed',type=int,default=0)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a,sync_loads=False,multi_gpu=False).app
from engineai_official_packages import activate
activate()
import hashlib
import importlib.metadata as meta
import json
import numpy as np
import torch
import isaaclab
import rsl_rl
import gymnasium
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from engineai_rl_lab.tasks.tracking.config.pm01.flat_env_cfg import PM01FlatWoStateEstimationEnvCfg
from engineai_rl_lab.tasks.tracking.config.pm01.agents.rsl_rl_ppo_cfg import PM01FlatPPORunnerCfg
from engineai_rl_lab.tasks.tracking.mdp.commands import MotionCommand

ROOT=Path(__file__).resolve().parents[1]
class FixedStart(MotionCommand):
    def _adaptive_sampling(self,env_ids):self.time_steps[env_ids]=0

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if a.output.exists():raise FileExistsError(a.output)
    assert 'IsaacLab-engineai-official' in isaaclab.__file__,isaaclab.__file__
    assert torch.__version__=='2.7.0+cu128',torch.__version__
    import sys
    wrong=[name for name,module in sys.modules.items() if (name=='torch' or name.startswith('torch.')) and isinstance(getattr(module,'__file__',None),str) and Path(module.__file__).is_file() and '.venv-engineai-official/' not in str(module.__file__)]
    assert not wrong,wrong
    assert meta.version('rsl-rl-lib')=='5.0.1'
    cfg=PM01FlatWoStateEstimationEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=a.seed
    cfg.scene.robot.spawn.usd_path=str(ROOT/'third_party/engineai_rl_lab/source/engineai_rl_lab/engineai_rl_lab/assets/pm01/serial_pm01_edu.usd')
    cfg.scene.terrain.visual_material=None;cfg.scene.contact_forces.debug_vis=False
    cfg.commands.motion.motion_file=str(a.motion.resolve(strict=True));cfg.commands.motion.debug_vis=False
    if a.mode!='train':
        cfg.commands.motion.class_type=FixedStart;cfg.observations.policy.enable_corruption=False
        with np.load(a.motion) as z:cfg.episode_length_s=max(cfg.episode_length_s,len(z['joint_pos'])/float(np.asarray(z['fps']).item())+1.)
    agent_cfg=PM01FlatPPORunnerCfg();agent_cfg.device=a.device;agent_cfg.seed=a.seed;agent_cfg.logger='tensorboard';agent_cfg.save_interval=100
    agent=agent_cfg.to_dict()
    # Same sanitation as official scripts/tracking/play.py; not a task change.
    allowed={'class_name','hidden_dims','activation','obs_normalization','distribution_cfg'}
    for model in ['actor','critic']:agent[model]={k:v for k,v in agent[model].items() if k in allowed}
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    raw=ManagerBasedRLEnv(cfg);env=RslRlVecEnvWrapper(raw)
    a.output.mkdir(parents=True)
    try:
        runner=OnPolicyRunner(env,agent,log_dir=str(a.output) if a.mode=='train' else None,device=a.device)
        # Native upstream checkpoint schema. No hand-built network conversion.
        public=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
        runner.alg.actor.load_state_dict(public['actor_state_dict'],strict=True)
        runner.alg.critic.load_state_dict(public['critic_state_dict'],strict=True)
        if a.mode=='train':
            with torch.no_grad():runner.alg.actor.distribution.std_param.fill_(.3)
        protocol={'versions':{'isaaclab':meta.version('isaaclab'),'rsl-rl-lib':meta.version('rsl-rl-lib'),'torch':torch.__version__,'gymnasium':gymnasium.__version__},'distribution_metadata':{x:meta.version(x) for x in ['torch','gymnasium']},'module_files':{'torch':torch.__file__,'rsl_rl':rsl_rl.__file__,'gymnasium':gymnasium.__file__},'torch_module_path_assertion_passed':True,'isaaclab_file':isaaclab.__file__,'motion':str(a.motion),'motion_sha256':digest(a.motion),'checkpoint_sha256':digest(a.checkpoint),'script_sha256':digest(__file__),'mode':a.mode,'agent':agent,'environments':a.num_envs,'seed':a.seed,'iterations':a.iterations,'training_changes':'fresh optimizer and exploration std0.3 for both arms' if a.mode=='train' else None,'eval_changes':['fixed reference start','no observation noise','full-reference time limit','no visual material/debug markers'],'upstream_task_rewards_and_terminations':'unchanged','claim':'environment validation/reference fit; not data utility'}
        (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        if a.mode=='train':
            runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
            (a.output/'COMPLETE.json').write_text(json.dumps({'iterations':a.iterations,'environment_steps':a.iterations*24*a.num_envs}));return
        cmd=raw.command_manager.get_term('motion');robot=raw.scene['robot']
        if cmd.motion.joint_names and cmd.motion.joint_names!=robot.joint_names:raise ValueError('Joint order mismatch')
        if abs(float(np.asarray(cmd.motion.fps).item())-1/raw.step_dt)>1e-3:raise ValueError('FPS mismatch')
        policy=runner.get_inference_policy(device=a.device);env.reset()
        cmd.time_steps.sub_(1);cmd._update_command();obs=env.get_observations()
        horizon=200 if a.mode=='witness' else cmd.motion.time_step_total-2
        alive=torch.ones(a.num_envs,dtype=torch.bool,device=a.device);counts=torch.zeros(a.num_envs,device=a.device)
        sums={k:torch.zeros_like(counts) for k in ['joint_mae_rad','root_error_m','body_error_m']}
        first=torch.full((a.num_envs,),horizon,device=a.device,dtype=torch.long);poses=[];living=[]
        for t in range(horizon):
            with torch.inference_mode():
                errors={'joint_mae_rad':(robot.data.joint_pos-cmd.joint_pos).abs().mean(-1),'root_error_m':torch.linalg.vector_norm(robot.data.root_pos_w-cmd.body_pos_w[:,0],dim=-1),'body_error_m':torch.linalg.vector_norm(cmd.robot_body_pos_w-cmd.body_pos_w,dim=-1).mean(-1)}
                counts+=alive
                for k,v in errors.items():sums[k]+=v*alive
                poses.append(torch.cat([robot.data.root_pos_w[0]-raw.scene.env_origins[0],robot.data.root_quat_w[0],robot.data.joint_pos[0]]).cpu().numpy().copy());living.append(bool(alive[0]))
                obs,reward,done,_=env.step(policy(obs))
                if not all(torch.isfinite(v).all() for v in obs.values()):raise FloatingPointError('Nonfinite observations')
                failed=alive & done.bool();first[failed]=t+1;alive &= ~failed
        x=torch.randn(16,16,device=a.device,requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
        strict=alive & (sums['joint_mae_rad']/counts.clamp(min=1)<.3) & (sums['root_error_m']/counts.clamp(min=1)<.15) & (sums['body_error_m']/counts.clamp(min=1)<.15)
        result={'horizon':horizon,'physics_steps_per_env':horizon*cfg.decimation,'completion_fraction':float(alive.float().mean()),'strict_tracking_fraction':float(strict.float().mean()),'first_failure_steps':first.cpu().tolist(),'mean_alive_errors':{k:float(v.sum()/counts.sum().clamp(min=1)) for k,v in sums.items()},'gradient_loss':loss.item(),'gpu':torch.cuda.get_device_name(a.device),'claim':'reference-fit diagnostic; not unseen data utility','native_runner':True}
        np.savez_compressed(a.output/'rollout_env0.npz',qpos=np.array(poses),alive=np.array(living),fps=50,joint_names=np.array(robot.joint_names))
        (a.output/'results.json').write_text(json.dumps(result,indent=2));print('ENGINEAI_OFFICIAL_WITNESS' if a.mode=='witness' else 'ENGINEAI_OFFICIAL_RESULT',json.dumps(result),flush=True)
    finally:env.close()

try:main()
except BaseException:
    import traceback
    failure=traceback.format_exc();print(failure,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);(a.output.parent/(a.output.name+'_FAILED.txt')).write_text(failure)
    raise
finally:SimulationContext.clear_instance();app.close(wait_for_replicator=False)
