"""Run the official URCI entrypoint headless, stop after one complete motion.

No observation, action, PD, reference-library or random-delay implementation is
replaced. An output hook records state and raises a private stop exception.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'third_party/PBHC'
PIN='ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--example',choices=['pose','punch','pose2'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=123)
    a=parser.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()==PIN
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=UP,text=True).strip()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    torch.cuda.manual_seed_all(a.seed)
    x=torch.randn(16,16,device='cuda:0',requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
    assert torch.isfinite(x.grad).all();kernel=float(loss.detach())
    ckpts={'pose':'pretrained_horse_stance_pose/exported/model_50000.onnx',
           'punch':'pretrained_horse_stance_punch/exported/model_33000.onnx',
           'pose2':'pretrained_horse_stance_pose_2/exported/model_119000.onnx'}
    checkpoint=UP/'example'/ckpts[a.example]
    os.chdir(UP)
    from humanoidverse.deploy.urcirobot import URCIRobot
    class RolloutComplete(BaseException):pass
    traces={k:[] for k in ['qpos','qvel','joint_reference','joint_mae','torque','actions','root_reference','time_s']}
    start=time.time()
    def record(robot):
        # Called by the unmodified routing() after ApplyAction(), before reset.
        r=robot._kick_motion_res_buffer
        reference=r['dof_pos'][0].cpu().numpy()
        traces['qpos'].append(robot.data.qpos.copy());traces['qvel'].append(robot.data.qvel.copy())
        traces['joint_reference'].append(reference.copy())
        traces['joint_mae'].append(float(np.abs(robot.data.qpos[7:]-reference).mean()))
        traces['root_reference'].append(r['root_pos'][0].cpu().numpy().copy())
        traces['torque'].append(robot.data.ctrl.copy());traces['actions'].append(robot.act.copy())
        traces['time_s'].append(robot.timer*robot.dt)
        assert np.isfinite(robot.data.qpos).all()
        if len(traces['qpos'])>=math.ceil(robot.motion_len/robot.dt):
            out.mkdir(parents=True)
            data={k:np.asarray(v) for k,v in traces.items()}
            np.savez_compressed(out/'rollout.npz',**data,fps=1/robot.dt)
            from omegaconf import OmegaConf
            OmegaConf.save(robot.cfg,out/'resolved_config.yaml')
            q=data['qpos'];falls=q[:,2]<.35;strict=(data['joint_mae']<.3)&~falls
            first=np.flatnonzero(falls)
            motion=Path(robot.cfg.robot.motion.motion_file)
            result={'example':a.example,'upstream_commit':PIN,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                'motion':str(motion),'motion_sha256':sha(motion),'seed':a.seed,'frames':len(q),
                'duration_s':len(q)*robot.dt,'control_dt':robot.dt,'physics_dt':robot.sim_dt,
                'cuda_kernel_witness_loss':kernel,'wall_s':time.time()-start,
                'joint_mae_rad':float(data['joint_mae'].mean()),'all_frame_joint_below_0p3':bool((data['joint_mae']<.3).all()),
                'no_fall':not bool(falls.any()),'first_fall_s':float((first[0]+1)*robot.dt) if len(first) else None,
                'strict_joint_and_height_complete':bool(strict.all()),'minimum_pelvis_height_m':float(q[:,2].min()),
                'max_sampled_torque_nm':float(abs(data['torque']).max()),
                'official_random_delay_enabled':robot.RAND_DELAY,'delay_range_ms':list(robot.delay_ratio),
                'policy_observations':list(robot.cfg.obs.obs_dict.actor_obs),
                'mujoco':__import__('mujoco').__version__,'numpy':np.__version__,'torch':torch.__version__,
                'script_sha256':sha(Path(__file__)),'training_steps_this_run':0,
                'controller_modified':False,'headless':True,
                'caveats':['Single-motion trained policy, not a universal data-utility evaluator',
                           'Joint+height acceptance is not previous TWIST root/body strict score',
                           'CPU ONNX and MuJoCo; separate CUDA kernel witness is environment validation',
                           'No real-robot or contact-force correctness claim']}
            (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print('PBHC_OFFICIAL_WITNESS '+json.dumps(result),flush=True)
            raise RolloutComplete()
    URCIRobot.TrySaveMotionStep=record
    # No reference/model relocation needed: all bundled paths resolve from UP.
    sys.argv=['humanoidverse/urci.py','+simulator=mujoco',f'+checkpoint={checkpoint}',
              '++deploy.render=False','++env.config.save_motion=False',
              f'eval_log_dir={out.parent / (out.name+"_hydra")}',
              f'hydra.run.dir={out.parent / (out.name+"_hydra")}']
    try:runpy.run_path(str(UP/'humanoidverse/urci.py'),run_name='__main__')
    except RolloutComplete:pass
    if not (out/'metrics.json').exists():raise RuntimeError('Missing completed physical rollout')

if __name__=='__main__':main()
