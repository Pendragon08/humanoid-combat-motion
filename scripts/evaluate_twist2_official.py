"""Bounded headless deployment using executable AST from pinned official TWIST2.

No import/initialization of Redis, viewer, SDK or real robot interfaces.
Policy update and PD arithmetic are extracted from upstream, not reimplemented.
"""
import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time
import numpy as np
import mujoco
import torch
import onnxruntime as ort
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'third_party/TWIST2'
PIN='d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def official_runtime(weight,frequency):
    source=UP/'deploy_real/server_low_level_g1_sim.py'
    tree=ast.parse(source.read_text())
    scope={'np':np,'torch':torch,'ort':ort,'deque':deque}
    rotfile=UP/'deploy_real/data_utils/rot_utils.py'
    rot=next(n for n in ast.parse(rotfile.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='quatToEuler')
    exec(compile(ast.Module(body=[rot],type_ignores=[]),str(rotfile),'exec'),scope)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['OnnxPolicyWrapper','load_onnx_policy']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RealTimePolicyController')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    fields={'num_actions','sim_dt','sim_decimation','last_action','default_dof_pos','mujoco_default_dof_pos',
        'stiffness','damping','torque_limits','action_scale','ankle_idx','n_mimic_obs','n_proprio','n_obs_single','history_len','total_obs_size'}
    cfg=SimpleNamespace();scope.update(self=cfg,policy_frequency=frequency)
    for n in init.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr in fields:
            exec(compile(ast.Module(body=[n],type_ignores=[]),str(source),'exec'),scope)
    cfg.proprio_history_buf=deque([np.zeros(cfg.n_obs_single,dtype=np.float32) for _ in range(cfg.history_len)],maxlen=cfg.history_len)
    cfg.device='cpu';cfg.policy=scope['load_onnx_policy'](str(weight),'cpu')
    run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loop=next(n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i')
    control=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
    names={'rpy','obs_body_dof_vel','obs_proprio','obs_full','obs_hist','future_obs','obs_buf','obs_tensor','raw_action','scaled_actions','pd_target'}
    selected=[]
    for n in control.body:
        if isinstance(n,ast.Assign):
            target=n.targets[0]
            if (isinstance(target,ast.Name) and target.id in names or
                isinstance(target,ast.Attribute) and target.attr=='last_action' or
                isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='obs_body_dof_vel'):
                selected.append(n)
        elif isinstance(n,ast.With) and any(isinstance(x,ast.Name) and x.id=='raw_action' for x in ast.walk(n)):
            selected.append(n)
        elif isinstance(n,ast.Assert):selected.append(n)
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='append' and ast.unparse(n.value.func.value)=='self.proprio_history_buf':
            selected.append(n)
    pd=[n for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='torque']
    assert len(pd)==2 and any('self.policy' in ast.unparse(n) for n in selected)
    return cfg,scope,compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),compile(ast.Module(body=pd,type_ignores=[]),str(source),'exec'),ast.unparse(ast.Module(body=selected+pd,type_ignores=[]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path);p.add_argument('--standing',action='store_true')
    p.add_argument('--checkpoint',choices=['20k','25k'],default='20k');p.add_argument('--hz',type=int,choices=[50,100],default=100)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.standing != bool(a.reference)
    if a.output.exists():raise FileExistsError(a.output)
    assert subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(UP),'diff','--name-only'],text=True).strip()
    torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123)
    x=torch.randn(16,16,device='cuda:0',requires_grad=True);loss=(x@x).square().mean();loss.backward()
    assert torch.isfinite(x.grad).all()
    weight=UP/f'assets/ckpts/twist2_1017_{a.checkpoint}.onnx'
    cfg,scope,update,pd,fragment=official_runtime(weight,a.hz)
    xml=UP/'assets/g1/g1_sim2sim_29dof.xml'
    model=mujoco.MjModel.from_xml_path(str(xml));model.opt.timestep=.001
    data=mujoco.MjData(model);rd=mujoco.MjData(model)
    assert model.nq==36 and model.nv==35 and model.nu==29
    names=[model.joint(i).name for i in range(model.njnt) if model.jnt_type[i]!=0]
    if a.standing:
        n=5*50+1;q=np.tile(cfg.mujoco_default_dof_pos,(n,1));vel=np.zeros((n,35));mapping=[]
    else:
        with np.load(a.reference,allow_pickle=False) as z:
            assert float(z['fps'])==50
            source_names=list(z['joint_names']);b=list(z['body_names']).index('pelvis')
            n=len(z['joint_pos']);q=np.zeros((n,36));vel=np.zeros((n,35))
            q[:,:3]=z['body_pos_w'][:,b];q[:,3:7]=z['body_quat_w'][:,b]
            vel[:,:3]=z['body_lin_vel_w'][:,b]
            vel[:,3:6]=Rotation.from_quat(q[:,[4,5,6,3]]).inv().apply(z['body_ang_vel_w'][:,b])
            mapping=[]
            for j,name in enumerate(names):
                if name in source_names:
                    k=source_names.index(name);q[:,7+j]=z['joint_pos'][:,k];vel[:,6+j]=z['joint_vel'][:,k]
                else:mapping.append(name)
    # Keep the same complete50Hz reference;100Hz policy repeats each command twice.
    euler=np.array([scope['quatToEuler'](r) for r in q[:,3:7]])
    localvel=Rotation.from_quat(q[:,[4,5,6,3]]).inv().apply(vel[:,:3])
    mimic=np.c_[localvel[:,:2],q[:,2],euler[:,:2],vel[:,5],q[:,7:]]
    assert mimic.shape==(n,35)
    data.qpos[:]=q[0];data.qvel[:]=vel[0];mujoco.mj_forward(model,data)
    tracked=['pelvis','torso_link','left_ankle_roll_link','right_ankle_roll_link','left_elbow_link','right_elbow_link','left_wrist_roll_link','right_wrist_roll_link']
    bids=[model.body(s).id for s in tracked]
    core=[j for j,name in enumerate(names) if 'wrist' not in name]
    poses=[];joint=[];common=[];root=[];body=[];fall=[];strict=[];yaw=[];torques=[];clips=0;steps=0;obs_record=[]
    t0=time.time()
    for t in range(n-1):
        for sub in range(20):
            scope.update(dof_pos=data.qpos[7:].copy(),dof_vel=data.qvel[6:].copy(),quat=data.qpos[3:7].copy(),ang_vel=data.qvel[3:6].copy(),action_mimic=mimic[t+1])
            if sub%int(cfg.sim_decimation)==0:
                exec(update,scope)
                if len(obs_record)<10:obs_record.append(scope['obs_buf'].copy())
            exec(pd,scope)
            torque=scope['torque'];data.ctrl[:]=torque
            clips+=int(np.any(abs(torque)>=cfg.torque_limits-1e-6));steps+=1;torques.append(torque.copy())
            mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data);rd.qpos[:]=q[t+1];mujoco.mj_forward(model,rd)
        actual=data.qpos.copy();poses.append(actual)
        je=float(np.mean(abs(actual[7:]-q[t+1,7:])));ce=float(np.mean(abs(actual[7:][core]-q[t+1,7:][core])))
        re=float(np.linalg.norm(actual[:3]-q[t+1,:3]));be=float(np.linalg.norm(data.xpos[bids]-rd.xpos[bids],axis=1).mean());f=bool(actual[2]<.35)
        joint.append(je);common.append(ce);root.append(re);body.append(be);fall.append(f);strict.append(je<.3 and re<.15 and be<.15 and not f)
        dy=scope['quatToEuler'](actual[3:7])[2]-euler[t+1,2];yaw.append(float(abs(np.arctan2(np.sin(dy),np.cos(dy)))))
        assert np.isfinite(actual).all()
    a.output.mkdir(parents=True);(a.output/'official_executable_fragment.py').write_text(fragment+'\n')
    np.savez_compressed(a.output/'rollout.npz',qpos=np.asarray(poses),reference=q[1:],joint_error=joint,common_joint_error=common,root_error=root,body_error=body,falls=fall,strict=strict,yaw_error=yaw,fps=50,first_observations=obs_record)
    first=lambda seq:float((np.flatnonzero(seq)[0]+1)*.02) if np.any(seq) else None
    record={'tracker':'TWIST2','upstream_commit':PIN,'checkpoint':a.checkpoint,'checkpoint_sha256':sha(weight),'policy_hz':a.hz,'metric_hz':50,'physics_hz':1000,
        'frames':n-1,'duration_s':(n-1)*.02,'wall_s':time.time()-t0,'reference_sha256':sha(a.reference) if a.reference else None,
        'joint_names':names,'added_neutral_joints':mapping,'common_nonwrist_joint_count':len(core),
        'joint_mae_rad':float(np.mean(joint)),'common_joint_mae_rad':float(np.mean(common)),'root_error_m':float(np.mean(root)),
        'body_error_m':float(np.mean(body)),'yaw_error_rad':float(np.mean(yaw)),'no_fall':not any(fall),'first_fall_s':first(fall),
        'strict_complete':all(strict),'strict_prefix_s':float(np.logical_and.accumulate(strict).sum()*.02),
        'first_joint_failure_s':first(np.array(joint)>=.3),'first_root_failure_s':first(np.array(root)>=.15),'first_body_failure_s':first(np.array(body)>=.15),
        'torque_clip_substep_fraction':clips/steps,'max_commanded_torque_nm':float(np.max(abs(np.array(torques)))),
        'cuda_witness_loss':float(loss.detach().cpu()),'onnx_provider':'CPUExecutionProvider','training_steps':0,
        'mujoco':mujoco.__version__,'numpy':np.__version__,'code_sha256':sha(Path(__file__)),'xml_sha256':sha(xml),
        'caveats':['Official observation/PD AST, offline reference at t+1, no Redis/viewer/wallclock wait.',
                   'Native29DOF model differs from old25DOF model; controller-system diagnostic, not isolated algorithm superiority.',
                   '100Hz repeats50Hz reference commands twice; full duration and amplitude unchanged.',
                   'Pretraining overlap unknown; this is not synthetic-data training utility.']}
    (a.output/'metrics.json').write_text(json.dumps(record,indent=2));print('TWIST2_OFFICIAL_WITNESS '+json.dumps(record),flush=True)

if __name__=='__main__':main()
