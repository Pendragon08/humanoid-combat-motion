"""SONIC/HumanTracker deployment diagnostic; not official C++ stack reproduction."""
import argparse,ast,hashlib,json,subprocess,time
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import mujoco
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1]
HT=ROOT/'third_party/HumanTracker'
PIN='79d637da5e07b8e2c588ca613d3254f9d8acc0f7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def backend():
    scope={'np':np,'R':R,'Dict':Dict}
    geom=HT/'src/humantracker/eval/core/geometry.py'
    nodes=[n for n in ast.parse(geom.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['quat_conj','quat_mul','quat_rotate']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(geom),'exec'),scope)
    file=HT/'src/humantracker/eval/backends/sonic.py';tree=ast.parse(file.read_text());nodes=[];started=False
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='NUM_JOINTS':started=True
        if isinstance(n,ast.ClassDef) and n.name=='SonicMjSim':break
        if started and isinstance(n,(ast.Assign,ast.FunctionDef,ast.ClassDef)):nodes.append(n)
    nodes+=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='action_to_motor_targets']
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),scope)
    return scope,file
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path);p.add_argument('--standing',action='store_true');p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert a.standing != bool(a.reference)
    if a.output.exists():raise FileExistsError(a.output)
    assert subprocess.check_output(['git','-C',str(HT),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(HT),'diff','--name-only'],text=True).strip()
    torch.set_num_threads(4);torch.manual_seed(123);x=torch.randn(16,16,device='cuda:0',requires_grad=True);loss=(x@x).square().mean();loss.backward();assert torch.isfinite(x.grad).all()
    b,file=backend();weights=ROOT/'assets/sonic_release_20260916'
    metadata=json.loads((weights/'model_metadata.json').read_text(encoding='utf-8-sig'))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    enc=ort.InferenceSession(str(weights/'model_encoder.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    dec=ort.InferenceSession(str(weights/'model_decoder.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    xml=HT/'storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml'
    model=mujoco.MjModel.from_xml_path(str(xml));model.opt.timestep=.001;data=mujoco.MjData(model);rd=mujoco.MjData(model)
    names=[model.joint(i).name for i in range(model.njnt) if model.jnt_type[i]!=0]
    assert model.nq==36 and model.nu==29 and len(names)==29
    if a.standing:q=np.tile(b['SONIC_DEFAULT_QPOS'],(251,1)).astype(float);v=np.zeros((251,35));added=[]
    else:
        with np.load(a.reference,allow_pickle=False) as z:
            assert float(z['fps'])==50
            sn=list(z['joint_names']);pelvis=list(z['body_names']).index('pelvis');q=np.zeros((len(z['joint_pos']),36));v=np.zeros((len(q),35))
            q[:,:3]=z['body_pos_w'][:,pelvis];q[:,3:7]=z['body_quat_w'][:,pelvis];v[:,:3]=z['body_lin_vel_w'][:,pelvis];v[:,3:6]=z['body_ang_vel_w'][:,pelvis]
            added=[]
            for j,name in enumerate(names):
                if name in sn:k=sn.index(name);q[:,7+j]=z['joint_pos'][:,k];v[:,6+j]=z['joint_vel'][:,k]
                else:added.append(name)
            assert len(added)==4,added
    data.qpos[:]=q[0];data.qvel[:]=0 # match HumanTracker reset exactly
    mujoco.mj_forward(model,data)
    builder=b['SonicObsBuilder']({'qpos':q,'qvel':v},encoder_dim=int(enc.get_inputs()[0].shape[-1]));builder.reset(data.qpos[3:7].copy())
    last=np.zeros(29,dtype=np.float32);order=b['ISAB_TO_MUJ']
    tracked=['pelvis','torso_link','left_ankle_roll_link','right_ankle_roll_link','left_elbow_link','right_elbow_link','left_wrist_roll_link','right_wrist_roll_link']
    bids=[model.body(n).id for n in tracked];core=[j for j,n in enumerate(names) if 'wrist' not in n]
    poses=[];je=[];ce=[];re=[];be=[];falls=[];strict=[];ye=[];clip=0;max_torque=0.;start=time.time()
    for t in range(len(q)-1):
        quat=data.qpos[3:7].copy()
        gyro=data.sensor('gyro_pelvis').data.copy()
        builder.update_history(gyro,data.qpos[7:][order]-b['DEFAULT_ANGLES_ISAB'],data.qvel[6:][order],last,quat)
        obs=builder.build_encoder_obs(t,quat)
        token=enc.run(None,{enc.get_inputs()[0].name:obs})[0]
        action=dec.run(None,{dec.get_inputs()[0].name:builder.build_decoder_obs(token)})[0][0]
        last=action.copy();target=b['action_to_motor_targets'](action)
        for _ in range(20):
            torque=b['SONIC_KPS'].astype(np.float32)*(target-data.qpos[7:])+b['SONIC_KDS'].astype(np.float32)*(-data.qvel[6:])
            data.ctrl[:]=np.clip(torque,-b['SONIC_TORQUE_LIMIT'].astype(np.float32),b['SONIC_TORQUE_LIMIT'].astype(np.float32))
            clip+=int(np.any(abs(torque)>=b['SONIC_TORQUE_LIMIT']));max_torque=max(max_torque,float(abs(data.ctrl).max()));mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data);rd.qpos[:]=q[t+1];mujoco.mj_forward(model,rd)
        actual=data.qpos.copy();assert np.isfinite(actual).all();poses.append(actual)
        j=float(abs(actual[7:]-q[t+1,7:]).mean());c=float(abs(actual[7:][core]-q[t+1,7:][core]).mean());r=float(np.linalg.norm(actual[:3]-q[t+1,:3]));body=float(np.linalg.norm(data.xpos[bids]-rd.xpos[bids],axis=1).mean());f=bool(actual[2]<.35)
        je.append(j);ce.append(c);re.append(r);be.append(body);falls.append(f);strict.append(j<.3 and r<.15 and body<.15 and not f)
        angles=R.from_quat(np.array([actual[3:7],q[t+1,3:7]])[:,[1,2,3,0]]).as_euler('xyz')[:,2];delta=angles[0]-angles[1];ye.append(float(abs(np.arctan2(np.sin(delta),np.cos(delta)))))
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'rollout.npz',qpos=poses,reference=q[1:],joint_error=je,common_joint_error=ce,root_error=re,body_error=be,falls=falls,strict=strict,yaw_error=ye,fps=50)
    first=lambda s:float((np.flatnonzero(s)[0]+1)*.02) if np.any(s) else None
    out={'tracker':'SONIC default release / HumanTracker adapter','backend_commit':PIN,'backend_sha256':sha(file),'weights_revision':metadata['sha'],
         'weight_sha256':{n:sha(weights/n) for n in ['model_encoder.onnx','model_decoder.onnx']},'model_path':str(xml),'model_sha256':sha(xml),
         'encoder_dim':int(enc.get_inputs()[0].shape[-1]),'decoder_dim':int(dec.get_inputs()[0].shape[-1]),'onnx_provider':'CPUExecutionProvider',
         'frames':len(je),'duration_s':len(je)*.02,'wall_s':time.time()-start,'joint_names':names,'added_neutral_joints':added,'joint_mae_rad':float(np.mean(je)),
         'common_joint_mae_rad':float(np.mean(ce)),'root_error_m':float(np.mean(re)),'body_error_m':float(np.mean(be)),'yaw_error_rad':float(np.mean(ye)),
         'no_fall':not any(falls),'first_fall_s':first(falls),'strict_complete':all(strict),'strict_prefix_s':float(np.logical_and.accumulate(strict).sum()*.02),
         'first_joint_failure_s':first(np.array(je)>=.3),'first_root_failure_s':first(np.array(re)>=.15),'first_body_failure_s':first(np.array(be)>=.15),
         'torque_clip_substep_fraction':clip/(20*len(je)),'max_commanded_torque_nm':max_torque,'cuda_witness_loss':float(loss.detach().cpu()),
         'training_steps':0,'reference_sha256':sha(a.reference) if a.reference else None,'code_sha256':sha(Path(__file__)),
         'caveats':['HumanTracker observation/action backend with its native29DOF model; not official C++ or full benchmark reproduction.',
                    'Reference current frame and future window feed encoder; metrics use post-step next reference frame.',
                    'Reset zero velocity follows backend; oldTWIST reset uses reference velocity. Robot model/gains differ.',
                    'MuJoCo3.2.3 warm environment differs from full HumanTracker dependency recipe; independent standing witness required.',
                    'No HumanScore, pretrained overlap unknown, no training utility or real-robot claim.']}
    (a.output/'metrics.json').write_text(json.dumps(out,indent=2));print('SONIC_PROBE_WITNESS '+json.dumps(out),flush=True)
if __name__=='__main__':main()
