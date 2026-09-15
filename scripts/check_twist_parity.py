"""Compare offline adapter against actual upstream run() with GUI/Redis stubs.

The upstream observation, history, action and torque expressions execute intact.
Only I/O, real-time sleeping and starting quaternion are explicitly controlled.
"""
import ast
import contextlib
import io
import json
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from combat.twist_tracking import ROOT, UPSTREAM, XML, Controller, scene, official_contract

class Recorder:
    def __init__(self, policy):self.policy=policy;self.inputs=[]
    def __call__(self,x):self.inputs.append(x.detach().cpu().numpy().copy());return self.policy(x)

def main():
    torch.manual_seed(123);torch.set_num_threads(4)
    cfg,scope=official_contract();default=scope['DEFAULT_MIMIC_OBS']['g1']
    viewer=SimpleNamespace(opt=SimpleNamespace(flags={}),cam=SimpleNamespace(distance=2.,lookat=np.zeros(3)),
        user_scn=SimpleNamespace(ngeom=0),sync=lambda:None,close=lambda:None)
    redis=SimpleNamespace(Redis=lambda **kw:SimpleNamespace(set=lambda *x:None,get=lambda *x:json.dumps(default.tolist())))
    scope.update(redis=redis,mujoco=mujoco,torch=torch,json=json,print=print,
        mjv=SimpleNamespace(launch_passive=lambda *x,**kw:viewer),
        time=SimpleNamespace(time=lambda:0.,sleep=lambda x:None),tqdm=lambda x,**kw:x,deque=deque)
    path=UPSTREAM/'deploy_real/server_low_level_g1_sim.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
    scope['draw_root_velocity']=lambda *args:0
    with contextlib.redirect_stdout(io.StringIO()):
        original=scope['RealTimePolicyController'](str(XML),str(UPSTREAM/'assets/twist_general_motion_tracker.pt'),device='cuda:0')
    initial=np.r_[0,0,default[0],1,0,0,0,default[8:]]
    original.mujoco_default_dof_pos=initial;original.sim_duration=.2
    original.policy=Recorder(original.policy.eval())
    with contextlib.redirect_stdout(io.StringIO()):original.run()
    model,_=scene(1);data=mujoco.MjData(model);data.qpos[:]=initial;mujoco.mj_forward(model,data)
    recorder=Recorder(original.policy.policy);ctrl=Controller(model,0,recorder,'cuda:0')
    for step in range(200):
        if step%20==0:ctrl.target(data,default)
        ctrl.torque(data);mujoco.mj_step(model,data)
    official=np.asarray(original.policy.inputs);adapted=np.asarray(recorder.inputs)
    assert official.shape==(10,1,1155),official.shape
    obs_error=float(np.max(abs(official-adapted)))
    pose_error=float(np.max(abs(original.data.qpos-data.qpos)))
    print('PARITY_DEBUG',np.max(abs(official-adapted),axis=(1,2)).tolist(),pose_error)
    assert obs_error<1e-5 and pose_error<1e-5,(obs_error,pose_error)
    result={'original_run_executed':True,'policy_calls':10,'physics_steps':200,
        'max_observation_difference':obs_error,'max_qpos_difference':pose_error,
        'modifications':'GUI/Redis/time I/O stubbed; initial quaternion explicitly wxyz identity; physics/controller logic unchanged'}
    (ROOT/'records/twist/parity.json').write_text(json.dumps(result,indent=2))
    print('TWIST_UPSTREAM_PARITY',json.dumps(result))

if __name__=='__main__':main()
