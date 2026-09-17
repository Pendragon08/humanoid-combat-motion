"""Real Gym GPU physics plus seeded gradient witness; no learned-skill claim."""
import argparse
import json
from pathlib import Path
from isaacgym import gymapi, gymtorch
import torch

p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
gym=gymapi.acquire_gym();params=gymapi.SimParams();params.dt=1/60
params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
params.use_gpu_pipeline=True;params.physx.use_gpu=True
sim=gym.create_sim(a.gpu,-1,gymapi.SIM_PHYSX,params);assert sim is not None
try:
    plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);gym.add_ground(sim,plane)
    box=gym.create_box(sim,.2,.2,.2,gymapi.AssetOptions())
    for i in range(4):
        env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),2)
        pose=gymapi.Transform();pose.p.z=1
        gym.create_actor(env,box,pose,'box',i,0)
    gym.prepare_sim(sim)
    state=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    for _ in range(120):gym.simulate(sim);gym.fetch_results(sim,True)
    gym.refresh_actor_root_state_tensor(sim)
    assert state.shape==(4,13) and torch.isfinite(state).all()
    assert torch.all((state[:,2]>.05)&(state[:,2]<.2))
    torch.manual_seed(1);x=torch.randn(8,8,device='cuda:'+str(a.gpu),requires_grad=True)
    loss=(x@x.T).square().mean();loss.backward();assert torch.isfinite(x.grad).all()
    result=dict(physics_steps=120,actors=4,root_height=state[:,2].cpu().tolist(),gradient_loss=loss.item(),
                torch=torch.__version__,device=torch.cuda.get_device_name(a.gpu),scope='Physics and kernel only; not policy or data validity')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print('PBHC_GYM_PHYSICS_WITNESS',json.dumps(result),flush=True)
finally:gym.destroy_sim(sim)
