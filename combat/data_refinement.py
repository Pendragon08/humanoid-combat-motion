"""Controlled contact/temporal data ablation, not robot dynamics certification."""
import numpy as np
import torch
from combat.contact_guidance import contact_objective
from combat.interaction import evaluate_event, validate_motion

CHAINS = [[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
EDGES = [edge for chain in CHAINS for edge in zip(chain[:-1],chain[1:])]


def refine_data(positions, radii, fps, event, contact_weight, acceleration_weight,
                steps=150, lr=0.01, device='cpu'):
    positions,radii=validate_motion(positions,radii,fps)
    if event is not None:
        evaluate_event(positions,radii,fps,event)
    elif contact_weight != 0:
        raise ValueError('Contact refinement requires an independently supplied plan')
    if positions.shape[2]!=22 or (event is not None and (event.a_actor!=0 or event.b_actor!=1)):
        raise ValueError('Explicit attacker=0, defender=1 and 22-joint representation required')
    if min(contact_weight,acceleration_weight)<0 or steps<=0:
        raise ValueError('Invalid optimization settings')
    initial=torch.tensor(positions,dtype=torch.float32,device=device)
    radius=torch.tensor(radii,dtype=torch.float32,device=device)
    parent,child=np.asarray(EDGES).T
    delta=initial[:,1,child]-initial[:,1,parent]
    lengths=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
    if torch.any(lengths<1e-6): raise ValueError('Degenerate bones')
    directions=(delta/lengths).clone().requires_grad_(True)
    root=initial[:,1,0].clone().requires_grad_(True)
    optimizer=torch.optim.Adam([root,directions],lr=lr)

    def decode():
        unit=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True).clamp(min=1e-8)
        body={0:root}
        for i,(p,c) in enumerate(EDGES):body[c]=body[p]+unit[:,i]*lengths[:,i]
        return torch.stack([initial[:,0],torch.stack([body[j] for j in range(22)],dim=1)],dim=1)

    history=[]
    for step in range(steps):
        optimizer.zero_grad(); x=decode()
        contact=contact_objective(x,radius,[event],tolerance=0.) if event is not None else x.new_zeros(())
        preserve=(x-initial).square().mean()
        change_velocity=(torch.diff(x-initial,dim=0)*fps).square().mean()
        acceleration=(torch.diff(x[:,1],n=2,dim=0)*fps**2).square().mean()
        loss=contact_weight*contact+preserve+.01*change_velocity+acceleration_weight*acceleration
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
        loss.backward();torch.nn.utils.clip_grad_norm_([root,directions],1);optimizer.step()
        if step==0 or step==steps-1:
            history.append({'step':step,'loss':float(loss.detach()),'contact':float(contact.detach()),
                            'acceleration_mse':float(acceleration.detach())})
    return decode().detach().cpu().numpy(),history


def quality_metrics(x, original, radii, fps, event):
    x,radii=validate_motion(x,radii,fps)
    original,_=validate_motion(original,radii,fps)
    parent,child=np.asarray(EDGES).T
    bones=lambda z:np.linalg.norm(z[:,:,child]-z[:,:,parent],axis=-1)
    velocity=np.diff(x[:,1],axis=0)*fps
    source_velocity=np.diff(original[:,1],axis=0)*fps
    acceleration=np.diff(velocity,axis=0)*fps
    speed=np.linalg.norm(velocity,axis=-1)
    source_path=float(np.linalg.norm(source_velocity,axis=-1).sum())
    return {'proxy_contact':evaluate_event(x,radii,fps,event),
            'max_bone_length_change_m':float(np.abs(bones(x)-bones(original)).max()),
            'max_attacker_change_m':float(np.abs(x[:,0]-original[:,0]).max()),
            'defender_displacement_rms_m':float(np.sqrt(np.mean((x[:,1]-original[:,1])**2))),
            'defender_speed_p95_m_s':float(np.percentile(speed,95)),
            'defender_speed_max_m_s':float(speed.max()),
            'defender_acceleration_rms_m_s2':float(np.sqrt(np.mean(acceleration**2))),
            'defender_path_length_ratio':float(speed.sum()/source_path) if source_path>1e-10 else None,
            'robot_dynamics_verified':False,'human_semantics_verified':False,
            'contact_truth':'same supplied weak geometric condition; not independent human labels'}
