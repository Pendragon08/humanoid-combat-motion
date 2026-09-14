"""Differentiable bone-direction optimization with exact source bone lengths.

No robot joint limits or dynamics are imposed. This is an exploratory human
contact baseline, not a robot-executable motion guarantee.
"""
import numpy as np
import torch
from combat.contact_guidance import contact_objective
from combat.interaction import validate_motion,evaluate_event


def refine_kinematic_contacts(positions,radii,fps,events,edges,steps=200,lr=.025,device='cpu',fixed_actor=0):
    positions,radii=validate_motion(positions,radii,fps)
    for event in events:evaluate_event(positions,radii,fps,event)
    if fixed_actor not in (0,1):raise ValueError('Require an explicitly fixed actor')
    moving=1-fixed_actor;joints=positions.shape[2]
    seen={0}
    for parent,child in edges:
        if parent not in seen or child in seen:raise ValueError('Require rooted topological tree')
        seen.add(child)
    if seen!=set(range(joints)):raise ValueError('Tree must span all joints')
    initial=torch.tensor(positions,dtype=torch.float32,device=device)
    radius=torch.tensor(radii,dtype=torch.float32,device=device)
    parent=torch.tensor([p for p,c in edges],device=device);child=torch.tensor([c for p,c in edges],device=device)
    delta=initial[:,moving,child]-initial[:,moving,parent]
    lengths=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
    if torch.any(lengths<1e-6):raise ValueError('Degenerate source bone')
    directions=(delta/lengths).clone().requires_grad_(True)
    root=initial[:,moving,0].clone().requires_grad_(True)
    opt=torch.optim.Adam([root,directions],lr=lr)
    def decode():
        unit=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True).clamp(min=1e-8)
        body={0:root}
        for i,(p,c) in enumerate(edges):body[c]=body[p]+unit[:,i]*lengths[:,i]
        body=torch.stack([body[j] for j in range(joints)],dim=1)
        return torch.stack([initial[:,0],body],dim=1) if fixed_actor==0 else torch.stack([body,initial[:,1]],dim=1)
    history=[]
    for step in range(steps):
        opt.zero_grad();x=decode()
        # Aim at touching surfaces (0m), inside the evaluator's 3cm tolerance.
        # Optimizing to the acceptance boundary leaves a positive residual under
        # preservation regularization and can never robustly satisfy that bound.
        contact=contact_objective(x,radius,events,tolerance=0.)
        preservation=(x-initial).square().mean()
        velocity=(torch.diff(x-initial,dim=0)*fps).square().mean()
        loss=contact+preservation+.01*velocity
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite FK objective')
        loss.backward();torch.nn.utils.clip_grad_norm_([root,directions],1);opt.step()
        history.append({'step':step,'loss':float(loss.detach()),'contact':float(contact.detach())})
    output=decode().detach().cpu().numpy()
    if not np.isfinite(output).all():raise FloatingPointError('Nonfinite FK output')
    return output,history
