"""Lightweight differentiable contact refinement on paired joint positions.

This is a kinematic postprocessor, not a physical controller. Outputs must pass
retargeting and learned tracking before they become accepted robot references.
"""
import torch
from combat.interaction import validate_motion, evaluate_event


def contact_objective(x, radii, events, tolerance=.03, penetration_tolerance=.02, clearance=.08):
    terms = []
    for event in events:
        a=x[event.start:event.end+1,event.a_actor,event.a_joint]
        b=x[event.start:event.end+1,event.b_actor,event.b_joint]
        gap=torch.linalg.vector_norm(a-b,dim=-1)-radii[event.a_joint]-radii[event.b_joint]
        if event.kind=='avoid':
            if not event.threatened:
                continue
            loss=torch.relu(clearance-gap).square().mean()
        else:
            # At least one contact in the window; no incentive to penetrate.
            loss=torch.relu(gap.min()-tolerance).square()
            loss=loss+torch.relu(-penetration_tolerance-gap).square().mean()
            if event.kind=='block':
                if event.protected_joint is None:
                    raise ValueError('Block needs protected_joint')
                target=x[event.start:event.end+1,event.b_actor,event.protected_joint]
                protected_gap=torch.linalg.vector_norm(a-target,dim=-1)-radii[event.a_joint]-radii[event.protected_joint]
                loss=loss+torch.relu(clearance-protected_gap).square().mean()
        terms.append(event.confidence*loss)
    return torch.stack(terms).mean() if terms else x.sum()*0


def refine_contacts(positions,radii,fps,events,edges,steps=80,lr=.01,device='cpu',fixed_actor=None,hard_bone_projection=False):
    positions,radii=validate_motion(positions,radii,fps)
    if fixed_actor not in (None,0,1):raise ValueError('Invalid fixed actor')
    # Reuse strict event validation before gradient optimization.
    for event in events:
        evaluate_event(positions,radii,fps,event)
    initial=torch.tensor(positions,dtype=torch.float32,device=device)
    x=initial.clone().requires_grad_(True)
    radius=torch.tensor(radii,dtype=x.dtype,device=device)
    if not edges:
        raise ValueError('Explicit skeleton edges required to regularize bone lengths')
    parent=torch.tensor([a for a,b in edges],device=device)
    child=torch.tensor([b for a,b in edges],device=device)
    reference_lengths=torch.linalg.vector_norm(initial[:,:,parent]-initial[:,:,child],dim=-1)
    optimizer=torch.optim.Adam([x],lr=lr)
    history=[]
    for i in range(steps):
        optimizer.zero_grad()
        contact=contact_objective(x,radius,events)
        preservation=(x-initial).square().mean()
        bone=(torch.linalg.vector_norm(x[:,:,parent]-x[:,:,child],dim=-1)-reference_lengths).square().mean()
        # Preserve velocity changes instead of flattening a fast strike.
        velocity=(torch.diff(x-initial,dim=0)*fps).square().mean()
        loss=contact+preservation+10*bone+.01*velocity
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite refinement objective')
        loss.backward()
        if fixed_actor is not None:x.grad[:,fixed_actor]=0
        torch.nn.utils.clip_grad_norm_([x],1)
        optimizer.step()
        if fixed_actor is not None:
            with torch.no_grad():x[:,fixed_actor]=initial[:,fixed_actor]
        if hard_bone_projection:
            # Project in parent-before-child order; keep source per-frame bone
            # lengths, without claiming those source lengths are noise-free.
            with torch.no_grad():
                for actor in (0,1):
                    if actor==fixed_actor:continue
                    for e,(p,c) in enumerate(edges):
                        delta=x[:,actor,c]-x[:,actor,p]
                        norm=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
                        original=initial[:,actor,c]-initial[:,actor,p]
                        direction=torch.where(norm>1e-8,delta/norm.clamp(min=1e-8),
                            original/reference_lengths[:,actor,e,None].clamp(min=1e-8))
                        x[:,actor,c]=x[:,actor,p]+direction*reference_lengths[:,actor,e,None]
        history.append({'step':i,'loss':float(loss.detach()),'contact':float(contact.detach()),'bone':float(bone.detach())})
    return x.detach().cpu().numpy(),history
