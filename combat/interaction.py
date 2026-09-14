"""Paired contact geometry in metres/seconds, independent of simulator or generator.

Sphere proxies are diagnostic approximations, not measured force or mesh contact.
Both actors MUST share a world frame. Avoid events require a labelled threat;
large separation alone cannot establish a successful dodge.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContactEvent:
    kind: str  # hit, block, avoid
    start: int
    end: int  # inclusive
    a_joint: int
    b_joint: int
    a_actor: int = 0
    b_actor: int = 1
    protected_joint: int | None = None
    threatened: bool = False
    confidence: float = 1.0
    provenance: str = 'manual'


def validate_motion(x, radii, fps):
    x, radii = np.asarray(x, dtype=float), np.asarray(radii, dtype=float)
    if x.ndim != 4 or x.shape[1] != 2 or x.shape[-1] != 3 or len(x) < 2:
        raise ValueError('Expected [T>=2, 2, J, 3] in a shared world frame')
    if radii.shape != (x.shape[2],) or not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError('One finite nonnegative radius per joint is required')
    if not np.isfinite(x).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Nonfinite coordinates or invalid fps')
    return x, radii


def surface_gap(x, radii, actor_a, joint_a, actor_b, joint_b):
    return np.linalg.norm(x[:,actor_a,joint_a] - x[:,actor_b,joint_b], axis=-1) - radii[joint_a] - radii[joint_b]


def swept_surface_gap(x, radii, actor_a, joint_a, actor_b, joint_b):
    """Minimum inter-frame gap with linear relative motion; catches tunnelling."""
    r = x[:,actor_a,joint_a] - x[:,actor_b,joint_b]
    v = np.diff(r, axis=0)
    alpha = np.clip(-np.sum(r[:-1]*v,axis=-1)/np.maximum(np.sum(v*v,axis=-1),1e-12),0,1)
    return np.linalg.norm(r[:-1]+alpha[:,None]*v,axis=-1)-radii[joint_a]-radii[joint_b]


def evaluate_event(x, radii, fps, event, tolerance=0.03, penetration_tolerance=0.02, clearance=0.08):
    x, radii = validate_motion(x,radii,fps)
    if not (0 <= event.start < event.end < len(x)):
        raise ValueError('Event window must contain at least two frames')
    if event.kind not in ('hit','block','avoid') or event.a_actor == event.b_actor:
        raise ValueError('Invalid event type or actor pair')
    if event.a_actor not in (0,1) or event.b_actor not in (0,1):
        raise ValueError('Actor must be 0 or 1')
    indices = [event.a_joint,event.b_joint] + ([] if event.protected_joint is None else [event.protected_joint])
    if any(j < 0 or j >= x.shape[2] for j in indices):
        raise ValueError('Joint outside representation')
    if not 0 <= event.confidence <= 1 or min(tolerance,penetration_tolerance,clearance) < 0:
        raise ValueError('Invalid confidence or threshold')
    gap = surface_gap(x,radii,event.a_actor,event.a_joint,event.b_actor,event.b_joint)
    window = gap[event.start:event.end+1]
    swept = swept_surface_gap(x,radii,event.a_actor,event.a_joint,event.b_actor,event.b_joint)[event.start:event.end]
    contacts = np.flatnonzero((window <= tolerance) & (window >= -penetration_tolerance))
    first = int(contacts[0]+event.start) if len(contacts) else None
    penetrates = bool(min(window.min(),swept.min()) < -penetration_tolerance)
    success = first is not None and not penetrates
    protected_hit = None
    if event.kind == 'block':
        if event.protected_joint is None:
            raise ValueError('Block needs the protected body part')
        protected = surface_gap(x,radii,event.a_actor,event.a_joint,event.b_actor,event.protected_joint)
        hits = np.flatnonzero(protected[event.start:event.end+1] <= tolerance)
        protected_hit = int(hits[0]+event.start) if len(hits) else None
        # A block must precede impact; impact within the event is still failure.
        protected_swept = swept_surface_gap(x,radii,event.a_actor,event.a_joint,event.b_actor,event.protected_joint)[event.start:event.end]
        success = success and protected_hit is None and bool(protected_swept.min() > tolerance)
    elif event.kind == 'avoid':
        success = event.threatened and bool(min(window.min(),swept.min()) >= clearance)
    return {
        'kind':event.kind, 'success':bool(success), 'min_surface_gap_m':float(window.min()),
        'min_swept_surface_gap_m':float(swept.min()), 'penetration':penetrates,
        'first_contact_frame':first, 'first_contact_time_s':None if first is None else first/fps,
        'protected_hit_frame':protected_hit, 'confidence':event.confidence,
        'provenance':event.provenance, 'geometry':'sphere_proxy_linear_sweep',
        'force_verified':False,
    }


def paired_features(x, fps):
    """Translation-invariant paired features; preserve relative actor placement."""
    x = np.asarray(x,dtype=np.float32)
    if x.ndim != 4 or x.shape[1] != 2 or x.shape[-1] != 3 or len(x)<2 or fps<=0 or not np.isfinite(x).all():
        raise ValueError('Invalid paired motion')
    centered = x-x[0:1,0:1,0:1,:]
    velocity = np.concatenate([np.zeros_like(x[:1]),np.diff(x,axis=0)*fps])
    return np.concatenate([centered.reshape(len(x),-1),velocity.reshape(len(x),-1)],axis=-1)
