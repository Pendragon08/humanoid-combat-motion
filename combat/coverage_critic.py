"""Controller-conditional time-to-tracking-failure prediction, not feasibility."""
import numpy as np
import torch
from torch import nn


def reference_features(qpos, fps):
    q=np.array(qpos,dtype=np.float32,copy=True)
    if q.ndim!=3 or q.shape[1:]!=(2,36) or len(q)<3 or fps<=0 or not np.isfinite(q).all():
        raise ValueError('Expected finite [T,2,36] G1 references')
    # Remove only shared horizontal translation: floor-relative height matters.
    q[:,:,:2]-=q[0,0,:2].copy()
    norms=np.linalg.norm(q[:,:,3:7],axis=-1)
    if np.any(np.abs(norms-1)>1e-3):raise ValueError('Non-unit root quaternion')
    # q and -q are the same orientation, including framewise sign flips.
    for actor in range(2):
        if q[0,actor,3]<0:q[0,actor,3:7]*=-1
        for t in range(1,len(q)):
            if np.dot(q[t,actor,3:7],q[t-1,actor,3:7])<0:q[t,actor,3:7]*=-1
    velocity=np.gradient(np.concatenate([q[:,:,:3],q[:,:,7:]],axis=-1),1/fps,axis=0)
    return np.concatenate([q,velocity],axis=-1).reshape(len(q),-1)


def prompt_split(index):
    if index not in range(4):raise ValueError('Exactly four historical prompt groups')
    return ('train','train','val','test')[index]


class ControllerCoverageCritic(nn.Module):
    def __init__(self,input_dim=136):
        super().__init__()
        self.layers=nn.ModuleList([nn.Conv1d(input_dim,32,5,padding=2),nn.SiLU(),nn.Conv1d(32,32,3,padding=1),nn.SiLU()])
        self.head=nn.Linear(32,1)

    def forward(self,x,lengths):
        mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        h=x.transpose(1,2)*mask[:,None,:]
        for layer in self.layers:h=layer(h)*mask[:,None,:]
        return self.head(h.sum(-1)/lengths[:,None]).squeeze(-1)
