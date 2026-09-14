"""Offline actor-motion conditioning; does not establish real-time causality."""
import torch
from torch import nn
from torch.nn import functional as F


class ActorMotionAdapter(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.first=nn.Conv1d(262,hidden,5)
        self.second=nn.Conv1d(hidden,hidden,3)
        self.kind=nn.Embedding(3,hidden,padding_idx=0)
        self.window=nn.Linear(2,hidden)
        self.out=nn.Linear(hidden,768)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,text,actor,lengths,kind,window):
        if actor.ndim!=3 or actor.shape[-1]!=262:raise ValueError('Expected normalized actor features')
        valid=torch.arange(actor.shape[1],device=actor.device)[None]<lengths[:,None]
        mask=valid[:,None,:]
        h=F.silu(self.first(F.pad((actor*valid[:,:,None]).transpose(1,2),(4,0))))*mask
        h=F.silu(self.second(F.pad(h,(2,0))))*mask
        pooled=h.sum(-1)/lengths[:,None]
        return text+self.out(pooled+self.kind(kind)+self.window(window))


def masked_defender_mse(pred,target,lengths,start_frames=None):
    if pred.shape!=target.shape or pred.shape[-1]!=524:raise ValueError('Expected paired normalized motion')
    mask=torch.arange(pred.shape[1],device=pred.device)[None]<lengths[:,None]
    if start_frames is not None:mask &= torch.arange(pred.shape[1],device=pred.device)[None]>=start_frames[:,None]
    error=(pred[...,262:]-target[...,262:]).square().mean(-1)
    return (error*mask).sum()/mask.sum().clamp(min=1)


def history_condition(raw_actor,cutoffs,normalizer):
    """Remove unseen frames and forward-difference lookahead before conditioning.

    Caller must also use a history-only world/floor transform. Upstream velocity
    and foot-contact channels at the last observed frame otherwise use t+1.
    """
    value=raw_actor.clone()
    for i,cutoff in enumerate(cutoffs.tolist()):
        if cutoff<2 or cutoff>value.shape[1]:raise ValueError('Invalid history cutoff')
        value[i,cutoff:]=0
        value[i,cutoff-1,66:132]=0
        value[i,cutoff-1,258:262]=0
    return normalizer.forward(value)


class CausalReactionResidual(nn.Module):
    """Explicit defender queries over visible attacker-frame keys and values.

    This constrains information access, not strategic or physical correctness.
    No attacker frame later than query time or the observation cutoff is visible.
    """
    def __init__(self,hidden=128,heads=4):
        super().__init__();self.heads=heads
        self.query=nn.Linear(262,hidden);self.key=nn.Linear(262,hidden)
        self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.norm=nn.LayerNorm(hidden);self.out=nn.Linear(hidden,262)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,prediction,actor,observed):
        b,t,_=prediction.shape
        if actor.shape!=(b,t,262) or observed.min()<1:raise ValueError('Invalid attacker history')
        visible=torch.arange(t,device=actor.device)[None,:]<observed[:,None]
        key=self.key(actor*visible[:,:,None]);query=self.query(prediction[...,262:])
        # A query at time u may use k<=u, and never k>=observed.
        causal=torch.arange(t,device=actor.device)[None,:]>torch.arange(t,device=actor.device)[:,None]
        h,_=self.attention(query,key,key,key_padding_mask=~visible,attn_mask=causal,need_weights=False)
        residual=self.out(self.norm(h+query))
        return torch.cat([prediction[...,:262],prediction[...,262:]+residual],dim=-1)
