"""Small trainable components; untrained weights are never used as physics labels."""
import torch
from torch import nn


class CombatConditionAdapter(nn.Module):
    """Structured categorical conditions -> residual CLIP condition (768 dims).

Unknown/unannotated values must use index zero, not inferred fine-grained labels.
The original TIMotion interaction blocks still couple both actors; this adapter
alone does NOT establish attacker-to-defender causal correctness.
"""
    def __init__(self,cardinalities,dim=768):
        super().__init__()
        self.names=tuple(sorted(cardinalities))
        self.embeddings=nn.ModuleDict({name:nn.Embedding(cardinalities[name],dim,padding_idx=0) for name in self.names})
        self.out=nn.Linear(dim,dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self,text_condition,conditions):
        if set(conditions)!=set(self.names):
            raise ValueError('Condition fields must match declared vocabulary')
        tokens=torch.stack([self.embeddings[name](conditions[name]) for name in self.names],dim=0).sum(0)
        return text_condition+self.out(tokens)


class DualRobotPhysicsCritic(nn.Module):
    """Four outputs: tracking error, fall/contact/collision logits.

Input is time-series paired robot features; lengths exclude padding from pooling.
Training requires independently recorded learned-controller simulation labels.
"""
    def __init__(self,input_dim,hidden=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(input_dim,hidden,5,padding=2),nn.SiLU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.SiLU())
        self.head=nn.Linear(hidden,4)

    def forward(self,x,lengths):
        if x.ndim!=3 or lengths.shape!=(len(x),) or (lengths<1).any() or (lengths>x.shape[1]).any():
            raise ValueError('Invalid sequence lengths')
        mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        # Zero padding before convolution so padded values cannot leak inward.
        h=self.encoder((x*mask[:,:,None]).transpose(1,2))
        pooled=(h*mask[:,None,:]).sum(-1)/lengths[:,None]
        y=self.head(pooled)
        return {'tracking_error_m':torch.nn.functional.softplus(y[:,0]),'fall_logit':y[:,1],
                'contact_logit':y[:,2],'collision_logit':y[:,3]}


def critic_loss(pred,target):
    loss=(pred['tracking_error_m']-target['tracking_error_m']).square().mean()
    for name,label in [('fall','fall'),('contact','contact_success'),('collision','invalid_collision')]:
        loss=loss+nn.functional.binary_cross_entropy_with_logits(pred[name+'_logit'],target[label].float())
    return loss


def rank_candidates(pred):
    """Lower is better; coefficients frozen defaults, calibrate on validation."""
    score=pred['tracking_error_m']+torch.sigmoid(pred['fall_logit'])+torch.sigmoid(pred['collision_logit'])+1-torch.sigmoid(pred['contact_logit'])
    return torch.argsort(score),score
