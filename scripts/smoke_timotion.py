"""Seeded real CUDA forward/backward on upstream interaction block, no fake baseline."""
import json
import torch
from models.blocks import TIMotionTransformerBlock

torch.manual_seed(0)
block=TIMotionTransformerBlock(latent_dim=64,num_heads=4,ff_size=128,dropout=0,LPA=False).cuda()
a=torch.randn(2,12,64,device='cuda',requires_grad=True)
b=torch.randn(2,12,64,device='cuda',requires_grad=True)
emb=torch.randn(2,64,device='cuda')
mask=torch.zeros(2,12,dtype=torch.bool,device='cuda')
y,z=block(a,b,emb,mask)
loss=y.square().mean()+z.square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
torch.cuda.synchronize()
print('TIMOTION_CUDA_WITNESS '+json.dumps({'shape':list(y.shape),'loss':loss.item(),'device':torch.cuda.get_device_name(), 'purpose':'kernel_only_not_pretrained_reproduction'}))
