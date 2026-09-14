"""Small temporal x0-prediction diffusion for robot-reference augmentation.

Inspired by DDPM/MDM's denoising principle; not an MDM/RobotMDM reproduction.
Input condition: seed root trajectory + coarse martial-arts style; output: 29 joint angles.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.film = nn.Linear(width, 2*width)
        self.output = nn.Conv1d(width, width, 1)

    def forward(self, x, emb):
        scale, shift = self.film(emb).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm(x)*(1+scale)+shift
        return x + self.output(F.silu(self.conv(F.silu(h))))*.5

class MotionDenoiser(nn.Module):
    def __init__(self, width=128, steps=100):
        super().__init__()
        self.steps = steps
        self.width = width
        self.input = nn.Conv1d(36, width, 1)
        self.time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.style = nn.Embedding(3, width)
        self.blocks = nn.ModuleList([TemporalBlock(width, d) for d in [1, 2, 4, 8, 16, 1]])
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, 29, 1))

    def forward(self, noisy, t, root, style):
        freq = torch.exp(torch.arange(32, device=t.device)*(-math.log(10000)/31))
        arg = t[:, None].float()*freq[None, :]
        emb = self.time(torch.cat([arg.sin(), arg.cos()], dim=-1)) + self.style(style)
        h = self.input(torch.cat([noisy, root], dim=-1).transpose(1, 2))
        for block in self.blocks:
            h = block(h, emb)
        return self.output(h).transpose(1, 2)

def schedule(steps, device):
    t = torch.arange(steps+1, device=device)/steps
    alpha = torch.cos((t+.008)/1.008*math.pi/2)**2
    alpha = alpha/alpha[0]
    beta = (1-alpha[1:]/alpha[:-1]).clamp(.0001, .999)
    return beta, torch.cumprod(1-beta, 0)

@torch.no_grad()
def augment(model, seed_joints, root, style, noise_step=40, random_seed=0):
    """Motion-to-motion denoising, with a fixed seed root path. Not text-to-motion."""
    model.eval()
    beta, abar = schedule(model.steps, seed_joints.device)
    rng = torch.Generator(device=seed_joints.device).manual_seed(random_seed)
    noise = torch.randn(seed_joints.shape, generator=rng, device=seed_joints.device)
    x = abar[noise_step].sqrt()*seed_joints + (1-abar[noise_step]).sqrt()*noise
    for i in range(noise_step, -1, -1):
        t = torch.full((len(x),), i, dtype=torch.long, device=x.device)
        x0 = model(x, t, root, style).clamp(-5, 5)
        if i == 0:
            x = x0
        else:
            previous = abar[i-1]
            current = abar[i]
            mean = previous.sqrt()*beta[i]/(1-current)*x0 + (1-beta[i]).sqrt()*(1-previous)/(1-current)*x
            variance = beta[i]*(1-previous)/(1-current)
            x = mean + variance.sqrt()*torch.randn(x.shape, generator=rng, device=x.device)
    return x
