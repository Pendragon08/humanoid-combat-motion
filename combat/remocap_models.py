"""Source-preserving upstream body adapters; not original benchmark reproductions."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PINS = {'ReMoS': 'd84c036d1a2279575325cde72f7f66a7a00b438a',
        'InterGen': 'b1750df4a859443b8a69329f759ce3a7503ae179'}


def check_source(name):
    path = ROOT / 'third_party' / name
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PINS[name]:
        raise ValueError('Upstream source version mismatch: ' + name)
    if subprocess.check_output(['git', '-C', str(path), 'diff', '--name-only'], text=True).strip():
        raise ValueError('Upstream source has unrecorded modifications')
    return path


class BodyModel(nn.Module):
    def __init__(self, method, frames=40, device='cuda'):
        super().__init__()
        self.method = method
        if method == 'remos':
            path = check_source('ReMoS') / 'src/Lindyhop/models/MotionDiffuse_body.py'
            spec = importlib.util.spec_from_file_location('remos_upstream_body', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.net = module.DiffusionTransformer(device=device, num_jts=22, num_frames=frames,
                input_feats=3, latent_dim=256, ff_size=512, num_layers=6, num_heads=4)
        elif method == 'intergen':
            source = check_source('InterGen')
            # Resolve InterGen's absolute imports before the warm TIMotion paths.
            if 'models' in sys.modules:
                raise RuntimeError('Run InterGen in its own process; models namespace already loaded')
            sys.path.insert(0, str(source))
            from models.nets import InterDenoiser
            self.net = InterDenoiser(input_feats=66, latent_dim=1024, ff_size=2048,
                num_layers=8, num_heads=8, num_frames=frames, dropout=0.1)
        else:
            raise ValueError(method)
        self.to(device)

    def forward(self, noisy, times, actor=None):
        if self.method == 'remos':
            return self.net(noisy, times, motion1=actor)[0]
        shape = noisy.shape
        return self.net(noisy.reshape(shape[0], shape[1], 132), times,
            cond=torch.zeros(shape[0], 768, device=noisy.device)).reshape(shape)


def cosine_alphas(steps, device):
    t = torch.arange(steps + 1, dtype=torch.float64, device=device) / steps
    abar = torch.cos((t + .008) / 1.008 * torch.pi / 2).square()
    beta = (1 - abar[1:] / abar[:-1]).clamp(max=.999)
    return (1 - beta).cumprod(0).float()


def q_sample(clean, t, noise, alphas):
    a = alphas[t].reshape((-1,) + (1,) * (clean.ndim - 1))
    return a.sqrt() * clean + (1 - a).sqrt() * noise


@torch.no_grad()
def sample_reaction(model, actor, alphas, steps=50, seed=0):
    gen = torch.Generator(device=actor.device).manual_seed(seed)
    shape = actor.shape if model.method == 'remos' else (actor.shape[0], actor.shape[1], 2, 22, 3)
    x = torch.randn(shape, generator=gen, device=actor.device)
    actor_noise = torch.randn(actor.shape, generator=gen, device=actor.device)
    timeline = torch.linspace(len(alphas) - 1, 0, steps, device=actor.device).long()
    for i, t in enumerate(timeline):
        times = torch.full((len(actor),), int(t), device=actor.device, dtype=torch.long)
        if model.method == 'intergen':
            x[:, :, 0] = q_sample(actor, times, actor_noise, alphas)
        prediction = model(x, times, actor)
        eps = (x - alphas[t].sqrt() * prediction) / (1 - alphas[t]).sqrt().clamp_min(1e-6)
        if i + 1 == len(timeline):
            x = prediction
        else:
            a_next = alphas[timeline[i + 1]]
            x = a_next.sqrt() * prediction + (1 - a_next).sqrt() * eps
    return x if model.method == 'remos' else x[:, :, 1]
