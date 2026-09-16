"""Seeded forward/backward witness, independent subprocess per upstream namespace."""
import argparse
import json
import subprocess
import sys
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['remos', 'intergen'])
    a = p.parse_args()
    if not a.method:
        for method in ['remos', 'intergen']:
            subprocess.run([sys.executable, '-m', 'scripts.smoke_remocap_models', '--method', method], check=True)
        print('REMOCAP_MODELS_WITNESS', flush=True)
        return
    from combat.remocap_models import BodyModel, cosine_alphas, q_sample, sample_reaction
    torch.manual_seed(0)
    model = BodyModel(a.method)
    data = torch.randn(2, 8, 2, 22, 3, device='cuda')
    target = data[:, :, 1] if a.method == 'remos' else data
    t = torch.tensor([7, 123], device='cuda')
    alphas = cosine_alphas(300, 'cuda')
    prediction = model(q_sample(target, t, torch.randn_like(target), alphas), t, data[:, :, 0])
    loss = (prediction - target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    out = sample_reaction(model, data[:, :, 0], alphas, steps=3)
    assert out.shape == (2, 8, 22, 3) and torch.isfinite(out).all()
    print(json.dumps(dict(method=a.method, loss=loss.item(), shape=list(out.shape),
        peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)), flush=True)


if __name__ == '__main__':
    main()
