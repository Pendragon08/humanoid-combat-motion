"""Seeded CUDA pilot training with source-disjoint validation and saved raw logs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from combat.diffusion import MotionDenoiser, schedule
from combat.robot import ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data/windows.npz", allow_pickle=False)
    device = "cuda"
    mean = torch.as_tensor(z["mean"], device=device)
    std = torch.as_tensor(z["std"], device=device)
    train = torch.as_tensor(z["train"], device=device)
    validation = torch.as_tensor(z["validation"], device=device)
    train_style = torch.as_tensor(z["train_style"], device=device)
    val_style = torch.as_tensor(z["validation_style"], device=device)
    model = MotionDenoiser().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    _, abar = schedule(model.steps, device)
    config = {**vars(args), "model": "6-block temporal convolution x0-denoiser", "parameters": sum(p.numel() for p in model.parameters()), "conditioning": "root trajectory + martial-arts style", "diffusion_steps": model.steps, "inference": "seed-conditioned motion augmentation", "loss": "normalized joint x0 MSE", "gpu": torch.cuda.get_device_name(), "window_sha256": hashlib.sha256((ROOT/"data/windows.npz").read_bytes()).hexdigest(), "torch": torch.__version__, "claim_scope": "reference generation pilot; no policy learning or paired combat"}
    (out/"config.json").write_text(json.dumps(config, indent=2))
    code_files = sorted(list((ROOT/'combat').glob('*.py')) + list((ROOT/'scripts').glob('*.py')) + list((ROOT/'scripts').glob('*.sh')))
    (out/'code_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}, indent=2))
    # Fixed validation noise and timesteps, no test data used for checkpoint selection.
    vrng = torch.Generator(device=device).manual_seed(901)
    vx = (validation[:, :, 7:]-mean)/std
    vt = torch.randint(0, model.steps, (len(vx),), device=device, generator=vrng)
    va = abar[vt, None, None]
    vn = torch.randn(vx.shape, device=device, generator=vrng)
    vy = va.sqrt()*vx+(1-va).sqrt()*vn
    best = float("inf")
    started = time.time()
    logfile = (out/"loss.jsonl").open("w")
    for step in range(args.steps+1):
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_loss = np.mean([float(F.mse_loss(model(vy[i:i+64], vt[i:i+64], validation[i:i+64, :, :7], val_style[i:i+64]), vx[i:i+64])) for i in range(0, len(vx), 64)])
            row = {"step": step, "validation_x0_mse": float(val_loss), "elapsed_s": time.time()-started, "peak_gpu_gib": torch.cuda.max_memory_allocated()/1024**3}
            if step:
                row["train_x0_mse"] = float(loss)
            logfile.write(json.dumps(row)+"\n")
            logfile.flush()
            print(json.dumps(row), flush=True)
            if val_loss < best:
                best = val_loss
                torch.save({"model": model.state_dict(), "config": config, "mean": mean.cpu(), "std": std.cpu(), "step": step, "validation_x0_mse": best}, out/"best.pt")
        if step == args.steps:
            break
        model.train()
        idx = torch.randint(len(train), (args.batch_size,), device=device)
        x = (train[idx, :, 7:]-mean)/std
        t = torch.randint(0, model.steps, (len(x),), device=device)
        a = abar[t, None, None]
        y = a.sqrt()*x+(1-a).sqrt()*torch.randn_like(x)
        loss = F.mse_loss(model(y, t, train[idx, :, :7], train_style[idx]), x)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
    logfile.close()
    torch.save({"model": model.state_dict(), "config": config, "mean": mean.cpu(), "std": std.cpu(), "step": args.steps}, out/"last.pt")
    (out/"TRAINING_COMPLETE.json").write_text(json.dumps({"steps": args.steps, "seconds": time.time()-started, "best_validation_x0_mse": best, "peak_gpu_gib": torch.cuda.max_memory_allocated()/1024**3}, indent=2))
    print("TRAINING_COMPLETE", args.run)

if __name__ == "__main__":
    main()
