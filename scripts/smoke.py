"""Seeded CUDA computation, actual physics stepping and offscreen rendering."""
import json
from pathlib import Path
import importlib.metadata as md
import numpy as np
import torch
import mujoco

torch.manual_seed(20260914)
a = torch.randn(64, 64, device="cuda")
b = a @ a.T
torch.cuda.synchronize()
assert torch.isfinite(b).all() and b.shape == (64, 64)
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
d = mujoco.MjData(m)
for _ in range(50):
    mujoco.mj_step(m, d)
assert d.qpos[2] < 1
with mujoco.Renderer(m, height=128, width=128) as r:
    r.update_scene(d)
    frame = r.render()
assert frame.std() > 1
record = {"gpu": torch.cuda.get_device_name(), "matrix_sum": float(b.sum()), "render_std": float(frame.std()), "versions": {k: md.version(k) for k in ["torch", "numpy", "scipy", "mujoco", "osqp"]}}
Path("audit/kernel_witness.json").write_text(json.dumps(record, indent=2))
print("KERNEL_WITNESS CUDA MUJOCO_RENDER", json.dumps(record))
