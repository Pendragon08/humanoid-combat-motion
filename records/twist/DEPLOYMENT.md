# TWIST bounded pretrained-controller deployment

Pinned source: https://github.com/YanjieZe/TWIST/tree/42d8c134739eee51f28d7cc0ff72a86728afb8dc

Remote host: ouyanghao@10.147.17.32; key C:\Users\Pende\.ssh\id_ed25519.
Project: /data/code/ouyanghao/oyh/humanoid-combat-motion.
Warm reuse of existing .venv, no package installation. Declarative spec:
env-twist-inference.json. Torch 2.5.1+cu121; MuJoCo 3.3.7; NumPy 1.26.4.
Official source/assets/checkpoint fetched using scripts/fetch_twist.py; hashes
are in records/twist/provenance.json (about 59 MB, no training dataset downloaded).

## Independent seeded physical witness

Execute verbatim in local PowerShell (unique output for this validation):

```powershell
ssh -i C:\Users\Pende\.ssh\id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run.sh scripts/evaluate_twist.py --standing --seconds 5 --output runs/twist_independent_witness_v2 --device cuda:0"
```

Expected: exit 0, TWIST_PHYSICS_WITNESS sentinel, 250 frames / 5 seconds,
strict_complete [true], no_fall_complete [true], nonzero joint movement/torque
from the actual pretrained policy. Output metrics.json and rollout.npz.
Torque is recorded as max_commanded_torque_nm. v2 matches upstream float32
state casting before PD arithmetic; check_twist_parity.py executes actual
upstream run() and verifies identical observations and qpos over 200 steps.
Do not repair code or silently change the invocation if it fails.

## What the adapter changes

Headless/offline transport replaces GUI and Redis. Constants and quaternion /
wrist functions are extracted from pinned official code. 105 current values
plus 10 historical vectors = 1155 inputs; 23 policy actions; two wrist-roll
targets routed directly to PD, 25 actuators total. 1 kHz simulation, 50 Hz policy,
official PD and torque clipping. Root quaternion explicitly wxyz. Official
high-level one-frame lookahead retained. Initial quaternion uses physical wxyz
identity instead of the low-level script's xyzw-looking [0,0,0,1] literal.

29-joint references are mapped by name; four wrist pitch/yaw joints have no
actuated counterpart in this official model and are explicitly reported as
omitted. Existing 29-DOF data are never overwritten. Each paired actor gets its
own policy history; both share the same physics scene and input world frame.
Official collision geometry retained. Any collision does not certify an
intended attack/block. This is not a matched-simulator BeyondMimic ablation.

## Current limits

Standing is an official default command, not a downloaded motion-dataset
example. Pretraining overlap with our motions is unknown. No training, new
synthetic-data utility conclusion, self-play, or real robot deployment is
claimed by this smoke test. Failed setup attempts (JIT helper loading and
offscreen framebuffer sizing) were fixed before the successful physical run.
Reference velocities are taken from our existing NPZ, not recomputed using
the official MotionLib's 19-frame smoothing; this is a documented input-data
choice. Only the low-level controller is parity-certified. Official data
examples are external downloads, and are not bundled in the pinned repository.
