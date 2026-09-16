# Single robot S0, 2026-09-16

Environment: warm reuse env-twist-inference.json @86ed9bff, no installs.
Server nlp03, two A6000. Existing outputs must not be overwritten.

## Independent witness invocation

Execute verbatim in PowerShell. Do not fix code or change the command.

```powershell
ssh -o BatchMode=yes -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run.sh scripts/evaluate_twist.py --standing --seconds 5 --output runs/single_robot_s0_independent_20260916 --device cuda:0"
```

Require exit0, TWIST_PHYSICS_WITNESS, 250 frames, strict_complete=[true],
no_fall_complete=[true], nonzero commanded torque. Check metrics.json and
rollout.npz exist. This is a simulator/policy witness, not data-utility evidence.

## Frozen experiment

Prepare: `bash scripts/run.sh scripts/single_robot_s0.py prepare` (already run;
refuses to overwrite). It audits all 53 raw file hashes and selects the first
training clip of the first six title groups in the frozen manifest. No selection
using tracking scores. Full source duration is retained at 50Hz.

Two persistent workers: `bash scripts/run.sh scripts/single_robot_s0.py run --shard 0`
and the same with `--shard 1`. Each worker sequentially evaluates its cases;
no environment or controller changes. Queue state, per-case logs, elapsed times,
source hashes, protocol and outputs: `runs/single_robot_s0_20260916/`.
Unexpected failures halt the worker for diagnosis; no silent metric relaxation.

The three arms are full reference, first150frames, constant initial-ground-height
offset. The offset only removes initial floor penetration, adding 2mm clearance
if penetration exceeds1mm. No temporal scaling or joint-angle correction.
It is an engineering control, **not** a full KungfuBot processing reproduction,
not the proposed phase-aware generator, and not S2's full ablation.

The same frozen official TWIST policy runs every case. Mapping uses the official
25DOF robot and explicitly drops four wrist pitch/yaw DOFs from the29DOF source.
World FK and finite-difference velocities are recomputed at50Hz. No learned
controller training occurs. Script evaluation thresholds remain unchanged.

Progression gate: standing witness passes and at least5/6 original full references
pass every-frame strict tracking before S4; independent skill/source annotations
are additionally required. A failed gate is reported; it is not proof that the
motion is dynamically impossible. Existing low-budget PPO results are historical,
not rerun results. No held-out test source is used in this pilot.

This bounded run answers S0 acceptance questions. S1--S4 and real-robot contact
tests are not completed merely by finishing these19 nominal rollouts.
