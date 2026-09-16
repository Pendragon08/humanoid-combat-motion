# Pinned PBHC official deployment witness

Spec env-pbhc-inference.json @3717ab81; upstream commit
ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0. New isolated inference overlay;
existing TWIST/IsaacLab environments unchanged. MuJoCo3.2.3, NumPy1.24.4,
ONNXRuntime1.20.1, inherited Torch2.5.1+cu121, Python3.10.21 rather than3.8.
No IsaacGym training installation is claimed.

## Independent documented invocation

Execute verbatim in PowerShell, no installation or command repair:

```powershell
ssh -o BatchMode=yes -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_pbhc_inference.sh scripts/evaluate_pbhc_official.py --example punch --output runs/pbhc_official_20260916/punch_independent_seed123 --seed 123 > records/pbhc_20260916/punch_independent_seed123.log 2>&1"
```

Require exit0, PBHC_OFFICIAL_WITNESS sentinel (read last8192bytes of log;
upstream prints carriage returns), metrics.json and rollout.npz present.
Expected334frames,6.68seconds, no_fall=true, all_frame_joint_below_0p3=true,
joint_mae approximately0.101964rad, finite CUDA backward witness35.832878.
Minor numeric variation permitted; report actual values and any divergence.

The wrapper executes the actual official urci.py entrypoint. Only disables
the viewer/optional upstream saving and hooks TrySaveMotionStep for recording
and stopping after one complete motion. Upstream observations, ONNX, PD,
motion library, history, switch EMA and4--20ms random actuation delay remain.
Official tracked source files must remain clean. Inference is CPU ONNX;
the separate seeded CUDA computation validates the software/GPU stack.

Original horse stance and horse-stance punch have matched bundled checkpoints.
Hooks/side/roundhouse references do not have bundled matching policies in this
commit. Do not pass modified reference data to a single-motion phase policy
and call that a universal tracking or data-utility experiment.

This acceptance uses joint error and pelvis-height fall checks; it is not
identical to the earlier TWIST root/body strict criterion and not real-robot
replication of paper results. Every result records this distinction.

Independent documentation validation passed on2026-09-16: agent
`/root/pbhc_doc_witness` ran the command verbatim, exit0,334frames/6.68s,
jointMAE0.10196432838103667rad, nofall, finite rollout arrays, clean pinned
upstream. No installation or code repair was required. The output directory
now exists; a future rerun must use a fresh output name (overwriting is refused).
