# TWIST2 offline deployment witness

Warm reuse `.venv-pbhc-inference` without installations: Python3.10.21,
NumPy1.24.4, MuJoCo3.2.3, ONNXRuntime1.20.1, Torch2.5.1+cu121.
Spec `env-twist2-inference.json`; upstream
`d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a`.

Independent reviewer executes this exact PowerShell command, without installs
or repairs:

```powershell
ssh -o BatchMode=yes -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_pbhc_inference.sh scripts/evaluate_twist2_official.py --standing --output runs/twist2_upgrade_20260916/standing_independent_20k100 > records/tracker_upgrade_20260916/standing_independent_20k100.log 2>&1"
```

Require exit0 and `TWIST2_OFFICIAL_WITNESS`,250 recorded frames,5seconds,
strict_complete=true,no_fall=true,finite `rollout.npz`, policy100Hz,
physics1000Hz, CPU ONNX explicitly selected. Primary jointMAE0.042009rad,
rooterror0.034783m, CUDA backward witness15.11279583. Tiny numeric differences
are allowed. Inspect upstream tracked source remains unmodified.

The wrapper executes the official ONNX wrapper, configuration assignments,
observation/history/inference fragment and PD arithmetic extracted by AST.
It omits Redis, real-time waiting, viewer and hardware interfaces. References
are fed directly at the next50Hz reference frame, held for two policy ticks.
This is a bounded offline deployment adaptation, not an end-to-end Redis
stack or full paper reproduction. It uses the official29DOF robot model and
20k checkpoint at100Hz, matching `sim2sim.sh`. A separate25k/50Hz diagnostic
matches HumanTracker's checkpoint/frequency choice but does not use its
backend, termination or HumanScore implementation.

Do not change prior strict metrics or claim data-training benefit. Native
robot differences prevent an isolated controller-algorithm comparison with
the earlier25DOF TWIST. The4 PBHC motions are development examples only.
