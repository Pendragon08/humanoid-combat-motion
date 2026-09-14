# G1 inverse-kinematics environment

Target: `ouyanghao@10.147.17.32`, project `/data/code/ouyanghao/oyh/humanoid-combat-motion`.
SSH key on this Windows host: `C:\Users\Pende\.ssh\id_ed25519`.

The declarative dependency versions are in `env-retarget.json`. The `.venv-retarget`
overlay borrows the existing Python 3.10.21 base read-only; it does not modify it.
GMR revision: `bb1bbe40774794fceb2a7c579a3464a28e68c844`.
MuJoCo 3.3.7, Mink 0.0.13 and DAQP 0.7.2 perform CPU inverse kinematics.
Use the following exact invocation after logging in:

```sh
cd /data/code/ouyanghao/oyh/humanoid-combat-motion
bash scripts/run_retarget.sh scripts/smoke_retarget.py
```

Expected: exit 0 and `GMR_IK_WITNESS`, nq=36, nv=35, finite configuration,
position error after solving less than 10% of initial error. The seeded test
moves the left wrist by 2 cm in x and 1 cm in z, with orientation unconstrained.
It applies joint configuration limits. This is a numerical solver test, not
robot dynamics, balance, collision safety or learned policy validation.

The witness is saved to `records/paired_v1/retarget_environment_witness.json`.
The optional XRobotStreamer SDK is absent; its import warning is unrelated to
this offline solver. Installation was completed using project-local temporary
storage. SciPy 1.15.3 is reused from the base rather than downloading 1.14.1.
