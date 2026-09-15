# Paired geometry refinement: reproducible bounded experiment

Environment: warm reuse of existing project .venv on
ouyanghao@10.147.17.32. No package changes. NumPy1.26.4, SciPy1.15.3,
MuJoCo3.3.7, OSQP1.0.5, Torch2.5.1+cu121. Existing TWIST env ledger
is .aris/compute/twist.md; task parameters configs/paired_refinement_v1.json.

## Independent numerical witness

Run this exact PowerShell invocation; it does not create or overwrite files:

```powershell
ssh -i C:\Users\Pende\.ssh\id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run.sh scripts/smoke_paired_refinement.py"
```

Expected exit0, PAIRED_QP_WITNESS with 4 frames, original floor penetration
greater than 4cm, corrected penetration below one quarter of original,
normal_jacobian_verified true, dynamics_certified false. This checks the
contact Jacobian sign and a real native OSQP/MuJoCo correction, not imports.
The test is deliberately geometrical, not learned tracking or force feasibility.
If it fails, report exact divergence; do not fix the invocation or code.

## Frozen experiment logic

All 24 generic +24 adapted paired references from the prior suite are used.
One CPU optimization per pair; the exact original reference is preserved.
Two arms use the SAME reference exporter and derivative convention:
R0 = original qpos, re-exported using TWIST FK and derivatives;
R1 = constrained corrected qpos, same exporter.
Compare R0 and R1 using unchanged evaluate_twist.py and pretrained checkpoint,
same original timing, nominal start, collision geometry and thresholds.
Prior raw-export TWIST results remain contextual data, not the sole control.

Method: sequential QP with contact normal Jacobians, joint limits, posture and
temporal correction penalties, heuristic stance foot height/XY anchors, bounded
root/joint changes and nonlinear objective line search. Both robots share one
model. Interrobot collision constraints never disable geometry. No slower
playback, no pose replacement, no actor separation/recentering.

Contact normal convention follows MuJoCo docs:
https://mujoco.readthedocs.io/en/stable/computation/
Actual residual penetration is measured using forward kinematics afterward.
Soft constraints can remain violated. A low-velocity/relative-height stance
heuristic is not a contact annotation. No torque/dynamic-feasibility guarantee,
no contact-semantic correctness guarantee, no synthetic-data-utility claim.

The first adapted clip's first8frames and full clip are implementation pilot
checks; the final48-case panel is development analysis, not an untouched test.

Final batch location: runs/paired_refinement_v2, records/paired_refinement_v2.
v1 was interrupted at generic003 because a previous-frame correction made
the next frame's initial joint angle exceed a limit and local trust region.
The final implementation projects warm-start joint angles to legal limits
before each frame's optimization. The v1 partial batch is not pooled with v2.
