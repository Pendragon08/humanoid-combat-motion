# PM01 isolated compatibility environment

Specification: configs/engineai_single_spec.json. EngineAI official source commit 14ec57be718586bd0ac45375aa1115bd896fbdbc; native SDK 335c60e88772c26c7852d0abd6b3c7439037dd8f.

Reuses Isaac Sim 4.5.0 / Isaac Lab 2.1.0. Upstream specifies 2.3.2; this is an explicitly isolated compatibility test, not an exact official environment reproduction. No pip installation or edits to existing environments. Overlay suppresses eager task registration and avoids newer PPO config imports; PM01 USD and actuator definitions are unchanged. PM01 assets are resolved absolutely.

Exact independent witness invocation from local PowerShell:

```powershell
ssh -b 10.147.17.250 -o ConnectTimeout=10 -o BatchMode=yes -i C:\Users\Pende\.ssh\id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_engineai.sh scripts/engineai_single_sim.py --mode witness --headless --device cuda:1 --output records/engineai_single_v1/witness_independent.json"
```

Expect exit 0, PM01_ISAAC_PHYSICS_WITNESS, 200 actual physics steps, 24 joints, finite CUDA backward loss, and saved JSON. This certifies model loading, stepping and CUDA only. Standing/tracking success is not the witness criterion; root movement must be retained in output. Do not install, fix or change settings during independent validation. If output already exists, report it rather than overwrite. The initial primary witness returned loss 0.0133422464; small physics roundoff may vary.

Retargeting uses run_engineai_retarget.sh with pre-existing Python3.10 overlays; retarget site packages precede the pilot overlay supplying OSQP. Derived SDK model preserves joint kinematics, inertia and primitive collisions; visuals are removed and hand target sites added. Not a dynamics certificate.
