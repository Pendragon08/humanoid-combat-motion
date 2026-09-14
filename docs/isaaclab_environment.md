# Isaac Lab G1 environment validation

Spec: env-isaaclab.json; hash 072cb4cb.

Server nlp03: ouyanghao@10.147.17.32. Ubuntu20.04, Isaac Sim standalone4.5.0, IsaacLab v2.1.0; G1-only dependencies. Bundled Torch2.5.1+cu118. Optional pink/dex controllers omitted.

Run exactly from local PowerShell:

```powershell
ssh ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_isaaclab.sh scripts/smoke_g1_isaaclab.py --headless --device cuda:1 --output records/paired_v1/g1_independent_validation.json'
```

Require exit code 0, output sentinel G1_ISAACLAB_PHYSICS_WITNESS and nonempty JSON, four articulated robots,200 physics steps,29joint names and finite gradient loss. Main witness gradient_loss=0.06026072055; small physics variation is allowed. Exit0 alone is not success. The wrapper unsets CUDA_VISIBLE_DEVICES; physical GPU is selected by --device. USD conversion uses project cache/usd/g1_tracking.

Shutdown diagnosis: Isaac Lab's timeline stop callback intentionally keeps rendering while paused. Closing the USD stage before unsubscribing this callback hung the first witnesses. The script now calls SimulationContext.clear_instance() before SimulationApp.close(wait_for_replicator=False). No Replicator writer is used. The v8 primary witness completed and its process exited. This changes lifecycle cleanup only, not simulation, policy, rewards or results.

This is physics + GPU backward validation, NOT learned tracking, paired contact or synthetic-data utility. Independent validation pending. Do not install, edit or repair during independent pass; report exact discrepancy.
