# TIMotion compatibility environment

Spec hash: bbd2963d. Specification: env-timotion.json.

Server: ouyanghao@10.147.17.32, existing authorized SSH key.
Working directory: /data/code/ouyanghao/oyh/humanoid-combat-motion

Exact invocation:
```sh
ssh ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_timotion.sh scripts/smoke_timotion.py'
```

Expected: exit 0, TIMOTION_CUDA_WITNESS, shape [2,12,64], finite loss around 7.9470152855 and NVIDIA RTX A6000. This tests upstream Transformer forward/backward, not pretrained inference.

Local overlay .venv-timotion; base environment remains read-only. Torch 2.5.1 differs from authors' 1.13.1. Independent doc-following validation passed on 2026-09-14: exit 0, identical witness and loss; no environment/doc divergence. This does not validate pretrained inference.
