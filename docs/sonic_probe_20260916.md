# SONIC / HumanTracker bounded adapter witness

Spec env-sonic-probe.json, warm unchanged .venv-pbhc-inference. This is a
Python observation/action adapter from HumanTracker79d637da5e07b8e2c588ca613d3254f9d8acc0f7,
not official C++ deployment or a full HumanTracker installation. Default SONIC
weights revision6733128a3d8a523b1418b06bca3cdf61c8b0987f. Encoder SHA256
013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3,
decoder c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed.
NVIDIA Open Model License applies to weights; source backend Apache2.0.

Independent reviewer executes exactly, no installation/repair:

```powershell
ssh -o BatchMode=yes -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "cd /data/code/ouyanghao/oyh/humanoid-combat-motion && bash scripts/run_pbhc_inference.sh scripts/evaluate_sonic_probe.py --standing --output runs/sonic_probe_20260916/standing_independent > records/tracker_upgrade_20260916/sonic_standing_independent.log 2>&1"
```

Expect exit0,SONIC_PROBE_WITNESS,250frames/5s,strict_complete=true,no_fall=true,
finite arrays. Encoder1762,decoder994,explicit CPU ONNX. Primary jointMAE
0.037237376rad,rooterror0.021384787m,CUDA witness15.11279583. Source tracked
files must stay clean. State actual values, do not fix discrepancies silently.

Policy50Hz,physics1000Hz. Wrapper executes pinned backend observation builder,
motor mapping and constants; physics uses its reset and PD arithmetic.
Post-step metrics compare next reference frame; full original strict thresholds
are retained. No learned HumanScore or original paper table reproduction.
MuJoCo3.2.3 differs from the full benchmark's dependency recipe. Native model,
reset and gains differ from previous TWIST systems. Pretraining overlap unknown.
