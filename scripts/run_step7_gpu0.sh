#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_timotion.sh scripts/infer_actor_reaction.py --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --events records/paired_v1/pseudo_contact_events/events.jsonl --adapter runs/actor_reaction_seed0/best_adapter.pt --output runs/step7_train20_generation --limit 20 --seed 7100 --split train > records/step7/train_generation_v2.log 2>&1
test -s runs/step7_train20_generation/evaluation.json
bash scripts/run_timotion.sh scripts/refine_reaction_contacts.py --run runs/step7_train20_generation --output runs/step7_train20_contact --method kinematic --contact-target 0 --device cuda:0 > records/step7/train_contact.log 2>&1
for arm in generic contact; do
  input=runs/step7_train20_generation/inpainting_only
  if [[ "$arm" == contact ]]; then input=runs/step7_train20_contact; fi
  bash scripts/run_retarget.sh scripts/retarget_paired_positions.py --input "$input" --output "runs/step7_${arm}_g1ik" > "records/step7/retarget_${arm}.log" 2>&1
  bash scripts/run_isaaclab.sh scripts/convert_paired_tracking_references.py --input "runs/step7_${arm}_g1ik" --output "runs/step7_${arm}_tracking_refs" --headless --device cuda:0 > "records/step7/convert_${arm}.log" 2>&1
done
bash scripts/run.sh scripts/prepare_step7_utility.py > records/step7/bundles.log 2>&1
for seed in 0 1; do
  bash scripts/run_step7_utility_job.sh generic "$seed" cuda:0
  bash scripts/run_step7_utility_job.sh contact "$seed" cuda:0
done
echo STEP7_GPU0_COMPLETE
