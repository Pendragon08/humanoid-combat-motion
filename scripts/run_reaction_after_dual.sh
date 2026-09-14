#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 1 ]]; then printf '%s\n' 'usage: script dual_queue_pid'; exit 2; fi
while kill -0 "$1" 2>/dev/null; do sleep 10; done
grep -q DUAL_DIAGNOSTIC_QUEUE_COMPLETE records/paired_v1/dual_diagnostic_queue.log
export CUDA_VISIBLE_DEVICES=1
bash scripts/run_timotion.sh scripts/train_actor_reaction.py --manifest records/paired_v1/interhuman_validated_manifest/manifest.jsonl --events records/paired_v1/pseudo_contact_events/events.jsonl --output runs/actor_reaction_seed0 --seed 0 --steps 1000 > records/paired_v1/actor_reaction_training.log 2>&1
printf '%s\n' 'ACTOR_REACTION_QUEUE_COMPLETE'
