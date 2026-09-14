#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 5; done
test -s runs/actor_reaction_kinematic_contact_g1ik/manifest.json
bash scripts/run_retarget.sh scripts/audit_retargeted_contact.py --human-evaluation runs/actor_reaction_test_seed123/evaluation.json --references runs/actor_reaction_kinematic_contact_g1ik --output runs/actor_reaction_kinematic_contact_g1ik/contact_audit.json > records/paired_v1/retargeted_contact_audit.log 2>&1
bash scripts/run_isaaclab.sh scripts/convert_paired_tracking_references.py --input runs/actor_reaction_kinematic_contact_g1ik --output runs/actor_reaction_kinematic_contact_tracking_refs --headless --device cuda:0 > records/paired_v1/reaction_contact_reference_conversion.log 2>&1
test -s runs/actor_reaction_kinematic_contact_tracking_refs/manifest.json
printf '%s\n' CONTACT_REFERENCE_FOLLOWUPS_COMPLETE
