#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/cache/engineai_overlay:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec bash scripts/run_isaaclab.sh "$@"
