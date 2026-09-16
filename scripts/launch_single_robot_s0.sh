#!/usr/bin/env bash
set -uo pipefail
cd /data/code/ouyanghao/oyh/humanoid-combat-motion
out=runs/single_robot_s0_20260916
test -f "$out/protocol.json" || exit 2
test ! -f "$out/launcher_started.txt" || exit 3
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv > "$out/gpu_samples.csv"
date -u > "$out/launcher_started.txt"
bash scripts/run.sh scripts/single_robot_s0.py run --shard 0 > "$out/worker0.log" 2>&1 &
pid0=$!
bash scripts/run.sh scripts/single_robot_s0.py run --shard 1 > "$out/worker1.log" 2>&1 &
pid1=$!
printf '%s\n%s\n' "$pid0" "$pid1" > "$out/worker_pids.txt"
while kill -0 "$pid0" 2>/dev/null || kill -0 "$pid1" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$out/gpu_samples.csv"
    sleep 2
done
wait "$pid0"; code0=$?
wait "$pid1"; code1=$?
printf '{"worker0_exit":%s,"worker1_exit":%s}\n' "$code0" "$code1" > "$out/launcher_finished.json"
test "$code0" -eq 0 && test "$code1" -eq 0
