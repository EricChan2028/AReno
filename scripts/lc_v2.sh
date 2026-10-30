#!/usr/bin/env bash
# usage: lc_v2.sh GPU DEADLINE_HHMM SPEC...   (each SPEC is "DTYPE MRP [RESIDUAL]")
# Long-context rollouts on the corrected prompts, one GPU, one run at a time.
# Each run starts only on an empty GPU and before the deadline.
LOG=/root/kivi/logs/orchestrate.log
S=/root/kivi/scripts/lc_throughput.sh
GPU=$1; DEADLINE=$2; shift 2
for spec in "$@"; do
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    [ "$used" -lt 1000 ] && break
    sleep 5
  done
  if [ "$(date +%H%M)" -ge "$DEADLINE" ] || [ "$used" -ge 1000 ]; then
    echo "$(date "+%F %T") lc_v2 gpu$GPU stopped before: $spec (used=${used}MiB)" >> $LOG
    break
  fi
  bash $S $GPU $spec
  echo "$(date "+%F %T") lc_v2 gpu$GPU done: $spec" >> $LOG
done
