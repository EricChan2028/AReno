#!/usr/bin/env bash
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until ! pgrep -f rollout_eval.py > /dev/null && ! pgrep -f teacher_forced_eval.py > /dev/null; do sleep 20; done
log phase1 done
log phase2 start
gpu=0
for seed in 1 2 3; do
  for dtype in auto kivi2; do
    bash /root/kivi/scripts/rl_final.sh $gpu s${seed}_${dtype} $seed $dtype 100 --save-interval 50 &
    gpu=$((gpu + 1))
  done
done
wait
log phase2 done
