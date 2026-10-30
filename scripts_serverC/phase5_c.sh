#!/usr/bin/env bash
# Server C: validate the final KIVI-4 policies and the two resumed policies.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "serverC phase3 done" $LOG 2>/dev/null; do sleep 60; done
log "serverC phase5 start"
V=/root/kivi/scripts/val_eval.sh
R=/root/kivi/runs/rl_final
for seed in 1 2 3; do bash $V $((seed - 1)) s${seed}_kivi4 $R/s${seed}_kivi4/ckpt/step_000100 & done
bash $V 3 s1_kivi2_resume50 $R/s1_kivi2_resume50/ckpt/step_000050 &
bash $V 4 s1_auto_resume50 $R/s1_auto_resume50/ckpt/step_000050 &
wait
log "serverC phase5 done"
