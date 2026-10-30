#!/usr/bin/env bash
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date "+%F %T") $*" >> $LOG; }
until grep -q "phase2 done" $LOG 2>/dev/null; do sleep 30; done
log phase3 start
for seed in 1 2 3; do bash /root/kivi/scripts/rl_final.sh $((seed - 1)) s${seed}_kivi4 $seed kivi4 100 --save-interval 50 & done
bash /root/kivi/scripts/rl_resume.sh 3 s1_kivi2_resume50 1 kivi2 50 /root/kivi/runs/rl_final/s1_kivi2/ckpt/step_000050 50 1600 &
(bash /root/kivi/scripts/eval_ablation.sh 4 kivi4_g64_r128 kivi4 64 128; bash /root/kivi/scripts/eval_ablation.sh 4 kivi4_g128_r128 kivi4 128 128) &
(bash /root/kivi/scripts/eval_ablation.sh 5 kivi2_g32_r256 kivi2 32 256; bash /root/kivi/scripts/eval_ablation.sh 5 kivi2_g32_r512 kivi2 32 512) &
wait
log phase3 done
