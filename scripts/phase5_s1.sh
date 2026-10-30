#!/usr/bin/env bash
# Server 1: validate the base model and the six final BF16/KIVI-2 policies on GPUs 1-5
# while phase4b benchmarks kernels on GPU 0.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "phase2 done" $LOG 2>/dev/null; do sleep 60; done
log phase5 start
V=/root/kivi/scripts/val_eval.sh
R=/root/kivi/runs/rl_final
(bash $V 1 s1_auto $R/s1_auto/ckpt/step_000100; bash $V 1 base /root/kivi/models/Qwen3-0.6B) &
(bash $V 2 s2_auto $R/s2_auto/ckpt/step_000100; bash $V 2 s3_kivi2 $R/s3_kivi2/ckpt/step_000100) &
bash $V 3 s3_auto $R/s3_auto/ckpt/step_000100 &
bash $V 4 s1_kivi2 $R/s1_kivi2/ckpt/step_000100 &
bash $V 5 s2_kivi2 $R/s2_kivi2/ckpt/step_000100 &
wait
log phase5 done
