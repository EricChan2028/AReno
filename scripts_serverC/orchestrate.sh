#!/usr/bin/env bash
# Phase 0: wait for the RL pilots and sampled evals to release their GPUs.
# Phase 1: teacher-forced long-context evals (C) + max-concurrency throughput (D).
# Phase 2: final 3-seed GSPO runs (A) on all six GPUs.
set -uo pipefail
source /root/kivi/env.sh
LOG=/root/kivi/logs/orchestrate.log
EVAL_WT=/root/kivi/wt/eval_2fa4279
log() { echo "$(date '+%F %T') $*" >> $LOG; }
log phase0 waiting
until ! pgrep -f train_seeded.py > /dev/null && ! pgrep -f rollout_eval.py > /dev/null; do sleep 20; done
log phase1 start
mkdir -p /root/kivi/runs/teacher_forced
tf() { # GPU LENGTH ROWS
  (cd $EVAL_WT && git rev-parse HEAD > /root/kivi/runs/teacher_forced/commit_$2.txt
   PYTHONPATH=$EVAL_WT CUDA_VISIBLE_DEVICES=$1 python examples/kivi/teacher_forced_eval.py --model /root/kivi/models/Qwen3-0.6B      --text /root/kivi/data/wikitext2_test.txt --length $2 --rows $3 --prefill 256 --kv-cache-dtypes kivi2 kivi4      --output /root/kivi/runs/teacher_forced/len$2_rows$3.json > /root/kivi/runs/teacher_forced/len$2_rows$3.log 2>&1)
}
tf 0 16384 4 &
tf 1 32768 2 &
(bash /root/kivi/scripts/throughput.sh 2 auto 128 1024; bash /root/kivi/scripts/throughput.sh 2 auto 192 1024) &
bash /root/kivi/scripts/throughput.sh 3 kivi2 256 1024 &
bash /root/kivi/scripts/throughput.sh 4 kivi2 512 1024 &
bash /root/kivi/scripts/throughput.sh 5 kivi4 512 1024 &
wait
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
