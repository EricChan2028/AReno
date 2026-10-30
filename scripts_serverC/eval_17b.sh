#!/usr/bin/env bash
# Qwen3-1.7B scaling check on GPU 6 while it waits for the KIVI-2 resume:
# first 512 GSM8K test problems x 4 samples, temperature 1.0, seed 1234.
# Each run must start before 05:10 so GPU 6 is free when the resume begins.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date "+%F %T") $*" >> $LOG; }
until grep -q "lc kivi2 r512 done" $LOG; do sleep 20; done
source /root/kivi/env.sh
WT=/root/kivi/wt/exp_390ac48
for dtype in auto kivi2 kivi4; do
  [ "$(date +%H%M)" -lt 0510 ] || { log "eval_17b $dtype skipped (late)"; continue; }
  OUT=/root/kivi/runs/eval_sampled_17b/$dtype
  mkdir -p $OUT
  (cd $WT && git rev-parse HEAD) > $OUT/commit.txt
  cd $WT && PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=6 python examples/kivi/rollout_eval.py --model /root/kivi/models/Qwen3-1.7B \
    --dataset /root/kivi/data/gsm8k_main_test --limit 512 --kv-cache-dtype $dtype --n-samples 4 --temperature 1.0 \
    --seed 1234 --max-new-tokens 2048 --max-running-prompts 128 --output $OUT > $OUT/run.log 2>&1
  log "eval_17b $dtype exit $?"
done
