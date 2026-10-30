#!/usr/bin/env bash
# Server 1 after validation: remaining long-context rollouts and a Qwen3-1.7B
# GSM8K check (1319 problems x 4 samples, temperature 1.0, seed 1234).
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "phase5 done" $LOG 2>/dev/null && grep -q "phase4 done" $LOG 2>/dev/null; do sleep 30; done
log phase6 start
source /root/kivi/env.sh
L=/root/kivi/scripts/lc_v2.sh
bash $L 0 2359 "kivi2 256" &
bash $L 1 2359 "kivi2 128 512" &
eval_17b() {  # GPU DTYPE
  local out=/root/kivi/runs/eval_sampled_17b/$2
  mkdir -p $out
  (cd /root/kivi/wt/exp_390ac48 && git rev-parse HEAD) > $out/commit.txt
  cd /root/kivi/wt/exp_390ac48 && PYTHONPATH=/root/kivi/wt/exp_390ac48 CUDA_VISIBLE_DEVICES=$1 python examples/kivi/rollout_eval.py \
    --model /root/kivi/models/Qwen3-1.7B --dataset /root/kivi/data/gsm8k_main_test --kv-cache-dtype $2 \
    --n-samples 4 --temperature 1.0 --seed 1234 --max-new-tokens 2048 --max-running-prompts 128 \
    --output $out > $out/run.log 2>&1
  log "eval_17b $2 exit $?"
}
eval_17b 2 auto &
eval_17b 3 kivi2 &
eval_17b 4 kivi4 &
wait
log phase6 done
