#!/usr/bin/env bash
# usage: eval_ablation.sh GPU NAME DTYPE GROUP RESIDUAL
set -euo pipefail
source /root/kivi/env.sh
WT=/root/kivi/wt/exp_390ac48
export PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=$1
OUT=/root/kivi/runs/eval_sampled/$2
mkdir -p $OUT
(cd $WT && git rev-parse HEAD) > $OUT/commit.txt
cd $WT
python examples/kivi/rollout_eval.py --model /root/kivi/models/Qwen3-0.6B --dataset /root/kivi/data/gsm8k_main_test \
  --kv-cache-dtype $3 --kivi-group-size $4 --kivi-residual-length $5 --n-samples 4 --temperature 1.0 --seed 1234 \
  --max-new-tokens 2048 --max-running-prompts 128 --output $OUT > $OUT/run.log 2>&1
