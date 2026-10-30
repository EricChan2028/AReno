#!/usr/bin/env bash
# usage: e2e_greedy.sh GPU DTYPE LIMIT
set -euo pipefail
source /root/kivi/env.sh
cd /root/kivi/AReno
export CUDA_VISIBLE_DEVICES=$1
OUT=/root/kivi/runs/e2e_greedy/$2
mkdir -p $OUT
git rev-parse HEAD > $OUT/commit.txt
python examples/kivi/rollout_eval.py --model /root/kivi/models/Qwen3-0.6B --dataset /root/kivi/data/gsm8k_main_test \
  --kv-cache-dtype $2 --limit $3 --max-new-tokens 2048 --max-running-prompts 128 --output $OUT > $OUT/run.log 2>&1
