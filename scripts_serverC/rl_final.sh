#!/usr/bin/env bash
# usage: rl_final.sh GPU NAME SEED DTYPE STEPS [extra areno train args...]
set -euo pipefail
source /root/kivi/env.sh
WT=/root/kivi/wt/exp_390ac48
GPU=$1; NAME=$2; SEED=$3; DTYPE=$4; STEPS=$5; shift 5
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT
RUN=/root/kivi/runs/rl_final/$NAME
mkdir -p $RUN
(cd $WT && git rev-parse HEAD) > $RUN/commit.txt
pip freeze 2>/dev/null | grep -i -E "^(torch|triton|flash.attn|transformers)==" > $RUN/versions.txt || true
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU > $RUN/gpu.txt
cd $WT
python examples/kivi/train_seeded.py --seed $SEED --kv-cache-dtype $DTYPE -- \
  --algo gspo --ckpt /root/kivi/models/Qwen3-0.6B \
  --dataset-path /root/kivi/data/gsm8k_main_train --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --batch-size 32 --n-samples 8 --max-running-prompts 128 \
  --max-prompt-tokens 512 --max-new-tokens 2048 --max-steps $STEPS \
  --metrics-log-dir $RUN/tb --save-path $RUN/ckpt "$@" > $RUN/train.log 2>&1
