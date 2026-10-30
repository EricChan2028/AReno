#!/usr/bin/env bash
set -euo pipefail
source /root/kivi/env.sh
cd /root/kivi/AReno
export CUDA_VISIBLE_DEVICES=5
RUN=/root/kivi/runs/pilot_bf16
git rev-parse HEAD > $RUN/commit.txt
nvidia-smi --query-gpu=index,name,memory.used --format=csv > $RUN/gpu_before.txt
areno train --algo gspo --ckpt /root/kivi/models/Qwen3-0.6B \
  --dataset-path /root/kivi/data/gsm8k_main_train --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --batch-size 32 --n-samples 8 --max-running-prompts 128 \
  --max-prompt-tokens 512 --max-new-tokens 2048 --max-steps 20 \
  --metrics-log-dir $RUN/tb --save-interval 100000
