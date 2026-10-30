#!/usr/bin/env bash
# usage: rl_resume.sh GPU NAME SEED DTYPE STEPS CKPT RESUME_STEP RESUME_PROMPTS
# Continues a saved run: loads CKPT, skips the prompts already consumed and
# keeps the per-step rollout seed schedule of the uninterrupted run.
set -euo pipefail
source /root/kivi/env.sh
WT=/root/kivi/wt/exp_390ac48
GPU=$1; NAME=$2; SEED=$3; DTYPE=$4; STEPS=$5; CKPT=$6; RSTEP=$7; RPROMPTS=$8
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT
RUN=/root/kivi/runs/rl_final/$NAME
mkdir -p $RUN
(cd $WT && git rev-parse HEAD) > $RUN/commit.txt
echo "resumed from $CKPT at step $RSTEP after $RPROMPTS prompts" > $RUN/resume.txt
cd $WT
python examples/kivi/train_seeded.py --seed $SEED --kv-cache-dtype $DTYPE --resume-step $RSTEP --resume-prompts $RPROMPTS -- \
  --algo gspo --ckpt $CKPT \
  --dataset-path /root/kivi/data/gsm8k_main_train --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --batch-size 32 --n-samples 8 --max-running-prompts 128 \
  --max-prompt-tokens 512 --max-new-tokens 2048 --max-steps $STEPS \
  --metrics-log-dir $RUN/tb --save-path $RUN/ckpt --save-interval 50 > $RUN/train.log 2>&1
