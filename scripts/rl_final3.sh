#!/usr/bin/env bash
# usage: WT=... [HARNESS_ARGS="--kivi-residual-length 512"] rl_final3.sh GPU NAME SEED DTYPE STEPS
# Same run as rl_final2.sh, with extra train_seeded.py options before "--",
# followed by held-out validation of the final policy on the same GPU.
set -euo pipefail
source /root/kivi/env.sh
WT=${WT:?set WT to the experiment worktree}
GPU=$1; NAME=$2; SEED=$3; DTYPE=$4; STEPS=$5
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT
RUN=/root/kivi/runs/rl_final/$NAME
mkdir -p $RUN
(cd $WT && git rev-parse HEAD) > $RUN/commit.txt
pip freeze 2>/dev/null | grep -i -E "^(torch|triton|flash.attn|transformers)==" > $RUN/versions.txt || true
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i $GPU > $RUN/gpu.txt
echo "train_seeded.py --seed $SEED --kv-cache-dtype $DTYPE ${HARNESS_ARGS:-}" > $RUN/harness.txt
cd $WT
python examples/kivi/train_seeded.py --seed $SEED --kv-cache-dtype $DTYPE ${HARNESS_ARGS:-} -- \
  --algo gspo --ckpt /root/kivi/models/Qwen3-0.6B \
  --dataset-path /root/kivi/data/gsm8k_main_train --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --batch-size 32 --n-samples 8 --max-running-prompts 128 \
  --max-prompt-tokens 512 --max-new-tokens 2048 --max-steps $STEPS --save-interval 50 \
  --metrics-log-dir $RUN/tb --save-path $RUN/ckpt > $RUN/train.log 2>&1
bash /root/kivi/scripts/val_eval.sh $GPU $NAME $RUN/ckpt/step_$(printf "%06d" $STEPS)
echo "$(date "+%F %T") $NAME done" >> /root/kivi/logs/orchestrate.log
