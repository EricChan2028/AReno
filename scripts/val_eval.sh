#!/usr/bin/env bash
# usage: val_eval.sh GPU NAME CKPT
# Held-out validation of a policy with the BF16 KV cache: greedy accuracy on the
# 1319 GSM8K test problems and teacher-forced NLL on GSM8K test solutions.
set -uo pipefail
source /root/kivi/env.sh
GPU=$1; NAME=$2; CKPT=$3
OUT=/root/kivi/runs/validation/$NAME
mkdir -p $OUT
echo $CKPT > $OUT/ckpt.txt
WT=/root/kivi/wt/exp_390ac48
(cd $WT && git rev-parse HEAD) > $OUT/commit.txt
cd $WT && PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=$GPU python examples/kivi/rollout_eval.py --model $CKPT \
  --dataset /root/kivi/data/gsm8k_main_test --kv-cache-dtype auto --max-new-tokens 2048 --max-running-prompts 128 \
  --output $OUT/gsm8k_greedy > $OUT/gsm8k_greedy.log 2>&1
echo "exit=$?" >> $OUT/gsm8k_greedy.log
WT=/root/kivi/wt/eval_5706d51
(cd $WT && git rev-parse HEAD) > $OUT/commit_tf.txt
cd $WT && PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=$GPU python examples/kivi/teacher_forced_eval.py --model $CKPT \
  --text /root/kivi/data/gsm8k_test_solutions.txt --length 4096 --rows 8 --prefill 256 --kv-cache-dtypes kivi2 kivi4 \
  --output $OUT/tf_gsm8k_test.json > $OUT/tf_gsm8k_test.log 2>&1
echo "exit=$?" >> $OUT/tf_gsm8k_test.log
