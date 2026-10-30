#!/usr/bin/env bash
source /root/kivi/env.sh
OUT=/root/kivi/runs/validation/base_serverC
WT=/root/kivi/wt/eval_5706d51
cd $WT && PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=3 python examples/kivi/teacher_forced_eval.py --model /root/kivi/models/Qwen3-0.6B \
  --text /root/kivi/data/gsm8k_test_solutions.txt --length 4096 --rows 8 --prefill 256 --kv-cache-dtypes kivi2 kivi4 \
  --output $OUT/tf_gsm8k_test.json > $OUT/tf_gsm8k_test.log 2>&1
echo "exit=$?" >> $OUT/tf_gsm8k_test.log
