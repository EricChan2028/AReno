#!/usr/bin/env bash
# usage: lc_throughput.sh GPU DTYPE MAX_RUNNING_PROMPTS [RESIDUAL_LENGTH]
# Greedy rollout of 512 GSM8K test problems, each preceded by an ~8K-token
# WikiText-2 passage; records tokens/s, accuracy and the GPU's peak memory
# sampled once per second by nvidia-smi.
set -uo pipefail
source /root/kivi/env.sh
WT=/root/kivi/wt/exp_390ac48
GPU=$1; DTYPE=$2; MRP=$3; RES=${4:-128}
OUT=/root/kivi/runs/long_context/${DTYPE}_mrp${MRP}
[ "$RES" != 128 ] && OUT=/root/kivi/runs/long_context/${DTYPE}_r${RES}_mrp${MRP}
mkdir -p $OUT
(cd $WT && git rev-parse HEAD) > $OUT/commit.txt
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $GPU > $OUT/mem_before.txt
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU >> $OUT/mem_trace.txt; sleep 1; done ) &
POLL=$!
cd $WT
PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=$GPU python examples/kivi/rollout_eval.py --model /root/kivi/models/Qwen3-0.6B \
  --dataset /root/kivi/data/gsm8k_test512_ctx8k --kv-cache-dtype $DTYPE --kivi-residual-length $RES --max-prompt-tokens 9216 \
  --max-new-tokens 2048 --max-running-prompts $MRP --output $OUT > $OUT/run.log 2>&1
echo "exit=$?" >> $OUT/run.log
kill $POLL
sort -n $OUT/mem_trace.txt | tail -1 > $OUT/peak_mem_mib.txt
