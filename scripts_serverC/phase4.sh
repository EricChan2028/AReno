#!/usr/bin/env bash
# Isolated kernel benchmark on the final code once every GPU job has finished.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q 'phase3 done' $LOG 2>/dev/null; do sleep 60; done
log phase4 start
source /root/kivi/env.sh
WT=/root/kivi/wt/final_f333cdc
mkdir -p /root/kivi/runs/bench_final
cd $WT && git rev-parse HEAD > /root/kivi/runs/bench_final/commit.txt
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > /root/kivi/runs/bench_final/gpus_before.txt
PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=0 python examples/kivi/bench_decode_attention.py --cuda-graph --warmup 2 --iters 20   --rows 1 8 32 128 256 --context 1024 4096 8192 16384 32768 --output /root/kivi/runs/bench_final/decode_graph.json   > /root/kivi/runs/bench_final/decode_graph.log 2>&1
PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=0 python examples/kivi/bench_decode_attention.py --warmup 2 --iters 20   --rows 1 32 128 --context 1024 8192 32768 --output /root/kivi/runs/bench_final/decode_eager.json   > /root/kivi/runs/bench_final/decode_eager.log 2>&1
log phase4 done
