#!/usr/bin/env bash
# G1 follow-up (KIVI #590, laptop RTX 5090):
#  - upstream GPU tests again now that flash-linear-attention (a requirement in
#    requirements/default.txt) is installed, on the PR tree and on main;
#  - decode benchmark without BF16 at 128 rows x 16K: upstream's native
#    `_decode_partial` computes cache offsets in 32 bits and faults once a layer
#    cache passes 2^31 elements (8320 blocks here); KIVI runs that size alone.
set -uo pipefail
source "${BASH_SOURCE%/*}/gpu_lib.sh"
require_idle_gpu
log "G1b start at $(git -C $PR log --oneline -1)"
[ -d $OUT/bench_graph ] && [ ! -d $OUT/bench_graph_native_fault_128x16k ] && mv $OUT/bench_graph $OUT/bench_graph_native_fault_128x16k
UP="tests/test_native_attention_gpu.py tests/test_cuda_phase_memory_gpu.py tests/test_logprobs_cuda.py"
run_guarded tests_upstream_gpu_pr timeout 2400 python -m pytest -q -p no:cacheprovider -rfEs $UP
run_guarded tests_upstream_gpu_main timeout 2400 bash -c \
  "cd $ROOT/AReno-main && PYTHONPATH=$ROOT/AReno-main exec python -m pytest -q -p no:cacheprovider -rfEs $UP"
tail -2 $OUT/tests_upstream_gpu_pr/run.log $OUT/tests_upstream_gpu_main/run.log | tee -a $OUT/g.log
B=examples/kivi/bench_decode_attention.py
run_guarded bench_graph timeout 1800 python $B --cuda-graph --warmup 2 --iters 20 \
  --rows 1 8 32 128 --context 1024 4096 8192 --output $OUT/bench_graph/decode_graph.json
run_guarded bench_graph_16k timeout 1200 python $B --cuda-graph --warmup 2 --iters 20 \
  --rows 1 8 32 --context 16384 --output $OUT/bench_graph_16k/decode_graph_16k.json
run_guarded bench_graph_128x16k_kivi timeout 1200 python $B --cuda-graph --warmup 2 --iters 20 \
  --rows 128 --context 16384 --dtypes kivi2 kivi4 --output $OUT/bench_graph_128x16k_kivi/decode_graph_128x16k_kivi.json
log G1B_DONE
