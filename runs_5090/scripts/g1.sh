#!/usr/bin/env bash
# G1 (KIVI #590 on the laptop RTX 5090): CUDA tests, decode-kernel benchmark
# against the default native BF16 backend, teacher-forced WikiText-2 quality.
set -uo pipefail
source "${BASH_SOURCE%/*}/gpu_lib.sh"
require_idle_gpu
python -c 'import areno.accel._areno_accel' || { log "AReno extension missing; run g0_extension.sh first"; exit 4; }
log "G1 start at $(git -C $PR log --oneline -1)"

mkdir -p $OUT/env
nvidia-smi > $OUT/env/nvidia-smi.txt
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv >> $OUT/env/nvidia-smi.txt
python - > $OUT/env/versions.txt <<'PY'
import platform, sys
import torch, triton, transformers
print("python", sys.version.split()[0], platform.platform())
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__, "transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0),
      "memory_gib", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except ImportError:
    print("flash_attn not installed (optional since the native default)")
PY
pip freeze > $OUT/env/pip_freeze.txt
git -C $PR log --oneline -3 > $OUT/env/commit.txt
(cd $ROOT && sha256sum models/Qwen3-0.6B/model.safetensors data/wikitext2_raw_test.parquet data/wikitext2_test.txt \
  data/gsm8k_main_train/data-00000-of-00001.arrow data/gsm8k_main_test/data-00000-of-00001.arrow) > $OUT/env/sha256.txt
cat $OUT/env/versions.txt | tee -a $OUT/g.log

# 1. Tests on this GPU: the KIVI suites plus the upstream native-attention,
#    phase-memory and log-prob GPU tests that share the inference path.
run_guarded tests_kivi timeout 2400 python -m pytest -q -p no:cacheprovider -rfEs \
  tests/test_kivi_kv_cache_cuda.py tests/test_kivi_kv_cache_cpu.py
run_guarded tests_upstream_gpu timeout 2400 python -m pytest -q -p no:cacheprovider -rfEs \
  tests/test_native_attention_gpu.py tests/test_cuda_phase_memory_gpu.py tests/test_logprobs_cuda.py
tail -3 $OUT/tests_kivi/run.log $OUT/tests_upstream_gpu/run.log | tee -a $OUT/g.log

# 2. Decode attention step, one layer, BF16 native backend vs KIVI (2 warmup + 20 measured).
B=examples/kivi/bench_decode_attention.py
run_guarded bench_graph timeout 2400 python $B --cuda-graph --warmup 2 --iters 20 \
  --rows 1 8 32 128 --context 1024 4096 8192 16384 --output $OUT/bench_graph/decode_graph.json
run_guarded bench_graph_32k timeout 1200 python $B --cuda-graph --warmup 2 --iters 20 \
  --rows 1 8 32 --context 32768 --output $OUT/bench_graph_32k/decode_graph_32k.json
run_guarded bench_eager timeout 1200 python $B --warmup 2 --iters 20 \
  --rows 1 32 128 --context 1024 8192 --output $OUT/bench_eager/decode_eager.json

# 3. Teacher-forced WikiText-2, 4 x 16384 tokens, G = 32, R = 128 (same windows as the report).
run_guarded tf_len16384 timeout 3600 python examples/kivi/teacher_forced_eval.py --model $M \
  --text $D/wikitext2_test.txt --length 16384 --rows 4 --prefill 256 --output $OUT/tf_len16384/len16384_rows4.json
log G1_DONE
