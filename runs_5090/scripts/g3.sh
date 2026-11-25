#!/usr/bin/env bash
# G3 (KIVI #590, laptop RTX 5090) on the commit that runs KIVI cached-prefix
# chunks through the native tiled paged kernel: CUDA tests, the long-context
# runs again (the pre-fix runs move to before_fix/), then GSM8K 512.
set -uo pipefail
S="${BASH_SOURCE%/*}"
source "$S/gpu_lib.sh"
require_idle_gpu
log "G3 start at $(git -C $PR log --oneline -1)"
python -c 'import areno.accel._areno_accel' || { log "AReno extension missing"; exit 4; }
run_guarded tests_kivi_fix timeout 2400 python -m pytest -q -p no:cacheprovider -rfEs \
  tests/test_kivi_kv_cache_cuda.py tests/test_kivi_kv_cache_cpu.py
grep -E '[0-9]+ (passed|failed)' $OUT/tests_kivi_fix/run.log | tail -1 | tee -a $OUT/g.log
grep -q " failed" $OUT/tests_kivi_fix/run.log && { log "G3: KIVI tests failed; stopping before the long-context runs"; exit 7; }
mkdir -p $OUT/before_fix
for d in $OUT/lc64_*; do [ -d "$d" ] && mv "$d" $OUT/before_fix/; done
bash $S/g2b.sh lc
bash $S/g2b.sh gsm8k
log G3_DONE
