#!/usr/bin/env bash
set -euxo pipefail
source /root/kivi/env.sh
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=120
export MAX_JOBS=96
export NVCC_THREADS=2
cd /root/kivi
pip install --no-build-isolation --no-deps "flash-attn==2.8.3" -v 2>&1 | grep -v -E "^\s+(copying|adding)" | tail -40
python - <<PY
import torch, flash_attn
from flash_attn import flash_attn_func
q=torch.randn(2,256,16,128,device="cuda",dtype=torch.bfloat16); k=torch.randn(2,256,8,128,device="cuda",dtype=torch.bfloat16)
o=flash_attn_func(q,k,k,causal=True); torch.cuda.synchronize()
print("flash_attn", flash_attn.__version__, "ok", tuple(o.shape))
PY
python -c "import areno, areno.accel; print(\"areno import ok\")"
echo FA_DONE
