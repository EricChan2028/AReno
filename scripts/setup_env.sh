#!/usr/bin/env bash
set -euxo pipefail
source /root/kivi/env.sh
cd /root/kivi
[ -d venv ] || python3 -m venv --system-site-packages venv
source venv/bin/activate
python -c "import torch; print(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI)"
pip install -r AReno/requirements/default.txt
pip install psutil ninja pytest pre-commit
cd AReno && pip install -e . --no-deps --no-build-isolation && cd ..
ABI=$(python -c "import torch;print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")
WHL=flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl
curl -fL --retry 3 -o cache/tmp/$WHL https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$WHL
pip install --no-deps cache/tmp/$WHL
python - <<PY
import torch, flash_attn
from flash_attn import flash_attn_func
q=torch.randn(1,128,8,128,device="cuda",dtype=torch.bfloat16)
o=flash_attn_func(q,q,q,causal=True)
torch.cuda.synchronize(); print("flash_attn", flash_attn.__version__, "ok", o.shape)
PY
python -c "import areno, areno.accel; print(\"areno import ok\")"
pip check || true
echo SETUP_DONE
