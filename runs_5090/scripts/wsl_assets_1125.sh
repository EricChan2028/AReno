#!/usr/bin/env bash
# KIVI #590 assets that only this task uses: GSM8K (ModelScope) and WikiText-2.
# No torch / nvidia-* / triton wheels and no model download here (shared copies
# come from verified copies already on this machine). Light pip packages only.
set -uo pipefail
# The host proxy currently breaks every HTTPS connection (SSL EOF); these
# downloads go direct. Only this script's processes are affected.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
ROOT=/root/kivi590
cd $ROOT
mkdir -p cache/pip cache/tmp cache/modelscope cache/hf data
export PIP_CACHE_DIR=$ROOT/cache/pip TMPDIR=$ROOT/cache/tmp MODELSCOPE_CACHE=$ROOT/cache/modelscope \
  HF_HOME=$ROOT/cache/hf HF_DATASETS_CACHE=$ROOT/cache/hf/datasets
N="nice -n 19 ionice -c 3"
MIRROR=https://pypi.org/simple
FALLBACK=https://mirrors.aliyun.com/pypi/simple
log() { echo "$(date '+%F %T') $*" | tee -a $ROOT/gpu_env.log; }
fail() { log "FAILED: $*"; exit 1; }
log "assets phase start (GSM8K + WikiText-2 only)"
rm -rf $ROOT/data/gsm8k_repo
source venv-gpu/bin/activate
PIPFLAGS="--resume-retries 100 --retries 20 --timeout 60"
$N python -m pip install -q $PIPFLAGS -i $MIRROR 'modelscope>=1.20' 'addict>=2.4' 'datasets==4.0.0' pyarrow \
  || $N python -m pip install -q $PIPFLAGS -i $FALLBACK 'modelscope>=1.20' 'addict>=2.4' 'datasets==4.0.0' pyarrow \
  || fail "modelscope/datasets"
$N python - <<'PY' || fail "GSM8K"
import glob
import os

root = "/root/kivi590"
missing = [split for split in ("train", "test") if not os.path.exists(f"{root}/data/gsm8k_main_{split}")]
if missing:
    from modelscope.msdatasets import MsDataset

    for split in missing:
        dataset = MsDataset.load("gsm8k", subset_name="main", split=split, trust_remote_code=True)
        # Older modelscope returns an MsDataset wrapper, 1.40 the datasets.Dataset itself.
        if hasattr(dataset, "to_hf_dataset"):
            dataset = dataset.to_hf_dataset()
        dataset.save_to_disk(f"{root}/data/gsm8k_main_{split}")
from datasets import load_from_disk

for split in ("train", "test"):
    print(split, len(load_from_disk(f"{root}/data/gsm8k_main_{split}")), "rows")
PY
if [ ! -f data/wikitext2_test.txt ]; then
  $N curl -fsSL --retry 10 --retry-delay 5 -o data/wikitext2_raw_test.parquet \
    https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet \
    || fail "WikiText-2 parquet"
  python -c "import pyarrow.parquet as pq; open('data/wikitext2_test.txt','w',encoding='utf-8').write(''.join(pq.read_table('data/wikitext2_raw_test.parquet').column('text').to_pylist()))" \
    || fail "WikiText-2 text"
fi
# Report: parquet sha256 5f1bea06...6d91; GSM8K 7473 train / 1319 test rows.
sha256sum data/wikitext2_raw_test.parquet data/wikitext2_test.txt data/gsm8k_main_*/data-*.arrow | tee -a $ROOT/gpu_env.log
log ASSETS_DONE
