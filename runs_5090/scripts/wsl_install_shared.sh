#!/usr/bin/env bash
# KIVI #590: install from shared, checksummed copies on this machine
# (read-only use of other jobs' directories).
#   wsl_install_shared.sh wheels WHEELHOUSE          torch stack from WHEELHOUSE (SHA256SUMS checked)
#   wsl_install_shared.sh model MODEL_DIR SHA256     copy Qwen3-0.6B into /root/kivi590/models
set -uo pipefail
# The host proxy currently breaks HTTPS (SSL EOF): mirror downloads go direct,
# for this script's processes only.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
ROOT=/root/kivi590
cd $ROOT
export PIP_CACHE_DIR=$ROOT/cache/pip TMPDIR=$ROOT/cache/tmp
N="nice -n 19 ionice -c 3"
MIRROR=https://pypi.org/simple
FALLBACK=https://mirrors.aliyun.com/pypi/simple
PIPFLAGS="--resume-retries 100 --retries 20 --timeout 60"
log() { echo "$(date '+%F %T') $*" | tee -a $ROOT/gpu_env.log; }
fail() { log "FAILED: $*"; exit 1; }
source venv-gpu/bin/activate
case "${1:-}" in
  wheels)
    WH=$2 SUMS_SHA=$3 REQ_SHA=$4
    [ "$(sha256sum "$WH/SHA256SUMS" | cut -d' ' -f1)" = "$SUMS_SHA" ] || fail "SHA256SUMS file hash"
    [ "$(sha256sum "$WH/requirements-hashed.txt" | cut -d' ' -f1)" = "$REQ_SHA" ] || fail "requirements-hashed.txt hash"
    (cd "$WH" && sha256sum --quiet -c SHA256SUMS) || fail "wheelhouse checksums"
    log "wheelhouse $WH verified ($(wc -l < "$WH/SHA256SUMS") files)"
    # torch and its CUDA / Triton wheels: offline, every wheel checked again against its pinned hash.
    $N python -m pip install -q --require-hashes --no-deps --no-index --find-links "$WH" -r "$WH/requirements-hashed.txt" \
      || fail "torch stack from wheelhouse"
    # torch's lower-bound-only dependencies, direct from pypi.org.
    $N python -m pip install -q $PIPFLAGS -i $MIRROR "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" filelock \
      "typing-extensions>=4.10.0" || fail "torch small dependencies"
    log "torch: $(python -c 'import torch, triton; print(torch.__version__, torch.version.cuda, "triton", triton.__version__)' 2>&1 | tail -1)"
    # Light runtime and tool packages (pins as in requirements/default.txt); torch is already satisfied.
    PKGS="ninja psutil pytest numpy pydantic>=2 click>=8.1 fastapi>=0.110 openai uvicorn>=0.27 safetensors>=0.4
      transformers>=5.15,<6 huggingface-hub>=0.25 datasets==4.0.0 tensorboard tqdm>=4.66 modelscope>=1.20 addict>=2.4
      math-verify pyarrow matplotlib"
    $N python -m pip install -q $PIPFLAGS -i $MIRROR --find-links "$WH" $PKGS \
      || $N python -m pip install -q $PIPFLAGS -i $FALLBACK --find-links "$WH" $PKGS \
      || fail "runtime packages"
    python -m pip check | tail -5 | tee -a $ROOT/gpu_env.log
    python -m pip freeze > $ROOT/cache/tmp/pip_freeze_after_wheels.txt
    log WHEELS_DONE
    ;;
  model)
    SRC=$2 WANT=$3
    mkdir -p models/Qwen3-0.6B
    got=$(sha256sum "$SRC/model.safetensors" | cut -d' ' -f1)
    [ "$got" = "$WANT" ] || fail "source model sha256 $got != $WANT"
    # Read-only copy of the files a checkpoint needs (no hidden/cache entries).
    find "$SRC" -maxdepth 1 -type f ! -name '.*' -exec cp -f {} models/Qwen3-0.6B/ \;
    got=$(sha256sum models/Qwen3-0.6B/model.safetensors | cut -d' ' -f1)
    [ "$got" = "$WANT" ] || fail "copied model sha256 $got != $WANT"
    ls -la models/Qwen3-0.6B | tee -a $ROOT/gpu_env.log
    log "MODEL_DONE sha256=$got (report: f47f7117...6874b)"
    ;;
  *) echo "usage: $0 wheels WHEELHOUSE | model MODEL_DIR SHA256"; exit 2 ;;
esac
