#!/usr/bin/env bash
# AReno CUDA extension for the KIVI #590 runs.
#   g0_extension.sh reuse /path/to/_areno_accel...so SHA256   copy a verified same-source build
#   g0_extension.sh build                                     compile here (sm_120, MAX_JOBS=4)
set -uo pipefail
source "${BASH_SOURCE%/*}/gpu_lib.sh"
mode=${1:-}
TARGETS="$PR $ROOT/AReno-main"
csrc_tree=$(git -C $PR rev-parse HEAD:areno/accel/csrc)
setup_blob=$(git -C $PR rev-parse HEAD:setup.py)
torch_version=$(python -c 'import torch; print(torch.__version__)')
abi=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
log "extension: mode=$mode csrc=$csrc_tree setup.py=$setup_blob torch=$torch_version suffix=$abi"
case "$mode" in
  reuse)
    src=$2 want=$3
    [ "$csrc_tree" = "f7818d61ba1f4a271be0ed55166c4734123a975a" ] || { log "csrc tree differs from the reused build"; exit 5; }
    [ "$setup_blob" = "6b25073a2c82bdc8a949310ba8e51a9660fc30a5" ] || { log "setup.py differs from the reused build"; exit 5; }
    [ "$torch_version" = "2.9.1+cu128" ] || { log "torch $torch_version differs from the reused build"; exit 5; }
    got=$(sha256sum "$src" | cut -d' ' -f1)
    [ "$got" = "$want" ] || { log "sha256 mismatch: $got != $want"; exit 5; }
    case "$src" in *"$abi") ;; *) log "ABI suffix of $src does not match $abi"; exit 5 ;; esac
    for tree in $TARGETS; do install -m 755 "$src" "$tree/areno/accel/_areno_accel$abi"; done
    echo "reused $src sha256=$got (built on this machine for the AReno #591 work from the same areno/accel/csrc tree and setup.py)" > $OUT/extension_source.txt
    ;;
  build)
    SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
    CH=$ROOT/cuda-home
    rm -rf $CH && mkdir -p $CH/include $CH/lib64
    ln -s $SITE/nvidia/cuda_nvcc/bin $CH/bin
    [ -d $SITE/nvidia/cuda_nvcc/nvvm ] && ln -s $SITE/nvidia/cuda_nvcc/nvvm $CH/nvvm
    for inc in $SITE/nvidia/*/include; do
      for entry in "$inc"/*; do [ -e "$CH/include/$(basename "$entry")" ] || ln -s "$entry" "$CH/include/"; done
    done
    for lib in $SITE/nvidia/*/lib; do
      for entry in "$lib"/*; do [ -e "$CH/lib64/$(basename "$entry")" ] || ln -s "$entry" "$CH/lib64/"; done
    done
    [ -e $CH/lib64/libcudart.so ] || ln -s "$(ls $CH/lib64/libcudart.so.* | head -1)" $CH/lib64/libcudart.so
    export CUDA_HOME=$CH PATH=$CH/bin:$PATH
    log "nvcc: $(nvcc --version | tail -1)"
    cd $PR
    TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 nice -n 19 ionice -c 3 python -m pip install -e . --no-deps --no-build-isolation \
      > $OUT/extension_build.log 2>&1 || { log "build failed, see $OUT/extension_build.log"; exit 6; }
    cp -f $PR/areno/accel/_areno_accel$abi $ROOT/AReno-main/areno/accel/
    echo "built here: TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4, nvcc $(nvcc --version | tail -1)" > $OUT/extension_source.txt
    ;;
  *) echo "usage: $0 reuse SO SHA256 | build"; exit 2 ;;
esac
for tree in $TARGETS; do
  (cd $tree && python -c 'import areno.accel._areno_accel as m; print(m.__file__)') || { log "import failed in $tree"; exit 7; }
done
sha256sum $PR/areno/accel/_areno_accel$abi | tee -a $OUT/extension_source.txt
echo "csrc_tree=$csrc_tree setup_py=$setup_blob torch=$torch_version" >> $OUT/extension_source.txt
log EXTENSION_READY
