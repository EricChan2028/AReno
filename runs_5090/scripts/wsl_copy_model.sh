#!/usr/bin/env bash
# Read-only copy of a verified Qwen3-0.6B download on this machine into /root/kivi590/models, then
# verify the copy against the source's SHA256SUMS.txt and the report's hash.
set -uo pipefail
SRC=/root/galore593/models/Qwen3-0.6B
DST=/root/kivi590/models/Qwen3-0.6B
REPORT=f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
log() { echo "$(date '+%F %T') $*" | tee -a /root/kivi590/gpu_env.log; }
mkdir -p $DST
for f in LICENSE SHA256SUMS.txt config.json generation_config.json merges.txt model.safetensors tokenizer.json \
  tokenizer_config.json vocab.json; do
  nice -n 19 ionice -c 3 cp --no-preserve=ownership "$SRC/$f" "$DST/$f" || { log "copy failed: $f"; exit 1; }
done
cd $DST
sha256sum -c SHA256SUMS.txt || { log "copy does not match SHA256SUMS.txt"; exit 1; }
got=$(sha256sum model.safetensors | cut -d' ' -f1)
[ "$got" = "$REPORT" ] || { log "model.safetensors $got differs from the report's $REPORT"; exit 1; }
log "MODEL_DONE: copied from $SRC, SHA256SUMS.txt OK, model.safetensors = report hash"
