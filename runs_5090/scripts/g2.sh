#!/usr/bin/env bash
# G2 (KIVI #590 on the laptop RTX 5090), sized for <= 20 GB of VRAM:
#   1. GSPO with the public Trainer (areno train), seed 1, STEPS steps per arm:
#      trainer-native train/logp_abs_diff_mean on the current main; KIVI-4
#      saves at STEPS/2 and a fresh run resumes from that checkpoint.
#   2. 8K-token-context GSM8K rollouts (first 64 problems of the report's set)
#      at equal concurrency and at the largest concurrency that fits ~17 GiB.
#   3. GSM8K test, first 512 problems, one sample at T = 1 (seed 1234).
set -uo pipefail
source "${BASH_SOURCE%/*}/gpu_lib.sh"
require_idle_gpu
python -c 'import areno.accel._areno_accel' || { log "AReno extension missing; run g0_extension.sh first"; exit 4; }
STAGE=${1:-all}
STEPS=${STEPS:-8}
HALF=$((STEPS / 2))
BATCH=16
log "G2 start at $(git -C $PR log --oneline -1), STEPS=$STEPS"

if [ $STAGE = all ] || [ $STAGE = rl ]; then
TRAIN_ARGS=(--algo gspo --dataset-path $D/gsm8k_main_train --dataset-loader-fn examples/math/dataset_loader.py
  --reward-fn-path examples/math/math_verify_reward.py --world-size 1 --tp-size 1 --batch-size $BATCH
  --n-samples 4 --mini-bs 4 --max-running-prompts 32 --max-prompt-tokens 512 --max-new-tokens 2048)
rl() {  # NAME [train_seeded.py options] -- [extra areno train options]
  local name=$1
  shift
  local opts=() extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  extra=("$@")
  run_guarded $name timeout 3600 python examples/kivi/train_seeded.py --seed 1 "${opts[@]}" -- "${TRAIN_ARGS[@]}" \
    --metrics-log-dir $OUT/$name/tb "${extra[@]}"
}
rl rl_s1_auto --kv-cache-dtype auto -- --ckpt $M --max-steps $STEPS
rl rl_s1_kivi4 --kv-cache-dtype kivi4 -- --ckpt $M --max-steps $STEPS --save-path $OUT/rl_s1_kivi4/ckpt --save-interval $HALF
rl rl_s1_kivi2 --kv-cache-dtype kivi2 -- --ckpt $M --max-steps $STEPS
rl rl_s1_kivi2_r512 --kv-cache-dtype kivi2 --kivi-residual-length 512 -- --ckpt $M --max-steps $STEPS
CKPT=$OUT/rl_s1_kivi4/ckpt/$(printf 'step_%06d' $HALF)
if [ -d $CKPT ]; then
  rl rl_s1_kivi4_resume$HALF --kv-cache-dtype kivi4 --resume-step $HALF --resume-prompts $((HALF * BATCH)) -- \
    --ckpt $CKPT --max-steps $((STEPS - HALF))
else
  log "no checkpoint at $CKPT; resume skipped"
fi
python examples/kivi/summarize_rl.py --window $STEPS --output $OUT/rl_summary.json \
  auto=$OUT/rl_s1_auto kivi4=$OUT/rl_s1_kivi4 kivi2=$OUT/rl_s1_kivi2 kivi2_r512=$OUT/rl_s1_kivi2_r512 \
  > $OUT/rl_summary.log 2>&1
tail -30 $OUT/rl_summary.log | tee -a $OUT/g.log
log G2_RL_DONE
fi

# 2. Long context: the first 64 problems of the report's 512-problem set.
if [ $STAGE = all ] || [ $STAGE = lc ]; then
LC=$D/gsm8k_test64_ctx8k
[ -d $LC ] || python examples/kivi/build_long_context_gsm8k.py --tokenizer $M --gsm8k $D/gsm8k_main_test \
  --text $D/wikitext2_test.txt --limit 64 --output $LC > $OUT/build_lc64.log 2>&1
lc() {  # NAME DTYPE MRP [options]
  local name=$1 dtype=$2 mrp=$3
  shift 3
  run_guarded $name timeout 2400 python examples/kivi/rollout_eval.py --model $M --dataset $LC --kv-cache-dtype $dtype \
    --max-prompt-tokens 9216 --max-running-prompts $mrp --output $OUT/$name "$@"
}
lc lc64_auto_mrp8 auto 8
lc lc64_kivi4_mrp8 kivi4 8
lc lc64_kivi2_mrp8 kivi2 8
lc lc64_kivi4_mrp24 kivi4 24
lc lc64_kivi2_mrp40 kivi2 40
log G2_LC_DONE
fi

# 3. GSM8K test, first 512 problems, T = 1, one sample, seed 1234.
if [ $STAGE = all ] || [ $STAGE = gsm8k ]; then
for dtype in auto kivi4 kivi2; do
  run_guarded gsm8k512_$dtype timeout 2400 python examples/kivi/rollout_eval.py --model $M --dataset $D/gsm8k_main_test \
    --kv-cache-dtype $dtype --limit 512 --temperature 1.0 --seed 1234 --max-running-prompts 32 --output $OUT/gsm8k512_$dtype
done
grep -h "KIVI KV cache:" $OUT/*/run.log | sed 's/.*KIVI KV cache:/KIVI KV cache:/' | sort | uniq -c | tee -a $OUT/g.log
log G2_GSM8K_DONE
fi
