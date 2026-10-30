#!/usr/bin/env bash
# Server 1 after phase 2: the s*_kivi2 runs here went through a harness whose KV
# override never reached the GSPO config, so they ran BF16 with the same seeds.
# Keep them as BF16 repeats, then validate base, BF16 and BF16-repeat policies.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "phase2 done" $LOG 2>/dev/null; do sleep 60; done
R=/root/kivi/runs/rl_final
for n in 1 2 3; do
  mv $R/s${n}_kivi2 $R/s${n}_bf16repeat
  echo "Launched as kv_cache_dtype=kivi2 through train_seeded.py before 5c49afe; the override did not reach RolloutTrainerConfig.cuda_config, so this run used the BF16 cache with seed $n." > $R/s${n}_bf16repeat/NOTE.txt
done
log "phase5 start (s*_kivi2 renamed to s*_bf16repeat)"
V=/root/kivi/scripts/val_eval.sh
(bash $V 1 s1_auto $R/s1_auto/ckpt/step_000100; bash $V 1 base /root/kivi/models/Qwen3-0.6B) &
(bash $V 2 s2_auto $R/s2_auto/ckpt/step_000100; bash $V 2 s3_bf16repeat $R/s3_bf16repeat/ckpt/step_000100) &
bash $V 3 s3_auto $R/s3_auto/ckpt/step_000100 &
bash $V 4 s1_bf16repeat $R/s1_bf16repeat/ckpt/step_000100 &
bash $V 5 s2_bf16repeat $R/s2_bf16repeat/ckpt/step_000100 &
wait
log phase5 done
