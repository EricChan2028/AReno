#!/usr/bin/env bash
# Re-run the 16K KIVI-4 G=64 teacher-forced case on GPU 6 once the long-context runs free it.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date "+%F %T") $*" >> $LOG; }
until grep -q "lc gpu6 done" $LOG; do sleep 30; done
source /root/kivi/env.sh
WT=/root/kivi/wt/eval_5706d51
OUT=/root/kivi/runs/teacher_forced/len16384_rows4_kivi4_g64
mv $OUT.log $OUT.failed_no_extension.log 2>/dev/null
cd $WT && PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=6 python examples/kivi/teacher_forced_eval.py \
  --model /root/kivi/models/Qwen3-0.6B --text /root/kivi/data/wikitext2_test.txt --length 16384 --rows 4 \
  --prefill 256 --kivi-group-size 64 --kv-cache-dtypes kivi4 --output $OUT.json > $OUT.log 2>&1
log "TF 16K kivi4 G64 retry exit $?"
