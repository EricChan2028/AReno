#!/usr/bin/env bash
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date "+%F %T") $*" >> $LOG; }
until grep -q "phase2 done" $LOG 2>/dev/null; do sleep 30; done
log phase3 start
source /root/kivi/env.sh
for seed in 1 2 3; do bash /root/kivi/scripts/rl_final.sh $((seed - 1)) s${seed}_kivi4 $seed kivi4 100 --save-interval 50 & done
bash /root/kivi/scripts/rl_resume.sh 3 s1_kivi2_resume50 1 kivi2 50 /root/kivi/runs/rl_final/s1_kivi2/ckpt/step_000050 50 1600 &
bash /root/kivi/scripts/eval_ablation.sh 4 kivi2_g32_r512 kivi2 32 512 &
(WT=/root/kivi/wt/eval_5706d51; cd $WT && PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=5 python examples/kivi/teacher_forced_eval.py --model /root/kivi/models/Qwen3-0.6B --text /root/kivi/data/wikitext2_test.txt --length 16384 --rows 4 --prefill 256 --kivi-group-size 64 --kv-cache-dtypes kivi4 --output /root/kivi/runs/teacher_forced/len16384_rows4_kivi4_g64.json > /root/kivi/runs/teacher_forced/len16384_rows4_kivi4_g64.log 2>&1) &
wait
log phase3 done
