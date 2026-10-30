#!/usr/bin/env bash
# Server C (172.17.0.31): KIVI-4 RL arm, save/resume from server 1's step-50
# checkpoints, remaining ablations. Server 1 exports runs/rl_final over HTTP.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
S1=http://172.17.0.46:18765/rl_final
source /root/kivi/env.sh
log "serverC phase3 start"
for seed in 1 2 3; do
  (bash /root/kivi/scripts/rl_final.sh $((seed - 1)) s${seed}_kivi4 $seed kivi4 100 --save-interval 50; log "s${seed}_kivi4 exit $?") &
done
resume() {  # GPU DTYPE: wait for server 1's s1_<DTYPE> step-50 save, fetch it, continue 50 steps
  local gpu=$1 dtype=$2 src=s1_$2
  local dst=/root/kivi/runs/from_server1/$src/ckpt
  until curl -sf $S1/$src/train.log | grep -q "stage=save_checkpoint_end path=.*step_000050"; do sleep 60; done
  mkdir -p $dst
  wget -q -r -l 8 -np -nH --cut-dirs=3 -R "index.html*" -P $dst $S1/$src/ckpt/step_000050/
  (cd $dst/step_000050 && find . -type f | sort | xargs sha256sum) > $dst/step_000050.sha256
  log "fetched $src step_000050 ($(du -sh $dst/step_000050 | cut -f1))"
  bash /root/kivi/scripts/rl_resume.sh $gpu ${src}_resume50 1 $dtype 50 $dst/step_000050 50 1600
  log "${src}_resume50 exit $?"
}
resume 3 kivi2 &
resume 4 auto &
(
  bash /root/kivi/scripts/eval_ablation.sh 5 kivi2_g32_r512 kivi2 32 512
  log "ablation kivi2_g32_r512 exit $?"
  WT=/root/kivi/wt/eval_5706d51
  cd $WT && PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=5 python examples/kivi/teacher_forced_eval.py \
    --model /root/kivi/models/Qwen3-0.6B --text /root/kivi/data/wikitext2_test.txt --length 16384 --rows 4 \
    --prefill 256 --kivi-group-size 64 --kv-cache-dtypes kivi4 \
    --output /root/kivi/runs/teacher_forced/len16384_rows4_kivi4_g64.json \
    > /root/kivi/runs/teacher_forced/len16384_rows4_kivi4_g64.log 2>&1
  log "TF 16K kivi4 G64 exit $?"
) &
wait
log "serverC phase3 done"
