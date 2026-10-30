#!/usr/bin/env bash
# Server C relaunch with the fixed harness (5c49afe): KIVI-2 and KIVI-4 arms,
# save/resume for BF16 and KIVI-2, then held-out validation of each policy.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
export WT=/root/kivi/wt/exp_5c49afe
S1=http://172.17.0.46:18765/rl_final
R=/root/kivi/runs/rl_final
V=/root/kivi/scripts/val_eval.sh
log "serverC relaunch with $WT"
run() {  # GPU NAME SEED DTYPE: train 100 steps, then validate the final policy on the same GPU
  bash /root/kivi/scripts/rl_final2.sh $1 $2 $3 $4 100 --save-interval 50
  log "$2 exit $?"
  bash $V $1 $2 $R/$2/ckpt/step_000100
  log "$2 validation done"
}
run 0 s1_kivi4 1 kivi4 &
run 1 s2_kivi4 2 kivi4 &
run 2 s3_kivi4 3 kivi4 &
run 7 s1_kivi2 1 kivi2 &
run 3 s2_kivi2 2 kivi2 &
run 4 s3_kivi2 3 kivi2 &
(  # BF16 resume control from server 1's seed-1 run, on GPU 5 once the long-context runs end
  until grep -q "lc gpu5 done" $LOG; do sleep 30; done
  until curl -sf $S1/s1_auto/train.log | grep -q "stage=save_checkpoint_end path=.*step_000050"; do sleep 60; done
  dst=/root/kivi/runs/from_server1/s1_auto/ckpt
  mkdir -p $dst
  wget -q -r -l 8 -np -nH --cut-dirs=3 -R "index.html*" -P $dst $S1/s1_auto/ckpt/step_000050/
  (cd $dst/step_000050 && find . -type f | sort | xargs sha256sum) > $dst/step_000050.sha256
  log "fetched s1_auto step_000050 ($(du -sh $dst/step_000050 | cut -f1))"
  bash /root/kivi/scripts/rl_resume2.sh 5 s1_auto_resume50 1 auto 50 $dst/step_000050 50 1600
  log "s1_auto_resume50 exit $?"
  bash $V 5 s1_auto_resume50 $R/s1_auto_resume50/ckpt/step_000050
  log "s1_auto_resume50 validation done"
) &
(  # KIVI-2 resume from this host's seed-1 run, on GPU 6 after the G=64 teacher-forced run
  until grep -q "TF 16K kivi4 G64 retry exit" $LOG; do sleep 30; done
  until grep -q "stage=save_checkpoint_end path=.*step_000050" $R/s1_kivi2/train.log 2>/dev/null; do sleep 60; done
  bash /root/kivi/scripts/rl_resume2.sh 6 s1_kivi2_resume50 1 kivi2 50 $R/s1_kivi2/ckpt/step_000050 50 1600
  log "s1_kivi2_resume50 exit $?"
  bash $V 6 s1_kivi2_resume50 $R/s1_kivi2_resume50/ckpt/step_000050
  log "s1_kivi2_resume50 validation done"
) &
wait
log "serverC all done"
