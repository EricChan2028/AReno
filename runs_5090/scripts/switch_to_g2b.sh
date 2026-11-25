#!/usr/bin/env bash
# When the running G2 runner (g2.sh rl, 14 GiB cap for every arm) finishes the
# BF16 arm and starts KIVI-4, stop it and continue the remaining arms with
# g2b.sh, which gives KIVI-4 and its resume the 16 GiB cap the machine's schedule
# allowed (only while WSL is below 18 GB). The BF16 arm is not touched.
set -uo pipefail
S=/root/kivi590/scripts
OUT=/root/kivi590/runs_5090
RUNNER=$1
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/g.log; }
for i in $(seq 1 1800); do
  grep -q "start rl_s1_kivi4" $OUT/g.log && break
  kill -0 $RUNNER 2>/dev/null || { log "switch: runner $RUNNER ended before KIVI-4 started"; exit 1; }
  sleep 1
done
grep -q "start rl_s1_kivi4" $OUT/g.log || { log "switch: KIVI-4 never started"; exit 1; }
log "switch: stopping runner $RUNNER at the start of KIVI-4 to apply the per-arm caps"
kill -TERM $RUNNER 2>/dev/null
sleep 2
pids=$(grep -l -z -s "^KIVI590_JOB=rl_s1_kivi4\$" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3)
[ -n "$pids" ] && kill -TERM $pids 2>/dev/null
sleep 10
pids=$(grep -l -z -s "^KIVI590_JOB=rl_s1_kivi4\$" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3)
[ -n "$pids" ] && kill -KILL $pids 2>/dev/null
sleep 2
rm -rf $OUT/rl_s1_kivi4
for i in $(seq 1 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')" -lt 1500 ] && break
  sleep 2
done
log "switch: continuing with g2b.sh (kivi4 kivi2 kivi2_r512 resume)"
ARMS="kivi4 kivi2 kivi2_r512 resume" exec bash $S/g2b.sh rl
