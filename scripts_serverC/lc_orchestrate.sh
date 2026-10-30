#!/usr/bin/env bash
# Long-context rollouts on server C's spare GPUs: same prompts, BF16 vs KIVI at
# equal concurrency and at the highest concurrency each cache fits.
set -uo pipefail
LOG=/root/kivi/logs/orchestrate.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
S=/root/kivi/scripts/lc_throughput.sh
log "long-context start"
(bash $S 6 auto 32; bash $S 6 auto 48; bash $S 6 auto 64; log "lc gpu6 done") &
(bash $S 7 kivi2 32; bash $S 7 kivi2 128; bash $S 7 kivi2 256; log "lc gpu7 done") &
(until grep -q "TF 16K kivi4 G64 exit" $LOG; do sleep 30; done; bash $S 5 kivi4 32; bash $S 5 kivi4 128; log "lc gpu5 done") &
wait
log "long-context done"
