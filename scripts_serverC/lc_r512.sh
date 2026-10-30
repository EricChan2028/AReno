#!/usr/bin/env bash
# KIVI-2 with a 512-token residual window on the 8K-context rollouts (GPU 6 is idle until the KIVI-2 resume).
LOG=/root/kivi/logs/orchestrate.log
bash /root/kivi/scripts/lc_throughput.sh 6 kivi2 32 512
bash /root/kivi/scripts/lc_throughput.sh 6 kivi2 128 512
echo "$(date "+%F %T") lc kivi2 r512 done" >> $LOG
