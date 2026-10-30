#!/usr/bin/env bash
# Full CPU suite under CI emulation for the KIVI head and its upstream base.
source /root/kivi/env.sh
mkdir -p /root/kivi/runs/ci_emulation
for w in final_f333cdc upstream_e17794c; do
  cd /root/kivi/wt/$w && PYTHONPATH=/root/kivi/wt/$w nice -n 10 timeout 3600 python /root/kivi/scripts/ci_emulate.py -q -p no:cacheprovider -rf tests/ examples/agentic/pi/ -k cpu > /root/kivi/runs/ci_emulation/$w.log 2>&1
  echo "exit=$?" >> /root/kivi/runs/ci_emulation/$w.log
done
