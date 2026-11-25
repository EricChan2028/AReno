#!/usr/bin/env bash
# CPU check while the GPU belongs to another task: `pytest tests/ -k cpu` the way
# the CPU CI job runs it (CPU torch, triton / flash-attn / CUDA hidden) on the
# rebased PR head and on main. Job RSS capped at 4 GiB (stop if exceeded),
# 4 threads, nice/ionice. Logs: /root/kivi590/ci/{pr,main}_<sha>.log
set -uo pipefail
ROOT=/root/kivi590
EMU=/root/kivi590/scripts/ci_emulate.py
LIMIT_KB=$((4 * 1024 * 1024))
source $ROOT/venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date '+%F %T') $*" | tee -a $ROOT/ci/cpu_suite_1125.log; }
tree_pids() {
  ps -eo pid=,ppid= | awk -v root="$1" '{ parent[$1] = $2 } END { keep[root] = 1; grown = 1
    while (grown) { grown = 0; for (p in parent) if (!(p in keep) && (parent[p] in keep)) { keep[p] = 1; grown = 1 } }
    for (p in keep) print p }'
}
run_suite() {  # NAME TREE
  local name=$1 tree=$2 sha out pid peak=0 rss pids wsl reason
  sha=$(git -C $tree rev-parse --short HEAD)
  out=$ROOT/ci/${name}_${sha}.log
  log "start $name at $sha"
  (cd $tree && PYTHONPATH=$tree exec nice -n 19 ionice -c 3 timeout 3600 python $EMU tests/ -k cpu -q \
    -p no:cacheprovider -rfE) > $out 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    pids=$(tree_pids $pid | paste -sd, -)
    rss=$(ps -o rss= -p "$pids" 2>/dev/null | awk '{ s += $1 } END { print s + 0 }')
    [ "$rss" -gt $peak ] && peak=$rss
    wsl=$(awk '/MemTotal/ { t = $2 } /MemAvailable/ { a = $2 } END { print t - a }' /proc/meminfo)
    reason=""
    [ "$rss" -gt $LIMIT_KB ] && reason="RSS $((rss / 1024)) MiB > 4096 MiB"
    [ "$wsl" -gt $((34 * 1024 * 1024)) ] && reason="WSL used $((wsl / 1024)) MiB > 34 GiB"
    if [ -n "$reason" ]; then
      log "STOPPING $name: $reason"
      kill -TERM $(tree_pids $pid) 2>/dev/null; sleep 5; kill -KILL $(tree_pids $pid) 2>/dev/null
      break
    fi
    sleep 2
  done
  wait $pid
  log "end $name exit=$? peak_rss=$((peak / 1024))MiB: $(grep -E '[0-9]+ (passed|failed)' $out | tail -1)"
}
run_suite pr $ROOT/AReno-pr
run_suite main $ROOT/AReno-main
grep -h "^FAILED\|^ERROR" $ROOT/ci/pr_$(git -C $ROOT/AReno-pr rev-parse --short HEAD).log | sed 's/ - .*//' | sort > $ROOT/ci/pr_failed.txt
grep -h "^FAILED\|^ERROR" $ROOT/ci/main_$(git -C $ROOT/AReno-main rev-parse --short HEAD).log | sed 's/ - .*//' | sort > $ROOT/ci/main_failed.txt
log "failures only on PR: $(comm -23 $ROOT/ci/pr_failed.txt $ROOT/ci/main_failed.txt | paste -sd' ' -)"
log "failures only on main: $(comm -13 $ROOT/ci/pr_failed.txt $ROOT/ci/main_failed.txt | paste -sd' ' -)"
log CPU_SUITE_DONE
