#!/usr/bin/env bash
# `pytest tests/ -k cpu` as the CPU CI job runs it, split into chunks of test
# files (one process per chunk) so memory is released between chunks. Same caps
# as approved: chunk process tree RSS <= 4 GiB, stop if WSL uses > 34 GB,
# 4 threads, nice/ionice. PR head first, then main; then compare failures.
set -uo pipefail
ROOT=/root/kivi590
EMU=/root/kivi590/scripts/ci_emulate.py
LIMIT_KB=$((4 * 1024 * 1024))
CHUNK=${CHUNK:-6}
source $ROOT/venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
LOG=$ROOT/ci/cpu_suite_chunked_1125.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
tree_pids() {
  ps -eo pid=,ppid= | awk -v root="$1" '{ parent[$1] = $2 } END { keep[root] = 1; grown = 1
    while (grown) { grown = 0; for (p in parent) if (!(p in keep) && (parent[p] in keep)) { keep[p] = 1; grown = 1 } }
    for (p in keep) print p }'
}
run_chunk() {  # TREE OUT FILES...
  local tree=$1 out=$2 pid peak=0 rss pids wsl reason=""
  shift 2
  (cd $tree && PYTHONPATH=$tree exec nice -n 19 ionice -c 3 timeout 1800 python $EMU "$@" -k cpu -q \
    -p no:cacheprovider -rfE) > $out 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    pids=$(tree_pids $pid | paste -sd, -)
    rss=$(ps -o rss= -p "$pids" 2>/dev/null | awk '{ s += $1 } END { print s + 0 }')
    wsl=$(awk '/MemTotal/ { t = $2 } /MemAvailable/ { a = $2 } END { print t - a }' /proc/meminfo)
    [ "$rss" -gt $peak ] && peak=$rss
    [ "$rss" -gt $LIMIT_KB ] && reason="RSS $((rss / 1024)) MiB > 4096 MiB"
    [ "$wsl" -gt $((34 * 1024 * 1024)) ] && reason="WSL used $((wsl / 1024)) MiB > 34 GiB"
    if [ -n "$reason" ]; then
      kill -TERM $(tree_pids $pid) 2>/dev/null; sleep 5; kill -KILL $(tree_pids $pid) 2>/dev/null
      echo "STOPPED: $reason" >> $out
      break
    fi
    sleep 1
  done
  wait $pid
  echo "PEAK_RSS_MIB $((peak / 1024))" >> $out
}
run_tree() {  # NAME TREE
  local name=$1 tree=$2 sha dir files index=0
  sha=$(git -C $tree rev-parse --short HEAD)
  dir=$ROOT/ci/chunked_${name}_${sha}
  rm -rf $dir && mkdir -p $dir
  mapfile -t files < <(cd $tree && find tests -name 'test_*.py' -not -path '*/__pycache__/*' | sort)
  log "start $name at $sha: ${#files[@]} files in chunks of $CHUNK"
  for ((start = 0; start < ${#files[@]}; start += CHUNK)); do
    index=$((index + 1))
    run_chunk $tree $dir/chunk_$(printf %03d $index).log "${files[@]:start:CHUNK}"
    printf '%s\n' "${files[@]:start:CHUNK}" > $dir/chunk_$(printf %03d $index).files
    if grep -q "^STOPPED" $dir/chunk_$(printf %03d $index).log; then
      log "$name chunk $index stopped: $(grep '^STOPPED' $dir/chunk_$(printf %03d $index).log) files: $(paste -sd' ' $dir/chunk_$(printf %03d $index).files)"
    fi
    if [ "$(awk '/MemTotal/ { t = $2 } /MemAvailable/ { a = $2 } END { print t - a }' /proc/meminfo)" -gt $((34 * 1024 * 1024)) ]; then
      log "WSL above 34 GB; stopping $name"; return 1
    fi
  done
  python - $dir <<'PY' | tee -a $LOG
import glob, re, sys
totals, failed, peak, stopped = {}, [], 0, []
for path in sorted(glob.glob(f"{sys.argv[1]}/chunk_*.log")):
    text = open(path, encoding="utf-8", errors="replace").read()
    summary = [line for line in text.splitlines() if re.search(r"\d+ (passed|failed|skipped|deselected|error)", line)]
    if summary:
        for count, kind in re.findall(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed)", summary[-1]):
            totals[kind] = totals.get(kind, 0) + int(count)
    failed += [line.split(" - ")[0] for line in text.splitlines() if line.startswith(("FAILED", "ERROR"))]
    peak = max([peak] + [int(m) for m in re.findall(r"PEAK_RSS_MIB (\d+)", text)])
    if "STOPPED:" in text:
        stopped.append(path)
print(sys.argv[1], totals, "peak chunk RSS MiB", peak, "stopped chunks", len(stopped))
open(f"{sys.argv[1]}/failed.txt", "w").write("\n".join(sorted(failed)) + ("\n" if failed else ""))
PY
}
run_tree pr $ROOT/AReno-pr
run_tree main $ROOT/AReno-main
P=$ROOT/ci/chunked_pr_$(git -C $ROOT/AReno-pr rev-parse --short HEAD)/failed.txt
Q=$ROOT/ci/chunked_main_$(git -C $ROOT/AReno-main rev-parse --short HEAD)/failed.txt
log "failures on both: $(comm -12 $P $Q | paste -sd' ' -)"
log "failures only on PR: $(comm -23 $P $Q | paste -sd' ' -)"
log "failures only on main: $(comm -13 $P $Q | paste -sd' ' -)"
log CPU_SUITE_CHUNKED_DONE
