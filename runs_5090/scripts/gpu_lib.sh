#!/usr/bin/env bash
# Shared setup for the KIVI #590 GPU runs on the laptop RTX 5090 (WSL).
# Limits on this shared machine: one GPU job at a time, VRAM <= 20 GB,
# job host RAM <= 14 GB, stop when WSL as a whole uses more than 34 GB,
# everything under nice/ionice and inside /root/kivi590.
ROOT=/root/kivi590
PR=$ROOT/AReno-pr
OUT=$ROOT/runs_5090
M=$ROOT/models/Qwen3-0.6B
D=$ROOT/data
VRAM_LIMIT_MIB=20000
JOB_RSS_LIMIT_KB=$((14 * 1024 * 1024))
WSL_USED_LIMIT_KB=$((34 * 1024 * 1024))
mkdir -p $OUT $ROOT/cache/tmp $ROOT/cache/triton $ROOT/cache/hf $ROOT/cache/modelscope
source $ROOT/venv-gpu/bin/activate
export PIP_CACHE_DIR=$ROOT/cache/pip TMPDIR=$ROOT/cache/tmp TRITON_CACHE_DIR=$ROOT/cache/triton \
  HF_HOME=$ROOT/cache/hf HF_DATASETS_CACHE=$ROOT/cache/hf/datasets MODELSCOPE_CACHE=$ROOT/cache/modelscope \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PR PYTHONUNBUFFERED=1
if [ -d $ROOT/cuda-home ]; then export CUDA_HOME=$ROOT/cuda-home PATH=$ROOT/cuda-home/bin:$PATH; fi
# Run everything from the WSL checkout so Python never imports another copy of areno.
cd $PR

log() { echo "$(date '+%F %T') $*" | tee -a $OUT/g.log; }

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9'; }

# Refuse to start while another job holds the GPU.
require_idle_gpu() {
  local used
  used=$(gpu_used_mib)
  if [ "${used:-99999}" -gt 1500 ]; then
    log "GPU busy (${used} MiB used); not starting"
    nvidia-smi | tee -a $OUT/g.log
    exit 3
  fi
}

tree_pids() {
  ps -eo pid=,ppid= | awk -v root="$1" '
    { parent[$1] = $2 }
    END {
      keep[root] = 1; grown = 1
      while (grown) { grown = 0; for (p in parent) if (!(p in keep) && (parent[p] in keep)) { keep[p] = 1; grown = 1 } }
      for (p in keep) print p
    }'
}

# Every process of a job: descendants of its root plus anything that carries the
# job's environment marker (covers re-parented or new-session processes).
job_pids() {  # NAME ROOT_PID
  { tree_pids "$2"; grep -l -z -s "^KIVI590_JOB=$1\$" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3; } | sort -u
}

job_rss_kb() {  # NAME ROOT_PID; always prints exactly one number
  local pids
  pids=$(job_pids "$1" "$2" | paste -sd, -)
  if [ -z "$pids" ]; then
    echo 0
    return
  fi
  # ps fails when a listed process has just exited; awk still prints the sum.
  { ps -o rss= -p "$pids" 2>/dev/null || true; } | awk '{ s += $1 } END { print s + 0 }'
}

# First line of a value, digits only, 0 when empty: keeps the guard's arithmetic safe.
as_int() { local v=${1%%$'\n'*}; v=${v//[^0-9]/}; echo ${v:-0}; }

kill_job() {  # NAME ROOT_PID
  local pids
  pids=$(job_pids "$1" "$2")
  kill -TERM $pids 2>/dev/null
  sleep 10
  kill -KILL $(job_pids "$1" "$2") $pids 2>/dev/null
}

wsl_used_kb() { awk '/MemTotal/ { t = $2 } /MemAvailable/ { a = $2 } END { print t - a }' /proc/meminfo; }

# Wait (up to MAX_MIN minutes) until WSL as a whole uses less than GB gigabytes.
wait_wsl_below() {  # GB MAX_MIN
  local waited=0
  while [ "$(wsl_used_kb)" -ge $(($1 * 1000 * 1000)) ]; do
    [ $waited -ge $(($2 * 60)) ] && { log "WSL still at $(( $(wsl_used_kb) / 1024 )) MiB after $2 min"; return 1; }
    [ $((waited % 300)) -eq 0 ] && log "waiting for WSL below $1 GB (now $(( $(wsl_used_kb) / 1024 )) MiB)"
    sleep 30
    waited=$((waited + 30))
  done
}

# Wait for another job's device memory to drain before starting.
wait_idle_gpu() {  # MAX_SEC
  local waited=0
  while [ "$(gpu_used_mib)" -gt 1500 ] && [ $waited -lt $1 ]; do sleep 5; waited=$((waited + 5)); done
}

# run_guarded NAME CMD...: run CMD (cwd = PR tree) with a 1 Hz VRAM trace and the
# limits above; outputs land in $OUT/NAME (run.log, mem_trace.txt, peak files).
run_guarded() {
  local name=$1
  shift
  local dir=$OUT/$name
  mkdir -p $dir
  gpu_used_mib > $dir/mem_before.txt
  git -C $PR rev-parse HEAD > $dir/commit.txt
  printf '%q ' "$@" > $dir/cmd.txt
  echo >> $dir/cmd.txt
  # A caller may raise the host-RAM cap for one run (JOB_RSS_LIMIT_OVERRIDE_KB=... run_guarded ...).
  local limit_kb=${JOB_RSS_LIMIT_OVERRIDE_KB:-$JOB_RSS_LIMIT_KB}
  log "start $name (job RAM cap $((limit_kb / 1024)) MiB)"
  (cd $PR && export KIVI590_JOB=$name && exec nice -n 19 ionice -c 3 "$@") > $dir/run.log 2>&1 &
  local pid=$! peak=0 peak_rss=0 peak_wsl=0 reason="" gpu rss wsl start=$SECONDS
  while kill -0 $pid 2>/dev/null; do
    gpu=$(as_int "$(gpu_used_mib)")
    rss=$(as_int "$(job_rss_kb $name $pid)")
    wsl=$(as_int "$(wsl_used_kb)")
    echo "$(date +%s) $gpu $((rss / 1024)) $((wsl / 1024))" >> $dir/mem_trace.txt
    [ "${gpu:-0}" -gt $peak ] && peak=$gpu
    [ "$rss" -gt $peak_rss ] && peak_rss=$rss
    [ "$wsl" -gt $peak_wsl ] && peak_wsl=$wsl
    if [ "${gpu:-0}" -gt $VRAM_LIMIT_MIB ]; then reason="VRAM ${gpu} MiB > ${VRAM_LIMIT_MIB} MiB"; fi
    if [ "$rss" -gt $limit_kb ]; then reason="job RSS $((rss / 1024)) MiB > $((limit_kb / 1024)) MiB"; fi
    if [ "$wsl" -gt $WSL_USED_LIMIT_KB ]; then reason="WSL used $((wsl / 1024)) MiB > 34 GiB"; fi
    if [ -n "$reason" ]; then
      log "STOPPING $name: $reason"
      kill_job $name $pid
      echo "$reason" > $dir/STOPPED
      break
    fi
    sleep 1
  done
  wait $pid
  local rc=$?
  echo $peak > $dir/peak_mem_mib.txt
  echo $((peak_rss / 1024)) > $dir/peak_job_rss_mib.txt
  echo $((peak_wsl / 1024)) > $dir/peak_wsl_used_mib.txt
  log "end $name exit=$rc seconds=$((SECONDS - start)) peak_vram=${peak}MiB peak_job_rss=$((peak_rss / 1024))MiB peak_wsl_used=$((peak_wsl / 1024))MiB"
  return $rc
}
