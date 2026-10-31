# Pod-side observability for RunPod jobs. Sourced by scripts/runpod_lab_entrypoint.sh; the launcher inlines this
# file into the pod's start command, so it must stay self-contained and must not need the network.
#
# Why: a failure on a pod used to cost the full lifetime and leave no evidence (the pod and its logs vanish
# together). This gives, with no human watching:
#   * `phase NAME LIMIT`    a named phase with a deadline; the watcher fails the job if it overruns
#   * heartbeats            a line every OBS_HEARTBEAT_SECS with phase, disk and GPU, so silence means trouble
#   * fatal signatures      known fatal log lines fail the job at once instead of at the lifetime limit
#   * stall detection       no new log output for OBS_STALL_SECS fails the job
#   * obs_fail / obs_finish ship a diagnostics (or results) bundle off the pod BEFORE terminating, and print
#                           the first error and the tail of the logs into the job log itself
# `terminate` must be defined by the caller; it deletes the pod. Everything acts once (a lock directory).

OBS_DIR="${OBS_DIR:-/workspace}"
LOG="${LOG:-$OBS_DIR/job.log}"
OBS_HEARTBEAT_SECS="${OBS_HEARTBEAT_SECS:-30}"
OBS_WATCH_SECS="${OBS_WATCH_SECS:-5}"
OBS_STALL_SECS="${OBS_STALL_SECS:-300}"
OBS_SHIP_WAIT_SECS="${OBS_SHIP_WAIT_SECS:-120}"
OBS_FATAL_RE="${OBS_FATAL_RE:-No space left on device|CUDA out of memory|Engine core initialization failed|vLLM exited|LAB_EXPERIMENT_FAILED|LAB_QUEUE_ERROR|BOOT_CHECK_FAILED}"
OBS_EXTRA_LOGS="${OBS_EXTRA_LOGS:-/tmp/arc3-lab-run-*/server.log}"
OBS_T0="$(date +%s)"
mkdir -p "$OBS_DIR"

obs_now() { date +%s; }

phase() { # phase NAME [LIMIT_SECS]: end the previous phase, start this one (names have no spaces)
  local name="$1" limit="${2:-0}" now pn ps pl
  now="$(obs_now)"
  if [ -f "$OBS_DIR/phase" ]; then
    read -r pn ps pl <"$OBS_DIR/phase"
    echo "PHASE_END=$pn dt=$((now - ps))s"
  fi
  echo "$name $now $limit" >"$OBS_DIR/phase.tmp" && mv "$OBS_DIR/phase.tmp" "$OBS_DIR/phase"
  echo "PHASE=$name t=$now since_start=$((now - OBS_T0))s limit=${limit}s"
}

obs_heartbeat() {
  local name disk gpu
  while sleep "$OBS_HEARTBEAT_SECS"; do
    name="-"
    [ -f "$OBS_DIR/phase" ] && read -r name _ <"$OBS_DIR/phase"
    disk="$(df -Pk "$OBS_DIR" 2>/dev/null | awk 'NR==2{printf "%.0f", $4/1048576}')"
    gpu="$(timeout 5 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    echo "HEARTBEAT t=$(obs_now) up=$(($(obs_now) - OBS_T0))s phase=$name disk_free_gb=${disk:-?} gpu=${gpu:--}"
  done
}

obs_watch() {
  local offset=0 last_size=-1 last_change now pn ps pl size new hit
  last_change="$(obs_now)"
  while sleep "$OBS_WATCH_SECS"; do
    now="$(obs_now)"
    if [ -f "$OBS_DIR/phase" ]; then # 1. the phase deadline
      read -r pn ps pl <"$OBS_DIR/phase"
      if [ "${pl:-0}" -gt 0 ] && [ $((now - ps)) -gt "$pl" ]; then
        obs_fail "phase $pn exceeded its ${pl}s limit"
        return
      fi
    fi
    size="$(wc -c <"$LOG" 2>/dev/null || echo 0)"
    if [ "$size" -gt "$offset" ]; then # 2. known-fatal lines in what was written since the last look
      new="$(tail -c +"$((offset + 1))" "$LOG" | grep -Ev 'HEARTBEAT|^FATAL:|^--- DIAG|PHASE' || true)"
      hit="$(printf '%s\n' "$new" | grep -Em1 -- "$OBS_FATAL_RE" || true)"
      offset="$size"
      if [ -n "$hit" ]; then
        obs_fail "fatal line in the job log: $(printf '%s' "$hit" | cut -c1-200)"
        return
      fi
    fi
    if [ "$size" != "$last_size" ]; then # 3. silence (heartbeats count as output)
      last_size="$size"
      last_change="$now"
    elif [ $((now - last_change)) -ge "$OBS_STALL_SECS" ]; then
      obs_fail "no output for ${OBS_STALL_SECS}s"
      return
    fi
  done
}

obs_ship() { # obs_ship KIND: collect, print the evidence into the log, send it off the pod
  local kind="$1" d="$OBS_DIR/diag" f code shown pid waited
  rm -rf "$d" && mkdir -p "$d"
  tail -n 400 "$LOG" >"$d/job.log.tail" 2>&1
  for f in $OBS_EXTRA_LOGS; do
    [ -f "$f" ] && cp "$f" "$d/$(basename "$(dirname "$f")")-$(basename "$f")"
  done
  {
    echo "--- nvidia-smi"; timeout 10 nvidia-smi 2>&1
    echo "--- df"; df -h 2>&1
    echo "--- free"; free -m 2>&1
    echo "--- ps"; ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu 2>&1 | head -25 | cut -c1-200
  } >"$d/system.txt" 2>&1
  if [ -n "${OBS_RESULTS_DIR:-}" ] && [ -d "$OBS_RESULTS_DIR" ]; then # results, run log tails, and traces if small
    mkdir -p "$d/results"
    for f in "$OBS_RESULTS_DIR"/*/; do
      [ -d "$f" ] || { cp "$f" "$d/results/" 2>/dev/null; continue; }
      n="$(basename "$f")"; mkdir -p "$d/results/$n"
      cp "$f"result.json "$f"scorecard.json "$d/results/$n/" 2>/dev/null
      [ -f "$f/run.log" ] && tail -n 400 "$f/run.log" >"$d/results/$n/run.log.tail"
      if [ -d "$f/traces" ] && [ "$(du -sk "$f/traces" | cut -f1)" -lt 102400 ]; then cp -r "$f/traces" "$d/results/$n/traces"; fi
    done
    for f in "$OBS_RESULTS_DIR"/*.json; do [ -f "$f" ] && cp "$f" "$d/results/"; done
  fi
  echo "--- DIAG excerpt ($kind) ---"
  for f in "$d"/*server.log; do
    [ -f "$f" ] || continue
    echo "[$(basename "$f")] first error:"
    grep -nEm1 -B3 -A12 'Error|ERROR|FAILED|Traceback' "$f" | cut -c1-220
    echo "[$(basename "$f")] last lines:"
    tail -n 15 "$f" | cut -c1-220
  done
  echo "[system]"; grep -A3 -E '^--- (df|free)' "$d/system.txt" | cut -c1-160
  echo "[job log tail]"; tail -n 20 "$LOG" | cut -c1-220
  tar czf "$OBS_DIR/diag.tgz" -C "$OBS_DIR" diag 2>/dev/null
  command -v runpodctl >/dev/null 2>&1 || { echo "RP_ARTIFACT_CODE=no-runpodctl"; return 0; }
  code="arc3lab-$kind-$RANDOM"
  timeout "$((OBS_SHIP_WAIT_SECS + 30))" runpodctl send --code "$code" "$OBS_DIR/diag.tgz" >"$OBS_DIR/send.log" 2>&1 &
  pid=$!
  shown=""
  for _ in $(seq 1 40); do
    shown="$(grep -oE "${code}[-0-9a-z]*" "$OBS_DIR/send.log" 2>/dev/null | head -1 || true)"
    [ -n "$shown" ] && break
    sleep 0.5
  done
  echo "RP_ARTIFACT_CODE=${shown:-send-failed}"
  waited=0 # stay up until the driver has received it, but never longer than the ship window
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$OBS_SHIP_WAIT_SECS" ]; do
    sleep 1
    waited=$((waited + 1))
  done
}

obs_fail() { # obs_fail REASON: the one place a failure is handled
  mkdir "$OBS_DIR/.acting" 2>/dev/null || return 0
  echo "FATAL: $*"
  echo "$*" >"$OBS_DIR/FAILED"
  obs_ship failure
  terminate "failure: $*"
}

obs_finish() { # obs_finish: normal end; ship the results, then terminate
  mkdir "$OBS_DIR/.acting" 2>/dev/null || return 0
  echo "DONE: job finished"
  obs_ship results
  terminate "job finished"
}

obs_start() { # obs_start: heartbeats and the watcher, both writing into the job log as well as stdout
  obs_heartbeat 2>&1 | tee -a "$LOG" &
  obs_watch 2>&1 | tee -a "$LOG" &
}
