#!/usr/bin/env bash
# Driver-side monitor for a RunPod job started via scripts/runpod_entrypoint.sh.
#
# This machine is layer 3 of the runpod-jobs skill's three-layer cost-safety
# design (account spend cap, pod-side dead-man's-switch, driver-side trap):
# whatever happens to this script -- success, failure, Ctrl-C, a crashed
# laptop -- the EXIT/INT/TERM trap deletes the pod. The pod's own
# runpod_entrypoint.sh already carries its own max-lifetime + idle watchdog
# (layer 2), so either side terminating is enough; neither is trusted alone.
#
# Usage: scripts/runpod_monitor.sh <pod_id> <artifact_prefix> <out_dir>
# Polls the pod's logs for "RP_ARTIFACT_CODE=<code>", pulls the artifact with
# `runpodctl receive` once seen, then deletes the pod and exits. Also exits
# (pod already deleted by its own dead-man's-switch) if the pod disappears
# from `runpodctl pod list` first.
#
# Lesson from a real 2026-09-21 incident (see the runpod-jobs skill's Rule 0
# and the runpod_dead_mans_switch_ordering memory): a monitor that polls
# silently for a full timeout with no interim visibility can't be diagnosed
# after the fact once the pod is deleted. This version does an early sanity
# check (EARLY_CHECK_SECS) and persists every raw log snapshot it reads to
# $OUT_DIR/monitor_log_snapshots.txt, not just a silent grep.
set -uo pipefail

POD_ID="${1:?usage: runpod_monitor.sh <pod_id> <artifact_prefix> <out_dir>}"
ARTIFACT_PREFIX="${2:?usage: runpod_monitor.sh <pod_id> <artifact_prefix> <out_dir>}"
OUT_DIR="${3:?usage: runpod_monitor.sh <pod_id> <artifact_prefix> <out_dir>}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MONITOR_TIMEOUT_SECS="${MONITOR_TIMEOUT_SECS:-6000}"
EARLY_CHECK_SECS="${EARLY_CHECK_SECS:-120}"

mkdir -p "$OUT_DIR"
SNAPSHOT_FILE="$OUT_DIR/monitor_log_snapshots.txt"
: >"$SNAPSHOT_FILE"

terminated=0
teardown() {
  if [ "$terminated" -eq 0 ]; then
    terminated=1
    echo "monitor: deleting pod $POD_ID" >&2
    runpodctl pod delete "$POD_ID" >/dev/null 2>&1 || true
  fi
}
trap teardown EXIT INT TERM

snapshot_logs() {
  local logs
  logs=$(runpodctl pod logs "$POD_ID" --source both --tail 5000 --max-wait 5s 2>&1 || true)
  {
    echo "=== snapshot at $(date -u +%FT%TZ) ==="
    printf '%s\n' "$logs"
  } >>"$SNAPSHOT_FILE"
  printf '%s' "$logs"
}

start=$(date +%s)
echo "monitor: waiting ${EARLY_CHECK_SECS}s for an early sign of life before the long poll" >&2
sleep "$EARLY_CHECK_SECS"
early_logs=$(snapshot_logs)
if [ -z "$early_logs" ]; then
  echo "monitor: WARNING -- no log output at all after ${EARLY_CHECK_SECS}s (see $SNAPSHOT_FILE); the bootstrap may be stuck" >&2
else
  echo "monitor: early check-in got $(printf '%s' "$early_logs" | wc -l) log line(s), see $SNAPSHOT_FILE" >&2
fi

deadline=$((start + MONITOR_TIMEOUT_SECS))
code=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! runpodctl pod list 2>/dev/null | grep -q "$POD_ID"; then
    echo "monitor: pod $POD_ID no longer listed (self-terminated)" >&2
    break
  fi

  logs=$(snapshot_logs)
  code=$(printf '%s\n' "$logs" | grep -oE "RP_ARTIFACT_CODE=${ARTIFACT_PREFIX}[-0-9]+" | tail -1 | cut -d= -f2)
  if [ -n "$code" ]; then
    echo "monitor: artifact code received: $code" >&2
    break
  fi
  if printf '%s\n' "$logs" | grep -q "pod: self-terminating"; then
    echo "monitor: pod reported self-termination without an artifact code" >&2
    break
  fi

  sleep "$POLL_SECONDS"
done

if [ -n "$code" ]; then
  echo "monitor: receiving artifact into $OUT_DIR" >&2
  (cd "$OUT_DIR" && runpodctl receive "$code")
  receive_exit=$?
  echo "monitor: receive exit code $receive_exit" >&2
fi

teardown
runpodctl pod list
