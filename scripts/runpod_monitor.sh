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
set -uo pipefail

POD_ID="${1:?usage: runpod_monitor.sh <pod_id> <artifact_prefix> <out_dir>}"
ARTIFACT_PREFIX="${2:?usage: runpod_monitor.sh <pod_id> <artifact_prefix> <out_dir>}"
OUT_DIR="${3:?usage: runpod_monitor.sh <pod_id> <artifact_prefix> <out_dir>}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MONITOR_TIMEOUT_SECS="${MONITOR_TIMEOUT_SECS:-6000}"

mkdir -p "$OUT_DIR"

terminated=0
teardown() {
  if [ "$terminated" -eq 0 ]; then
    terminated=1
    echo "monitor: deleting pod $POD_ID" >&2
    runpodctl pod delete "$POD_ID" >/dev/null 2>&1 || true
  fi
}
trap teardown EXIT INT TERM

deadline=$(($(date +%s) + MONITOR_TIMEOUT_SECS))
code=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! runpodctl pod list 2>/dev/null | grep -q "$POD_ID"; then
    echo "monitor: pod $POD_ID no longer listed (self-terminated)" >&2
    break
  fi

  logs=$(runpodctl pod logs "$POD_ID" 2>/dev/null || true)
  code=$(printf '%s\n' "$logs" | grep -oE "RP_ARTIFACT_CODE=${ARTIFACT_PREFIX}[-0-9]+" | tail -1 | cut -d= -f2)
  if [ -n "$code" ]; then
    echo "monitor: artifact code received: $code" >&2
    break
  fi
  if printf '%s\n' "$logs" | grep -q "^pod: self-terminating"; then
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
