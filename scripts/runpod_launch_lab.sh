#!/usr/bin/env bash
# Launch the ARC-AGI-3 agent-lab pod on RunPod (runpod-jobs skill). Secrets are passed BY REFERENCE: jq reads them
# from this shell's environment (never argv), curl gets the API key through a config file on stdin.
#
#   set -a; . .env; set +a            # KAGGLE_USERNAME, KAGGLE_KEY, RUNPOD_API_KEY
#   scripts/runpod_launch_lab.sh up   # create (refuses above LAB_MAX_HOURLY_USD, verifies after create)
#   scripts/runpod_launch_lab.sh watch --max-usd 3.4   # REQUIRED after up: detects failure, saves logs, deletes pod
#   scripts/runpod_launch_lab.sh logs # tail the job log
#   scripts/runpod_launch_lab.sh down # terminate and verify nothing is left
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RUNPOD_API_KEY:?}" "${KAGGLE_USERNAME:?}" "${KAGGLE_KEY:?}"
IMAGE="${LAB_IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"
GPU="${LAB_GPU:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
CLOUD="${LAB_CLOUD:-COMMUNITY}"
MAX_USD="${LAB_MAX_HOURLY_USD:-1.80}"
export LAB_LIFETIME="${LAB_LIFETIME_SECS:-10800}"
NAME="arc3-agent-lab"
STATE="/tmp/arc3-lab-pod-id"
CMDFILE="$(mktemp)"

api() { # method path [body]
  local args=(--silent --show-error --fail-with-body --connect-timeout 10 --max-time 30 -X "$1")
  [ -n "${3:-}" ] && args+=(-H 'content-type: application/json' --data-binary "$3")
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl "${args[@]}" -K - "https://rest.runpod.io/v1$2"
}
redact() { sed -e "s|${RUNPOD_API_KEY}|***|g" -e "s|${KAGGLE_KEY}|***|g"; }
pod_id() { runpodctl pod list 2>/dev/null | jq -r --arg n "$NAME" '.[]? | select(.name==$n) | .id' | head -1; }

case "${1:-}" in
up)
  [ -n "$(pod_id)" ] && { echo "pod already exists: $(pod_id)" >&2; exit 1; }
  cat scripts/lab_obs.sh scripts/runpod_lab_entrypoint.sh >"$CMDFILE"   # one self-contained start command
  bash -n "$CMDFILE" || { echo "start command has a syntax error" >&2; exit 1; }
  body="$(jq -n --rawfile cmd "$CMDFILE"  --arg name "$NAME" --arg img "$IMAGE" --arg gpu "$GPU" --arg cloud "$CLOUD" '{
    name:$name, imageName:$img, computeType:"GPU", gpuTypeIds:[$gpu], gpuCount:1, cloudType:$cloud, interruptible:false,
    containerDiskInGb:220, volumeInGb:0, ports:[],
    env:{KAGGLE_USERNAME:env.KAGGLE_USERNAME, KAGGLE_KEY:env.KAGGLE_KEY, RUNPOD_API_KEY:env.RUNPOD_API_KEY,
         MAX_LIFETIME_SECS:env.LAB_LIFETIME, LAB_LIFETIME_S:env.LAB_LIFETIME},
    dockerStartCmd:["bash","-lc",$cmd]}')"
  resp="$(api POST /pods "$body" 2>&1 | redact)" || { echo "create failed: $(printf '%s' "$resp" | head -c 400)" >&2; exit 1; }
  id="$(printf '%s' "$resp" | jq -r '.id // empty')"
  [ -n "$id" ] || { echo "create returned no id" >&2; exit 1; }
  printf '%s' "$id" >"$STATE"
  info="$(api GET "/pods/$id" 2>&1 | redact || true)"
  cost="$(printf '%s' "$info" | jq -r '.. | (.costPerHr? // empty)' | head -1)"
  echo "created pod $id cost=\$${cost:-?}/hr gpu=$(printf '%s' "$info" | jq -r '.gpu.displayName // .machine.gpuTypeId // empty' | head -1)" >&2
  if [ -n "$cost" ] && awk "BEGIN{exit !($cost > $MAX_USD)}"; then
    api DELETE "/pods/$id" >/dev/null 2>&1 || true; echo "cost \$$cost/hr above cap \$$MAX_USD; terminated" >&2; exit 1
  fi
  echo "$id" ;;
watch) # the driver-side watcher: polls, saves every log snapshot, enforces the cost cap, receives artifacts
  exec python3 scripts/runpod_watch.py --pod-name "$NAME" "${@:2}" ;;
logs) runpodctl pod logs "$(pod_id)" "${@:2}" ;;
down)
  id="$(pod_id)"; [ -z "$id" ] && [ -f "$STATE" ] && id="$(cat "$STATE")"
  [ -n "$id" ] && { api DELETE "/pods/$id" >/dev/null 2>&1 || runpodctl pod delete "$id" >/dev/null 2>&1 || true; }
  rm -f "$STATE"
  left="$(pod_id)"; [ -n "$left" ] && { echo "WARNING: pod $left still present" >&2; exit 1; } || echo "verified: no $NAME pod remains" >&2 ;;
status) runpodctl pod list 2>&1 | jq -r --arg n "$NAME" '.[]? | select(.name==$n) | "\(.id) \(.desiredStatus) \(.costPerHr)/hr"' ;;
*) echo "usage: $0 up|watch|logs|down|status" >&2; exit 2 ;;
esac
