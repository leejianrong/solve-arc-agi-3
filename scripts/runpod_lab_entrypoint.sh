#!/usr/bin/env bash
# Pod-side driver for the ARC-AGI-3 agent lab (kaggle/run/lab_worker.py). Embedded as the CONTENT of the pod's
# dockerStartCmd by scripts/runpod_launch_lab.sh; never fetched at boot.
#
# Cost safety (runpod-jobs skill, Rule 0): the max-lifetime guard is armed as the FIRST action, before any
# network or package step, and terminate() needs nothing that can hang (the runpod/pytorch image ships
# runpodctl; the self-install fallback is wrapped in `timeout`). An idle watchdog follows. Expects
# KAGGLE_USERNAME, KAGGLE_KEY and RUNPOD_API_KEY in the pod env (set from the launcher by reference).
set -uo pipefail
mkdir -p /workspace
# lab_obs.sh (phases, heartbeats, fatal-line and stall detection, evidence shipping) is prepended by the launcher.
# When this file is run on its own, source it from next to it.
type obs_start >/dev/null 2>&1 || . "$(dirname "$0")/lab_obs.sh"
LOG=/workspace/job.log
: >"$LOG"
MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-10800}"
IDLE_TIMEOUT_SECS="${IDLE_TIMEOUT_SECS:-1500}"

terminate() {
  echo "pod: self-terminating ($1)" | tee -a "$LOG" >&2
  if ! command -v runpodctl >/dev/null 2>&1; then
    timeout 60 curl -fsSL -o /usr/local/bin/runpodctl \
      https://github.com/runpod/runpodctl/releases/download/v2.14.0/runpodctl-linux-amd64 \
      && chmod +x /usr/local/bin/runpodctl
  fi
  timeout 60 runpodctl pod delete "$RUNPOD_POD_ID" 2>/dev/null || timeout 60 runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
  kill -TERM -1 2>/dev/null || true
  sleep 5
  kill -KILL -1 2>/dev/null || true
}

( sleep "$MAX_LIFETIME_SECS" && terminate "max-lifetime ${MAX_LIFETIME_SECS}s" ) &
watchdog_pid=$!

main_job() {
  set +e
  phase boot 300
  # Boot self-check: fail in seconds, not at the lifetime limit, if the pod is unusable.
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  [ -n "$gpu_name" ] || echo "BOOT_CHECK_FAILED: no GPU visible"
  free_gb="$(df -Pk /workspace | awk 'NR==2{printf "%.0f", $4/1048576}')"
  [ "${free_gb:-0}" -ge 120 ] || echo "BOOT_CHECK_FAILED: only ${free_gb}GB free disk (need 120)"
  timeout 20 curl -fsS -o /dev/null https://www.kaggle.com || echo "BOOT_CHECK_FAILED: kaggle.com unreachable"
  [ -n "${KAGGLE_KEY:-}" ] || echo "BOOT_CHECK_FAILED: KAGGLE_KEY missing in pod env"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python3 --version; nvcc --version | tail -2; which -a python3.12 uv curl runpodctl
  export DEBIAN_FRONTEND=noninteractive
  phase apt 1500
  timeout 600 apt-get update -qq && timeout 900 apt-get install -y -qq ninja-build unzip jq >/dev/null
  export PATH="/root/.local/bin:/usr/local/cuda/bin:$PATH"
  command -v uv >/dev/null 2>&1 || timeout 300 sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  timeout 300 uv tool install kaggle --quiet
  # Parallel kaggle invocations race to create ~/.config/kaggle (FileExistsError killed one download): create it
  # and make the first call serially.
  mkdir -p /root/.config/kaggle && kaggle --version >/dev/null 2>&1
  export UV_LINK_MODE=copy
  mkdir -p /kaggle/input/datasets/driessmit1 /kaggle/input/datasets/mirzamilanfarabi /kaggle/input/competitions/arc-prize-2026-arc-agi-3
  phase download 3300
  dl() {  # slug dest : download then unzip and drop the zip (peak disk is about 2x the dataset)
    timeout 3000 kaggle datasets download -d "$1" -p "$2" --unzip -q
    find "$2" -maxdepth 1 -name '*.zip' -delete
  }
  dl driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot /kaggle/input/datasets/driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot &
  dl mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3 /kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3 &
  timeout 1800 kaggle competitions download arc-prize-2026-arc-agi-3 -p /tmp/comp -q \
    && unzip -qo /tmp/comp/*.zip -d /kaggle/input/competitions/arc-prize-2026-arc-agi-3 && rm -rf /tmp/comp &
  dl leejianrong/solve-arc-agi-3-src /kaggle/input/datasets/leejianrong/solve-arc-agi-3-src &
  wait
  for d in driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3 leejianrong/solve-arc-agi-3-src; do
    [ -n "$(ls -A "/kaggle/input/datasets/$d" 2>/dev/null)" ] || echo "BOOT_CHECK_FAILED: dataset $d is empty after download"
  done
  du -sh /kaggle/input/datasets/* /kaggle/input/competitions/* 2>&1 | head
  phase venv 900
  # Python the agents run on: 3.12 venv with the competition's own wheels (arc_agi) plus pydantic/yaml/requests.
  uv venv --python 3.12 /opt/agent-venv --quiet
  uv pip install --python /opt/agent-venv/bin/python --quiet pydantic pyyaml requests python-dotenv \
    --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi
  /opt/agent-venv/bin/python -c "import arc_agi,arcengine,pydantic;print('agent venv ok')"
  phase lab 7200   # the worker narrows this itself (vllm / experiment) through $OBS_DIR/phase
  ARC3_PY312="$(command -v python3.12 || command -v python3)" \
    python3 /kaggle/input/datasets/leejianrong/solve-arc-agi-3-src/run/lab_worker.py
  rc=$?
  echo "PHASE=lab_exit code=$rc"
  [ "$rc" -eq 0 ] && obs_finish || obs_fail "lab worker exited with code $rc"
}

export OBS_RESULTS_DIR="${OBS_RESULTS_DIR:-/workspace/out}"
obs_start
(main_job 2>&1 | tee -a "$LOG") &
job_pid=$!

(
  last_size=0
  last_change=$(date +%s)
  while kill -0 "$job_pid" 2>/dev/null; do
    sleep 60
    size=$(wc -c <"$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$size" != "$last_size" ]; then
      last_size=$size; last_change=$now
    elif [ $((now - last_change)) -ge "$IDLE_TIMEOUT_SECS" ]; then
      terminate "idle ${IDLE_TIMEOUT_SECS}s"; break
    fi
  done
) &
watcher_pid=$!

wait "$job_pid"
sleep 60   # keep the pod (and its readable logs) briefly after the job ends; the guard still runs
kill "$watchdog_pid" "$watcher_pid" 2>/dev/null
terminate "job exited"
