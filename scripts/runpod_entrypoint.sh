#!/usr/bin/env bash
# Pod-side driver for the ARC-42 RunPod FP8 acceptance run.
#
# Bakes in the dead-man's-switch (hard max-lifetime + idle watchdog) required
# by the runpod-jobs skill, then: clones the public repo, downloads the two
# pinned Kaggle datasets, runs scripts/runpod_benchmark.py (offline install,
# full preflight, vLLM serve, real agent-path smoke, benchmark), and relays
# the report + run directory back over the runpodctl file-transfer relay --
# this pod has no public IP. Self-terminates unconditionally when the job
# ends, times out, or goes idle. Expects KAGGLE_USERNAME, KAGGLE_KEY,
# RUNPOD_API_KEY, and CONTAINER_IMAGE in the environment (set via the pod's
# create-body env, not embedded here).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace
cd /workspace

MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-3300}"
IDLE_TIMEOUT_SECS="${IDLE_TIMEOUT_SECS:-900}"
LOG=/workspace/job.log
: >"$LOG"

install_runpodctl() {
  command -v runpodctl >/dev/null 2>&1 && return 0
  tag=$(curl -fsSL https://api.github.com/repos/runpod/runpodctl/releases/latest \
    | grep -m1 '"tag_name"' | cut -d'"' -f4)
  curl -fsSL -o /usr/local/bin/runpodctl \
    "https://github.com/runpod/runpodctl/releases/download/${tag}/runpodctl-linux-amd64"
  chmod +x /usr/local/bin/runpodctl
}

terminate() {
  echo "pod: self-terminating ($1)" | tee -a "$LOG" >&2
  install_runpodctl || true
  runpodctl pod delete "$RUNPOD_POD_ID" 2>/dev/null \
    || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null || true
  kill -TERM -1 2>/dev/null || true
  sleep 5
  kill -KILL -1 2>/dev/null || true
}

(sleep "$MAX_LIFETIME_SECS" && terminate "max-lifetime reached") &
watchdog_pid=$!

main_job() {
  set +e
  set -x
  # build-essential: Triton JIT-compiles CUDA kernels at runtime and needs a
  # C compiler on PATH, or vLLM's engine core fails during profile_run with
  # "Failed to find C compiler".
  apt-get update -qq && apt-get install -y -qq git curl ca-certificates jq build-essential >/dev/null
  install_runpodctl

  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"

  git clone --depth 1 https://github.com/leejianrong/solve-arc-agi-3.git repo
  cd repo || { echo "PHASE=clone_failed"; return 1; }
  # Pin 3.12: the wheelhouse's core wheels (torch, flashinfer, ...) are
  # cp312-tagged and will not install into any other interpreter ABI.
  uv python install 3.12
  uv sync --group dev --python 3.12
  # uv's managed venvs ship without pip; the pinned offline-install command
  # (build_offline_install_command) shells out to "python -m pip" to match
  # the real Kaggle environment, so bootstrap pip here rather than change it.
  uv run python -m ensurepip --upgrade
  uv tool install kaggle --quiet

  mkdir -p /workspace/assets/model /workspace/assets/wheelhouse
  echo "PHASE=kaggle_download_start $(date -u +%FT%TZ)"
  time uv tool run kaggle datasets download \
    -d driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot -p /workspace/assets/model --unzip
  model_dl=$?
  time uv tool run kaggle datasets download \
    -d driessmit1/arc3-vllm-h100-wheelhouse-v3 -p /workspace/assets/wheelhouse --unzip
  wheelhouse_dl=$?
  echo "PHASE=kaggle_download_done $(date -u +%FT%TZ) model_exit=$model_dl wheelhouse_exit=$wheelhouse_dl"
  du -sh /workspace/assets/model /workspace/assets/wheelhouse 2>&1 || true

  run_dir="/workspace/runs/runpod-$(date -u +%Y%m%dT%H%M%SZ)"
  uv run python scripts/runpod_benchmark.py \
    --model-dir /workspace/assets/model \
    --wheelhouse-dir /workspace/assets/wheelhouse \
    --run-dir "$run_dir" \
    --report-out /workspace/report.json \
    --container-image "${CONTAINER_IMAGE:-unknown}"
  bench_exit=$?
  echo "PHASE=benchmark_script_exit code=$bench_exit"

  mkdir -p /workspace/artifact
  cp /workspace/job.log /workspace/artifact/job.log 2>/dev/null
  cp /workspace/report.json /workspace/artifact/report.json 2>/dev/null
  cp -r "$run_dir" /workspace/artifact/run 2>/dev/null
  echo "PHASE=artifact_contents $(find /workspace/artifact -type f 2>/dev/null | tr '\n' ' ')"
  tar czf /workspace/artifact.tar.gz -C /workspace artifact 2>&1

  base="arc42-$RANDOM"
  runpodctl send --code "$base" /workspace/artifact.tar.gz >/tmp/send.log 2>&1 &
  send_pid=$!
  code=""
  for _ in $(seq 1 40); do
    code=$(grep -oE "${base}[-0-9]+" /tmp/send.log 2>/dev/null | head -1)
    [ -n "$code" ] && break
    sleep 0.5
  done
  echo "RP_ARTIFACT_CODE=${code:-send-failed}"
  wait "$send_pid"
  return "$bench_exit"
}

(main_job; echo "PHASE=job_exit code=$?") 2>&1 | tee -a "$LOG" &
job_pid=$!

(
  last_size=0
  last_change=$(date +%s)
  while kill -0 "$job_pid" 2>/dev/null; do
    sleep 60
    size=$(wc -c <"$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$size" != "$last_size" ]; then
      last_size=$size
      last_change=$now
    elif [ $((now - last_change)) -ge "$IDLE_TIMEOUT_SECS" ]; then
      terminate "idle ${IDLE_TIMEOUT_SECS}s"
      break
    fi
  done
) &
watcher_pid=$!

wait "$job_pid"
kill "$watchdog_pid" "$watcher_pid" 2>/dev/null
terminate "job exited"
