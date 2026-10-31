#!/usr/bin/env bash
# Rehearses scripts/lab_obs.sh locally with stubs, on a seconds-long clock. Usage: lab_obs_harness.sh <scenario> <workdir>
# Prints what the pod would have printed, then FACT lines the test reads. `terminate` is a stub that only records.
set -uo pipefail
SCENARIO="$1"; T="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$T/bin" "$T/w"
cat >"$T/bin/runpodctl" <<'STUB'
#!/usr/bin/env bash
if [ "$1" = "send" ]; then echo "Sending '$4' on code ${3}-77"; sleep "${STUB_SEND_SECS:-1}"; fi
STUB
cat >"$T/bin/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
echo "1024 MiB, 5 %"
STUB
chmod +x "$T/bin/runpodctl" "$T/bin/nvidia-smi"
export PATH="$T/bin:$PATH"
export OBS_DIR="$T/w" LOG="$T/w/job.log" OBS_HEARTBEAT_SECS=1 OBS_WATCH_SECS=1 OBS_STALL_SECS=300 OBS_SHIP_WAIT_SECS=4
export OBS_EXTRA_LOGS="$T/run-*/server.log"
: >"$LOG"
terminate() { echo "$1" >>"$T/terminated"; }
case "$SCENARIO" in
  stall) export OBS_HEARTBEAT_SECS=12 OBS_STALL_SECS=3 ;;
  heartbeat_prevents_stall) export OBS_HEARTBEAT_SECS=1 OBS_STALL_SECS=3 ;;
esac
export SECRET_SENTINEL="s3cr3t-sentinel-value"
source "$ROOT/scripts/lab_obs.sh"
obs_start
case "$SCENARIO" in
  phase_deadline) phase slow 2 | tee -a "$LOG"; sleep 7 ;;
  fatal_line) phase work 60 | tee -a "$LOG"; sleep 1; echo "RuntimeError: Engine core initialization failed" | tee -a "$LOG"; sleep 6 ;;
  vllm_log_excerpt)
    mkdir -p "$T/run-1"; printf 'INFO starting\nERROR engine core failed to start\nTraceback (most recent call last):\n  File x\nRuntimeError: boom\n' >"$T/run-1/server.log"
    phase serve 60 | tee -a "$LOG"; sleep 1; echo "vLLM exited early (code 1)" | tee -a "$LOG"; sleep 6 ;;
  stall) sleep 7 ;;
  heartbeat_prevents_stall) sleep 6 ;;
  finish) mkdir -p "$T/results"; echo ok >"$T/results/result.json"; export OBS_RESULTS_DIR="$T/results"; phase run 60 | tee -a "$LOG"; obs_finish ;;
  single_action) (obs_fail "first" &); (obs_fail "second" &); sleep 6 ;;
  healthy) phase work 60 | tee -a "$LOG"; sleep 4 ;;
esac
sleep 1
kill $(jobs -p) 2>/dev/null
echo "FACT terminated_count=$(wc -l <"$T/terminated" 2>/dev/null || echo 0)"
echo "FACT terminated_reasons=$(tr '\n' '|' <"$T/terminated" 2>/dev/null)"
echo "FACT has_tgz=$([ -f "$T/w/diag.tgz" ] && echo yes || echo no)"
