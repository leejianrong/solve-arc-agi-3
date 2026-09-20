# RunPod FP8 acceptance evidence -- 2026-09-20

The first passing real-GPU acceptance run for ARC-42, produced by
`scripts/runpod_entrypoint.sh` + `scripts/runpod_benchmark.py` on a rented
RunPod pod. Not a fake-server or OpenRouter run -- see
[`../../qwen36-vllm-artifact.md`](../../qwen36-vllm-artifact.md) for how these
evidence tiers differ and why only this one counts as control/performance
evidence.

## Files

- `report.json` -- the machine-readable `GpuBenchmarkReport` (schema in
  `src/solve_arc_agi_3/gpu_report.py`): deployment provenance, control
  provenance, measured performance, and acceptance flags.
- `job.log` -- the pod's full driver output: Kaggle asset download, offline
  wheelhouse install, preflight, vLLM startup, the real agent-path smoke, and
  the benchmark probes, each marked with a `PHASE=` line.
- `run/server.log` -- vLLM's own stdout/stderr for this run.
- `run/lifecycle.jsonl` -- the owned-process lifecycle events (start, ready,
  terminate, stop) with their timings.
- `run/launch_config.json` -- the exact vLLM command and the full preflight
  report it launched with.
- `run/pid.json` -- the owned vLLM process id.
- `run/smoke/<id>/trace.jsonl` -- the real agent-path smoke trace: the exact
  request sent to the pinned model, its full response (including the
  reasoning trace), the parsed action decision, and the terminal outcome.

## What this proves

Read together with `report.json`'s `acceptance` block:

- `full_asset_preflight_passed: true` -- every pinned wheel and the model's
  `model.safetensors`/`tokenizer.json` were SHA-256 verified against the real
  35.9GB Kaggle snapshot and 5.1GB wheelhouse, not synthetic fixtures.
- `real_smoke_request_passed: true` -- the pinned `vrfai/Qwen3.6-27B-FP8`
  model, served by the pinned vLLM command, produced a request the agent's
  own parser accepted as a legal ARC action, through the same code path used
  by the fake-server smoke.
- `offline_download_attempts: 0` -- vLLM's own log contains no network-fetch
  indicators once the mounted assets were verified.
- `server_exit_code: 0` -- the owned vLLM process shut down cleanly on
  SIGTERM.

Provenance in `report.json.deployment`: GPU `NVIDIA RTX PRO 6000 Blackwell
Server Edition` (the actual planned Kaggle hardware), driver `580.82.07`,
CUDA `12.8`, `vllm==0.19.0`, `torch==2.10.0`, container image pinned by
digest. `report.json.control` ties the run to the exact pinned manifest
(`manifest_sha256`), model revision, and wheelhouse dataset version.

## What it does not prove

One request at concurrency 1. It does not establish throughput under
concurrent load, a stable-context ceiling at the full 65536-token
`max_model_len`, or behavior across the 110-environment competition
simulator -- those are ARC-40 and later slices.
