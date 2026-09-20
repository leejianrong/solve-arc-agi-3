# Qwen3.6 27B FP8 baseline artifact

Status: local CPU slice implemented; real RunPod FP8 acceptance pending

This is an original clean-room harness for the control pinned in
[`duck-control-v1.json`](../../configs/baselines/duck-control-v1.json). It does
not contain Duck source, prompts, tests, or notebook cells. Provider transport,
the agent loop, official ARC action adapter, and trace schema are separate, so
switching among the fake endpoint, OpenRouter, RunPod, and Kaggle-local vLLM
does not alter prompt or action behavior.

## Evidence by execution environment

| Environment | What it proves | Current evidence |
|---|---|---|
| In-process fake OpenAI endpoint | Request shape, reasoning/tool response parsing, one observation/action/terminal transition, token/timing trace, and run isolation without sockets | Passing CPU test and `solve-arc-agi-3 smoke` |
| OpenRouter | Development-only request formatting against a real hosted model | Catalogue checked 2026-09-20; `qwen/qwen3.6-27b` is listed, but no API key was present for a live request |
| RunPod | Exact pinned FP8 weights and vLLM performance/acceptance | Pending; no GPU was provisioned in the CPU-only implementation session |
| Offline Kaggle mount | Exact model/wheel hashes, no downloads, local server lifecycle | Code and synthetic checksum tests pass; real 35.9 GB mounted assets still need full preflight |

OpenRouter identifies `qwen/qwen3.6-27b` with Hugging Face source
`Qwen/Qwen3.6-27B`. It does not claim the pinned
`vrfai/Qwen3.6-27B-FP8` revision, quantisation, or vLLM settings. Therefore an
OpenRouter run is never control or performance evidence. The live catalogue is
the authoritative availability check:
<https://openrouter.ai/api/v1/models>.

When `OPENROUTER_API_KEY` is available, exercise the same agent loop with:

```sh
solve-arc-agi-3 provider-smoke \
  --config configs/baselines/openrouter-qwen36-27b.dev.json
```

The same command accepts an `InferenceConfig` for a RunPod or local vLLM base
URL. The configuration names the API-key environment variable; it never stores
the credential. A live OpenRouter smoke was not run in this session because
`OPENROUTER_API_KEY` was unset.

## Local deterministic smoke

```sh
make smoke
```

Each invocation creates a fresh UUID directory under `runs/smoke/`. Its
`trace.jsonl` records configuration, observation, exact OpenAI request, model
response with usage and elapsed time, parsed action, environment transition,
and terminal outcome. The fake endpoint is an injected JSON transport: it
speaks the same request/response schema but opens no socket and makes no network
request.

## Offline assets

Both Kaggle inputs are named in the control manifest. Point the preflight at the
mounted snapshot roots:

```sh
solve-arc-agi-3 preflight \
  --model-dir /kaggle/input/vrfai-qwen3-6-27b-fp8-hf-snapshot \
  --wheelhouse-dir /kaggle/input/arc3-vllm-h100-wheelhouse-v3 \
  --full
```

Preflight verifies the in-repo pin for `SHA256SUMS`, its 179-entry count,
the expected checksums in that list, every core runtime wheel on disk, and—when
`--full` is set—the model and tokenizer bytes. A missing or changed file is a
hard failure before a server process is created. Nothing in preflight has a
download path.

The offline pip command can be inspected without executing it:

```sh
solve-arc-agi-3 install-command \
  --model-dir "$MODEL_DIR" \
  --wheelhouse-dir "$WHEELHOUSE_DIR"
```

It always includes `--no-index`, uses only the verified wheelhouse, and installs
its pinned `requirements.lock`.

## vLLM lifecycle

Inspect the complete command:

```sh
solve-arc-agi-3 vllm-command \
  --model-dir "$MODEL_DIR" \
  --wheelhouse-dir "$WHEELHOUSE_DIR"
```

Or start the owned process after preflight:

```sh
solve-arc-agi-3 serve \
  --model-dir "$MODEL_DIR" \
  --wheelhouse-dir "$WHEELHOUSE_DIR" \
  --run-dir "runs/vllm/$(date -u +%Y%m%dT%H%M%SZ)" \
  --full
```

The command preserves the pinned served name, model length, tool parser,
reasoning parser, generation config, prefix cache, and thinking template. In
offline mode it sets Hugging Face, datasets, and Transformers offline flags.
The option spelling was checked against the pinned
[vLLM 0.19 serve reference](https://docs.vllm.ai/en/v0.19.0/cli/serve/) and
[reasoning guide](https://docs.vllm.ai/en/v0.19.0/features/reasoning_outputs/).
The run directory contains launch configuration, preflight report, PID,
combined server log, and lifecycle events. Readiness is the presence of the
pinned served-model name from `/v1/models`. The launcher creates a fresh process
group, retains its process object, and signals only that owned group during
shutdown; timeout escalation cannot target an unrelated PID or group.

## RunPod acceptance record

`solve_arc_agi_3.gpu_report.GpuBenchmarkReport` is the machine-readable report
contract. The first report must be concurrency 1 and must include a
digest-pinned image, exact control revision and launch command, hardware/runtime
versions, load time, peak VRAM, TTFT, prefill/decode throughput, stable context,
request latencies, full asset-preflight outcome, offline download-attempt count,
and real smoke outcome. Do not create a passing report from OpenRouter or fake
server measurements.

RunPod work must begin by loading the `/run` skill and must terminate paid
resources at the end. That skill was not available in this local session, so no
pod was provisioned and no GPU figures are claimed here.
