# RunPod concurrency-sweep evidence -- 2026-09-21

The first passing real-GPU evidence for ARC-40 (the concurrency/context-length sweep of the
Kaggle GPU resource envelope), produced by `scripts/runpod_entrypoint.sh` +
`scripts/runpod_concurrency_sweep.py` on a rented RunPod pod. See
[`../../qwen36-vllm-artifact.md`](../../qwen36-vllm-artifact.md) for how this evidence tier
relates to ARC-42's acceptance run.

**GPU caveat, read first:** this run is on an **NVIDIA A100-SXM4-80GB**, not the RTX PRO 6000
Blackwell Server Edition that ARC-42 validated and that is the actual planned Kaggle hardware.
The Blackwell GPU hit `500 "no instances currently available"` at provisioning time, and its
listed price (~$1.69-2.09/hr) was above the originally-approved cap; the repo owner explicitly
chose to proceed on A100 (~$1.59/hr) rather than raise the cap and keep retrying Blackwell. The
numbers below are **directional evidence about the concurrency-scaling shape of this serving
stack**, not a direct measurement of the Kaggle target's envelope -- a Blackwell run remains
open. Where a number is likely GPU-independent (peak VRAM footprint for a fixed model/KV-cache
config) that's noted; where it isn't (raw throughput, which differs by GPU compute/memory
bandwidth), that's noted too.

## Files

- `evidence/concurrency-{1,4,8,16}.json` -- one machine-readable `ConcurrencyBenchmarkReport`
  per level (schema in `src/solve_arc_agi_3/gpu_report.py`): deployment provenance, control
  provenance, and aggregate throughput/latency/VRAM/stable-context metrics for that level.
- `job.log` -- the pod's full driver output: Kaggle asset download, offline wheelhouse install,
  preflight, vLLM startup, and the sweep's own `PHASE=` markers for each level.
- `run/server.log` -- vLLM's own stdout/stderr for the whole sweep (one server, all four levels).
- `run/lifecycle.jsonl` -- the owned-process lifecycle events (start, ready, terminate, stop).
- `run/launch_config.json` -- the exact vLLM command and the preflight report it launched with.
- `run/smoke/concurrency-<N>/<run-id>/trace.jsonl` -- one real agent-episode trace per concurrent
  request at level N (same schema as ARC-42's smoke trace): exact request, full response
  including reasoning, parsed action, and terminal outcome. This is the real evidence path --
  every concurrent "request" in this sweep is a full agent episode through
  `BaselineAgentCore`/`OpenAICompatibleClient` (`smoke.run_provider_smoke`), not a raw HTTP probe.

## What this proves

Read together with each level's `metrics` block:

- **All four levels passed with 100% success:** 1/1, 4/4, 8/8, and 16/16 concurrent agent
  episodes each reached the fixture's `WIN` terminal state on the pinned FP8 model/vLLM stack,
  with `server_exit_code: 0` (clean shutdown) at the end of the whole sweep.
- **Aggregate decode throughput scales with concurrency** on this stack: 30.44 -> 107.61 ->
  140.23 -> 354.13 completion tokens/sec (aggregate across all concurrent episodes at that
  level) as concurrency goes 1 -> 4 -> 8 -> 16. Per-request latency grows more slowly than
  concurrency (e.g. median ~10-18s per full episode across all levels), consistent with vLLM's
  continuous batching doing real work rather than serializing requests.
- **Peak VRAM stayed flat at ~72.3GB (77,620,838,400 bytes) across every level**, comfortably
  under the A100's 80GB. Because this reflects the model weights + KV cache for a fixed
  `max_model_len`/`tensor_parallel_size` config rather than anything A100-specific, it's a
  reasonable (not exact) proxy for headroom on the 96GB Blackwell card too -- 16 concurrent
  25k-token contexts did not visibly grow peak VRAM beyond the model's own footprint in this run.
- **Model load: 272.2s; clean shutdown: 1.8s** -- both fast enough to be a small fraction of any
  nine-hour budget.
- Provenance: `vllm==0.19.0`, `torch==2.10.0`, CUDA 12.8, same pinned model revision
  (`vrfai/Qwen3.6-27B-FP8`, `076636763143ca2b7cf0d66e16a09c2a0a689dfa`) and manifest hash as
  ARC-42's acceptance run -- this is the same control, just a different GPU.

## What this does not prove

- **Not the Blackwell envelope.** See the caveat above. A100 and the RTX PRO 6000 Blackwell
  differ in VRAM (80GB vs 96GB), memory bandwidth, and compute profile; Blackwell also needed the
  FlashInfer JIT-compilation path ARC-42 had to fix, which doesn't apply to A100. Throughput
  numbers here should not be used as the final Kaggle projection without a Blackwell run.
- **`stable_context_tokens: 25010` is a probe ceiling, not a measured failure point, at every
  level.** `runpod_probes.measure_stable_context_tokens` (unchanged from ARC-42) only tries
  {25000, 12000, 4000}-word prompts and returns the first that succeeds; it was never designed to
  search for where a request actually fails. Getting the identical value (25010 tokens) at
  concurrency 1, 4, 8, and 16 -- on two different GPUs, now that ARC-42 also measured 25010 on
  Blackwell at concurrency 1 -- confirms this is the probe's own ceiling, not evidence that
  concurrent KV-cache pressure has no effect on the real supportable context length. Finding the
  actual ceiling (especially how it shrinks under concurrency) needs a probe that pushes past
  25k tokens until a request actually fails or truncates, which is unbuilt follow-up work.
- **Not the smallest context length that preserves behavior.** SLICES.md's V1 plan also calls for
  sweeping down to "the smallest context lengths that preserve behavior"; this run only measured
  the upper end (does more concurrency break things at the pinned `max_model_len=65536`), not a
  downward context sweep. Still open.
- **Not the 110-environment competition simulator.** This sweep used the same one-step fixture
  game as ARC-42's smoke (deterministic single-action WIN), fired concurrently -- it is evidence
  about serving-stack concurrency, not about real multi-turn ARC-AGI-3 game behavior under load.

## Toward a nine-hour budget (illustrative, not a commitment)

Using the measured concurrency-16 aggregate decode throughput (354.13 tokens/sec) against the
plan's 7h20m (26,400s) game-work window from `docs/PLAN.md`: ~9.35M decodable tokens total across
16 concurrent environments in that window, or ~584K tokens/environment if divided evenly for the
whole window. That is far above the plan's current per-environment guard of 4,096 generated
tokens x 12 model turns = 49,152 tokens/environment -- so at this measured throughput and this
concurrency, **token budget is not the binding constraint**; per-turn latency (the ~9-18s median
observed here per full episode, itself dominated by the model's own reasoning-token generation
time, not infra overhead) and the number of environments that must actually run concurrently
(110 in the full competition vs. 16 tested here) are the more likely limits. This projection is
illustrative given the A100-vs-Blackwell caveat above and the single-fixture-game workload; it is
not a substitute for measuring the real target hardware and real multi-turn games.

## Infra fixes from this run (real, not simulated)

Provisioning this run surfaced two real bugs, both now fixed and documented in the
`runpod-jobs` skill (`~/.claude/skills/runpod-jobs/`) and this project's memory:

- A hand-rolled `dockerStartCmd` that fetched `runpod_entrypoint.sh` via `curl` before executing
  it left an unprotected step in front of the pod's own dead-man's-switch; if that fetch hangs,
  the watchdog never arms. Fixed by embedding the entrypoint script's content directly into
  `dockerStartCmd` instead of fetching it. A first attempt on this exact fix path burned ~$2.77
  over ~100 minutes with no artifact and no recoverable logs before this was caught and fixed.
- `containerDiskInGb: 60` was too small: `kaggle datasets download --unzip` keeps both the
  downloaded `.zip` and the extracted files, so the 35.9GB model + 5.1GB wheelhouse pair needs
  ~82GB for that step alone. Failed with `OSError: [Errno 28] No space left on device`; fixed by
  raising to `220`.

Total session cost across all three provisioning attempts (two failed, one passing): RunPod
balance went from $13.94 to $10.02 (~$3.92). RunPod resources were terminated after every
attempt, verified via `runpodctl pod list` (and `runpodctl network-volume list`, which was
already empty -- no volumes were created).
