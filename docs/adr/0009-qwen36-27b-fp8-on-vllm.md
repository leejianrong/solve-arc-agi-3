# ADR-0009: Qwen 3.6 27B FP8 on vLLM as the competition baseline

- Status: Accepted
- Date: 2026-09-20
- Deciders: repo owner (assumed default from current evidence)
- Supersedes: ADR-0002, ADR-0007

## Context

The original plan chose Qwen3-30B-A3B in GGUF through llama.cpp/Ollama because
vLLM packaging looked risky and there were no hardware-matched results. That
uncertainty is gone. Tufa Labs won ARC-AGI-3 Milestone 1 with the open-source
Duck harness, Qwen 3.6 27B FP8, and a local vLLM server on Kaggle. The official
starter now exposes an RTX PRO 6000 Blackwell `g4-standard-48` target with 96GB
VRAM and a nine-hour GPU runtime limit.

The existing GGUF notebook did not complete a baseline and targets an older,
text-only model. Qwen 3.6 27B is multimodal, has already been exercised with the
competition's image-plus-text/tool pattern, and its FP8 checkpoint is about
34GB. Reproducing a proven competition stack is a lower-risk starting point than
continuing to debug an unproven one.

## Decision

Use Qwen 3.6 27B FP8 served by a pinned vLLM wheelhouse as the default model and
runtime. Package the exact wheels and weights as Kaggle datasets so scoring needs
no internet. Record GPU model, driver, CUDA, vLLM/model revisions, startup time,
VRAM, prefill/decode throughput, and concurrency in every benchmark.

FP8 is the quality-first submission format. Benchmark NVFP4 or AWQ only as a
fallback if the measured full-run projection misses the internal 8h30m deadline
or the required concurrent contexts do not fit. Do not change quantization on
model-size intuition alone.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Continue Qwen3-30B-A3B GGUF/Ollama | Older model and text-only path; the repo has no completed benchmark, while the replacement is already proven in this competition |
| llama.cpp with a Qwen 3.6 GGUF | Simpler single binary, but loses the proven Duck deployment, vLLM continuous batching, and straightforward multimodal serving |
| NVFP4 from day one | Blackwell supports it and it saves memory, but the likely binding risk is reasoning quality rather than fitting a 34GB FP8 model into 96GB |
| A larger model | Violates the requested 20–30B target and consumes the memory/concurrency margin needed by a long-horizon harness |

## Consequences

The first experiment becomes reproduction and measurement rather than runtime
invention. The project inherits vLLM's compiled-wheel/version constraints and
must pin an offline-compatible wheelhouse. The original notebook and local-model
configuration are historical prototypes, not the implementation starting point.
Fine-tuned checkpoints must ultimately be exportable to a vLLM-supported format,
not GGUF.
