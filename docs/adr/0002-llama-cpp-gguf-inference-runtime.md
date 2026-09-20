# ADR-0002: llama.cpp/GGUF as the v1 inference runtime

- Status: Superseded by ADR-0009
- Date: 2026-08-31
- Deciders: repo owner

## Context

The harness needs one LLM call (at least) per game action, inside a Kaggle
submission that has no internet access at scoring time and runs on whatever
GPU Kaggle assigns (T4x2, P100, or an RTX 6000 Ada, per the ARC-AGI-3 2026
rules). No verified hardware-matched throughput numbers for a 27-30B model on
any of those GPUs turned up in research. vLLM and TensorRT-LLM are generally
faster once a quantized model is fully resident in VRAM, but both typically
need CUDA-toolchain-matched compiled kernels or Python extensions, which is
risky to get right blind on a no-internet target where a build mismatch
can't be fixed after the fact.

## Decision

Serve the quantized base model against a GGUF checkpoint on the llama.cpp
engine for the v1 harness, rather than vLLM or TensorRT-LLM. Fine-tuned
checkpoints (post slice 4) get exported to GGUF as part of the training
pipeline, not as an afterthought. See the 2026-08-31 update below for which
concrete tool serves that GGUF file on Kaggle: it isn't llama.cpp's own
release binaries, and the first choice tried (`llama-cpp-python`) failed in
practice.

## Alternatives considered

| Option | Why not |
|--------|---------|
| vLLM + AWQ | Faster once resident in VRAM, but the packaging risk (compiled kernels, CUDA/driver matching) is worse for a no-internet target than the throughput upside is worth for a first working version |
| TensorRT-LLM + FP8 | This is what the winning AIMO Kaggle solution used, and it's the fastest option researched, but it has the heaviest setup and packaging burden of the three; revisit only if GGUF/llama.cpp throughput turns out to be the binding constraint |

## Consequences

This buys the simplest possible path to "does a quantized 27-30B model even
run inside the Kaggle submission's constraints," which is exactly the
question slice 1 exists to answer. It costs raw throughput compared to
vLLM/TensorRT-LLM, so if the phase-0 benchmark (slice 1) shows llama.cpp is
itself the bottleneck rather than the model or the harness logic, this
decision gets revisited, not defended. It also means GGUF conversion has to
work for whatever base model gets chosen (ADR-0007) before that model can be
used at all, which is flagged as an open risk in PLAN.md rather than assumed
to just work.

## Update (2026-08-31): the concrete tool is Ollama, not `llama-cpp-python`

Slice 1's first attempt used `llama-cpp-python` (Python bindings over
llama.cpp) and its pip install failed on Kaggle trying to build from source
(`error: failed-wheel-build-for-install`). Checked directly against
`ggml-org/llama.cpp`'s own GitHub releases: **there is no prebuilt CUDA
binary for Linux** in that project's release assets (CUDA builds are
published for Windows only; Linux gets CPU, Vulkan, ROCm, SYCL, and a few
others, but not CUDA). `llama-cpp-python` falling back to a from-source CUDA
build on Kaggle was therefore expected, not a fluke, and from-source CUDA
builds are exactly the fragile, slow, version-matching-dependent path this
ADR's original reasoning wanted to avoid.

**Ollama** is the fix: it bundles the same GGUF/llama.cpp-family inference
engine as a single static binary with CUDA support already compiled in
(`curl -fsSL https://ollama.com/install.sh | sh`, no build step, no matching
Kaggle's CUDA toolkit version by hand), pulls GGUF checkpoints directly from
Hugging Face (`ollama pull hf.co/<repo>:<quant>`), and exposes the same
OpenAI-compatible `/v1/chat/completions` endpoint this harness already needs
(ADR-0001's adapter boundary doesn't change). This is a well-documented,
widely-used pattern specifically on Kaggle notebooks (multiple public Kaggle
notebooks run Ollama this way), not a novel workaround. It still uses GGUF
and still avoids vLLM/TensorRT-LLM's heavier packaging burden, so the
decision this ADR records is unchanged; only the specific serving tool is
corrected. `configs/model_configs.local.yaml` and
`notebooks/slice1_phase0_baseline.ipynb` were updated to match.

## Update (2026-08-31, same day): Ollama's own installer script is unreliable in-notebook

Ollama's documented `curl -fsSL https://ollama.com/install.sh | sh` installer
was tried first and failed silently: no error surfaced, but the `ollama`
binary was not on `PATH` afterward (`FileNotFoundError` on
`subprocess.Popen(["ollama", "serve"])`). Checked directly against the
installer script's own source (`ollama/ollama`'s `scripts/install.sh`): it
should place the binary at `/usr/local/bin/ollama` in the normal case, so the
likely cause is the ~1.4GB download failing partway inside the piped shell
script without the failure propagating visibly through a notebook `!` cell,
not a fundamentally wrong install location.

Rather than keep debugging a shell script running inside someone else's
piped-curl pattern, the notebook now downloads the release tarball
(`ollama-linux-amd64.tar.zst` from `ollama/ollama`'s GitHub releases)
directly, extracts it with Python's `zstandard` + `tarfile` (no dependency on
a system `zstd`/`unzstd` CLI, which also isn't guaranteed present), and calls
the binary by its absolute path (`/usr/local/bin/ollama`) rather than through
`PATH` at all. Each step is now its own visible cell, so a failure shows up
where it happens instead of downstream as a confusing `FileNotFoundError`.
