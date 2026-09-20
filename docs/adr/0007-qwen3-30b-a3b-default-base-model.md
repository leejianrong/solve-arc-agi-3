# ADR-0007: Qwen3-30B-A3B as the default base model, pending the phase-0 benchmark

- Status: Superseded by ADR-0009
- Date: 2026-08-31
- Deciders: repo owner

## Context

The project targets a ~27-30B parameter open-weight model. Research turned up
two shapes at that size: dense models (Gemma 3 27B, Qwen2.5-32B) and
Qwen3-30B-A3B, a mixture-of-experts model with only about 3B active
parameters per token. Dense 27-30B QLoRA fine-tuning is tight even at 4-bit;
Qwen3-30B-A3B reportedly fits QLoRA in around 17.5GB, and Unsloth ships
ready-made notebooks for it. Lower active parameters also help the actual
bottleneck this project expects to hit first (inference latency, ADR-0002),
since fewer parameters are evaluated per token regardless of the model's
total size on disk.

## Decision

Qwen3-30B-A3B is the default base model for both the phase-0 benchmark and
the fine-tuning pipeline, unless that benchmark (slice 1) shows a problem
specific to it (most likely: GGUF/llama.cpp support for its MoE architecture
turns out to be immature).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Gemma 3 27B (dense) | Well-supported by Unsloth and llama.cpp today, but a dense model's full parameter count is active on every token, which works against the latency constraint that's expected to be the binding one |
| Qwen2.5-32B (dense) | Same active-parameter cost as Gemma 3 27B, and reportedly tighter on QLoRA VRAM at this size |

## Consequences

This buys a lower per-token inference cost, which matters most for a
harness that may need several LLM calls per game action inside a fixed
wall-clock budget. It costs betting on a newer, less-proven MoE architecture
in the GGUF/llama.cpp toolchain (ADR-0002), which is exactly the interaction
flagged as an open risk in PLAN.md. Slice 1's build plan includes verifying
GGUF conversion and llama.cpp inference for this specific model before
anything else is built on top of it, so this decision gets checked early
rather than discovered broken deep into the project.
