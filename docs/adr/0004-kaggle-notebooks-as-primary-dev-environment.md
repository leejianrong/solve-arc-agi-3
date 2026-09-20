# ADR-0004: Kaggle notebooks as the primary GPU dev/test environment

- Status: Accepted
- Date: 2026-08-31
- Deciders: repo owner

## Context

This development machine has no GPU (`nvidia-smi` is not present) and 7.8GB
of RAM, which cannot run inference on a 27-30B model even quantized to
4-bit (roughly 15-18GB of weights alone). Every GPU-touching piece of this
project, from the phase-0 latency benchmark through QLoRA fine-tuning, needs
somewhere else to run.

## Decision

Kaggle notebooks are the primary dev and test loop for anything that needs a
GPU. It's free within Kaggle's quota, and it's the literal environment the
final submission is scored in, so a result measured there transfers directly
instead of needing re-validation on a different GPU later. A rented cloud
GPU (RunPod or Vast.ai) is the fallback, used only if Kaggle's weekly quota
starts throttling iteration speed once fine-tuning work is underway.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Rent a cloud GPU (RunPod/Vast.ai) as the primary loop | More control and no weekly quota, but costs money from day one and doesn't match Kaggle's exact GPU/driver/library environment, reintroducing the "works here, breaks on Kaggle" risk this decision exists to avoid |
| Google Colab | Familiar UX, but the same environment-mismatch problem as a rented GPU, without even the cost-control upside of choosing your own instance type |

## Consequences

This buys environment parity with the actual scoring target for free, which
removes an entire class of surprises. It costs iteration speed: Kaggle
notebooks are slower to iterate in than a personal rented instance, and the
free quota is finite, which is exactly the risk flagged in PLAN.md against
slice 6 (fine-tuning). It also means every GPU-dependent slice's build plan
has to specify "run this in a Kaggle notebook," not "run this locally," which
changes how those slices get tested day to day.
