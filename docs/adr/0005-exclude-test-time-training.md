# ADR-0005: Exclude test-time training from this project

- Status: Accepted
- Date: 2026-08-31
- Deciders: repo owner

## Context

Test-time training (fine-tuning a per-task adapter at inference time on a
task's own examples plus augmentations) is exactly how the 2024 ARC-AGI-1
Kaggle winners (MindsAI, "the ARChitects") won, and NVIDIA's own NVARC team
used a version of it (synthetic data plus TTT on a 4B model) to win ARC Prize
2025 on ARC-AGI-2. The earlier research writeup for this project (see
`docs/research/arc-agi-3-solve-strategy.md`) budgeted an ARC-AGI-3 analog
(fine-tuning a LoRA adapter in-session on a game's own unfolding trajectory)
as a plausible R&D experiment, on the reasoning that TTT's track record on
the two earlier benchmarks was strong enough to be worth a spike.

## Decision

Test-time training, in any form, is out of scope for this project. This
includes in-session weight adaptation on a game's own trajectory. This is an
explicit exclusion on the repo owner's own judgment (stated gut feel that
it's a dead end for this problem), not a resourcing or difficulty call, and
it overrides the research writeup's "budget it as R&D" framing.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Keep it as a stretch-goal R&D spike, per the original research writeup | Explicitly rejected by the repo owner; recorded here so a future planning pass doesn't quietly reintroduce it on the strength of the ARC-AGI-1/ARC-AGI-2 precedent alone |

## Consequences

This buys a smaller, more focused project: no separate R&D track chasing an
unproven adaptation of a technique that was proven on a structurally
different problem (static labeled input/output pairs per task, not an
unfolding trajectory of frames and actions). It costs giving up a technique
with a real track record on the two prior ARC benchmarks, on the chance that
the interactive setting breaks the analogy badly enough that it wouldn't have
worked anyway. If this project stalls on harness-only approaches and TTT
looks worth revisiting, that has to come from the repo owner raising it
again, not from an agent reintroducing it because the research once flagged
it as promising.
