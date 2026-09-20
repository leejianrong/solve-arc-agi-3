# ADR-0012: Gate fine-tuning on measured harness failures

- Status: Accepted
- Date: 2026-09-20
- Deciders: repo owner (assumed default)

## Context

The original roadmap made synthetic game generation, frontier-teacher traces, and
QLoRA a required path. Ten days remain before Milestone 2, no local GPU exists,
and no baseline or training corpus has been measured. The strongest recent
ARC-AGI-3 improvements came from memory, context, tools, and verification without
changing model weights. Trace SFT can also teach a student's format while failing
to teach recovery from its own states.

Kaggle permits publicly available external data, but the ARC-AGI-3 technical
report treats training on demonstration games or many close lookalikes as
domain-specific benchmark optimization. Regardless of eligibility, training
before failures are classified gives no principled target.

## Decision

Fine-tuning, synthetic-game generation, and teacher distillation are not required
for the September submission. After the harness is measured, cluster failures
into transferable skills. Only run a small QLoRA pilot when one cluster is both
common and plausibly trainable, using synthetic or teacher-labeled examples that
do not encode solutions to public demonstration games.

Admit an adapter only if paired held-out evaluation improves level depth or RHAE,
does not materially regress other games, and still meets the projected full-run
budget. Keep the base-model path available for every ablation and submission.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Build the full data pipeline before the harness | Delays the first submission and trains against hypothetical rather than observed weaknesses |
| Distill complete frontier playthroughs | Expensive, vulnerable to exposure bias, and likely to reproduce a much stronger model's policy rather than teach recoverable subskills |
| Never fine-tune | Throws away a useful lever for narrow, repeated tool-use or patching failures once evidence exists |

## Consequences

The September work stays focused on a valid, measured harness. Training moves to
an evidence gate in October and may never happen. If it does, the data pipeline is
smaller and skill-specific, while the cost is less time for large-scale dataset
experimentation before the November final.
