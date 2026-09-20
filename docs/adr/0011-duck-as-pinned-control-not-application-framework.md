# ADR-0011: Use Duck as a pinned control and component source, not the application framework

- Status: Accepted
- Date: 2026-09-20
- Deciders: repo owner (assumed default)

## Context

Duck is the only published Milestone-winning, Kaggle-sized LLM harness found. It
already contains vLLM packaging, connected-component segmentation, an ephemeral
Python tool, structured traces, paired significance checks, a competition Arcade
simulator, and a Kaggle notebook. Ignoring it would duplicate solved work.

The existing project has a provider-agnostic core, an official starter adapter
decision, and sibling trace formats. Adopting Duck's TAAF framework wholesale
would make the new hypothesis/verifier experiments harder to isolate and would
couple the project to a much larger framework on a ten-day milestone schedule.

## Decision

Pin a Duck revision and reproduce it unchanged as the behavioral control. Its
`ARC3-Inference/pyproject.toml` declares the MIT license classifier, but no root
`LICENSE` file was available at the expected repository path during this review.
Do not copy or adapt Duck source until an actual distribution license is verified.
Until then, reimplement documented behavior—offline vLLM setup, segmentation,
sandbox patterns, the 110-environment simulator, and paired evaluation—behind this
repository's own contracts. Use the official Kaggle starter as the submission
boundary.

Every candidate report compares against the pinned control under the same model,
quantization, prompt budget, seeds/passes, game set, hardware, and wall-clock cap.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Ignore Duck and build from scratch | Wastes the only directly relevant open competition baseline and repeats packaging risk |
| Fork Duck/TAAF as the whole project | Fast initially, but makes the experimental core and deployment framework inseparable and imports more surface than needed |
| Compare only against the official one-call-per-action harness | Too weak to identify whether a new idea beats the current open-weight competition baseline |

## Consequences

Reproduction becomes slice one. License evidence, notices, and upstream revision
hashes must travel with any reused code. If the Duck license remains ambiguous,
the control may run from its published artifact for comparison while the shipped
agent uses an independent implementation. Reports must distinguish upstream
behavior, clean reimplementation, and this project's changes.
