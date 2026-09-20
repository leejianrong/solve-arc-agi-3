# ADR-0008: Exclude satay-runtime from this project

- Status: Accepted
- Date: 2026-08-31
- Deciders: repo owner
- Supersedes: ADR-0003

## Context

ADR-0003 deferred `satay-runtime` past slice 1, planning to introduce it
later (originally slice 4) for crash-safe resume and fan-out over candidate
action branches. Before slice 1 started, the repo owner decided to drop it
from the project entirely rather than defer it.

## Decision

`satay-runtime` is not part of this project, at any slice. If crash-safe
resume or fan-out over candidate action branches turns out to be genuinely
needed later, that need gets met with whatever the harness core or Kaggle
adapter implements directly (plain local checkpointing, or nothing, if the
need doesn't materialize), not by reintroducing satay.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Keep ADR-0003's plan (introduce satay once the core loop is proven) | Rejected on the repo owner's own call, not for a new technical reason found since ADR-0003. Recorded here so a future planning pass doesn't reintroduce it on the strength of ADR-0003's original reasoning alone |

## Consequences

This buys one fewer dependency to vendor and package for a no-internet
Kaggle run, and removes the async/sync bridging cost ADR-0003 had already
flagged as a real cost. It costs giving up a working crash-safe-resume and
fan-out primitive that fit the problem reasonably well on paper. If a scored
run's lack of resume becomes a real, observed problem (for example, a
long-running submission failing partway with no way to pick back up), that
observation is the trigger to revisit this decision, not a re-read of
ADR-0003's original fit assessment.
