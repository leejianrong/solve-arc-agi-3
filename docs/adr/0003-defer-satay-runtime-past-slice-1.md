# ADR-0003: Defer satay-runtime past slice 1

- Status: Superseded by ADR-0008
- Date: 2026-08-31
- Deciders: repo owner

## Context

`satay-runtime` (a sibling project, `/home/jianlee/projects/abang-ai/satay-runtime`)
is a local-first durable-execution runtime: it journals every durable call to
SQLite and replays a crashed workflow from the top, and offers `map`/`gather`
fan-out for trying independent branches. It's a plausible fit for crash-safe
resume across a long Kaggle session and for fanning out candidate action
sequences. It also has an async-only core, while a typical Kaggle ML stack
(torch, a generation loop) is normally sync, so wiring it in means bridging
async and sync code somewhere. The riskiest open question for this whole
project is whether the core harness loop (state grounding, mechanism
discovery, short-horizon planning) even works at all and fits the wall-clock
budget, not whether it can resume from a crash.

## Decision

Slice 1 and slice 2 do not use satay-runtime. It's introduced once the core
loop is proven end to end against real games (after slice 2), most likely
alongside the Kaggle submission adapter (slice 4 in SLICES.md), where
crash-safe resume actually starts to matter because a real scored run is on
the line.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Wire satay in from slice 1 | Adds an async/sync bridging cost on top of a harness whose core logic isn't proven yet, which is exactly the wrong order: de-risk the loop first, add durability once there's something worth making durable |
| Never use satay at all | Throws away a real fit (crash-safe resume, fan-out over candidate branches) for no reason other than avoiding the bridging cost; the cost is real but one-time, not ongoing |

## Consequences

This buys the fastest possible path to proving the core loop works, since
slice 1 and slice 2 stay a plain sync script. It costs a retrofit later: the
harness's control flow has to accommodate an async bridge once satay is
introduced, which is more work than writing it async-first would have been.
That cost is accepted deliberately, in exchange for not paying it at all if
the core loop turns out to need major rework once real games are tried. It
also means satay's packaging (vendoring, not `pip install`, for a no-internet
Kaggle run) has to be validated at the point it's introduced, not assumed
solved by then.
