# AGENTS.md

## WHY

Building a custom agent harness to solve ARC-AGI-3 (arcprize.org) within Kaggle's
compute constraints, using an open-weight ~27-30B model rather than a frontier
closed model. `docs/PLAN.md` is the agreed plan and `docs/SLICES.md` tracks what's
built versus planned; trust those, and the code, over any stale prose here.

Sibling repos: `../play-arc-agi-3` (local client for playing ARC-AGI-3 games, with
a recording/trace format already designed), `../arc-agi-3-benchmarking` (the
official arcprize multi-provider benchmarking harness), and `../../arc-agi-1` (an
unrelated prior project: an RL/GP agent for the older static-grid ARC-AGI
benchmark). Check the first two for prior art and existing schemas (game
interface, action space, recording format) before designing anew.

## HOW

See `README.md` for install/lint/type/test commands. Run `make check` before
considering any change done.

## Subagent policy

Cap subagent/research-agent fan-out at **3 running at once**, loosely following
the same instinct as `../../arc-agi-1/CLAUDE.md`'s "Research subagent policy" —
keep parallel research/work focused and reviewable rather than fanned out into
results nobody reads.
