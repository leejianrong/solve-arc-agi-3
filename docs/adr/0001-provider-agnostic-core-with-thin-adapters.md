# ADR-0001: Provider-agnostic agent core with thin dev and Kaggle adapters, fail-soft at the boundary

- Status: Accepted
- Date: 2026-08-31
- Deciders: repo owner

## Context

The harness has to run in two shapes: against a local game client during
development (`play-arc-agi-3`'s Arcade wrapper, or `arc-agi-3-benchmarking`'s
client), and against Kaggle's own submission interface at scoring time, where
there's no internet and a strict wall-clock budget. Kaggle's exact submission
format could still change before the competition's later milestones, and the
local dev tooling is not the same code as the submission tooling. Something
also has to decide what happens when a frame is malformed, or an LLM call
fails or times out mid-game, since one game misbehaving should not be able to
sink an entire scored run.

## Decision

The agent core (state grounding, mechanism discovery, short-horizon
planning) is pure Python with no IO of its own: given a frame and its running
world-model state, it returns an action or short action plan and an updated
state. Two thin adapters call into it identically: a dev adapter over the
local game client, and a Kaggle adapter implementing the competition's
required agent-class interface. The same boundary owns fail-soft handling: a
malformed or unexpected frame (an empty frame after `WIN`/`GAME_OVER` is
normal, not an error) is logged and skipped rather than crashing the loop,
and an LLM call that errors or times out gets a bounded retry, then falls
back to a safe default action rather than aborting the run.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Build directly against Kaggle's submission interface from day one | Couples every line of core logic to a format that could still change, and makes local dev testing depend on faking Kaggle's exact runtime |
| Let each adapter own its own error handling independently | Duplicates the fail-soft policy in two places and risks the dev adapter silently masking a bug that the Kaggle adapter then hits for the first time in a scored run |
| Hard-fail on any unexpected frame or LLM error | Turns one bad frame or one flaky API call into a total loss for the whole game's score, which is worse than continuing with a degraded action |

## Consequences

This buys a bug found against the dev adapter being a bug in the thing that
actually ships, and one place to harden against a scored run getting killed
by a transient failure. It costs an extra layer of indirection for what could
otherwise be one script, and it means the core's data contract (frame in,
action out, state threaded through) has to be nailed down before either
adapter is built, rather than growing organically from whichever adapter
gets written first. It also means fail-soft behavior has to be tested
deliberately (fixture frames simulating malformed input, a stub LLM client
simulating timeouts), since it's exactly the kind of code that's easy to
under-test.

## Update (2026-08-31): the Kaggle interface is now known, not generic

The "competition's required agent-class interface" above was a placeholder
pending confirmation. It's now confirmed via ARC Prize's own docs
(`docs.arcprize.org/create-agent`) and the official
`github.com/arcprize/ARC-AGI-3-Kaggle-Starter` repo:

```python
from .agent import Agent
from .structs import FrameData, GameAction, GameState


class MyAgent(Agent):
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool: ...
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction: ...
```

`GameState` is an enum including at least `NOT_PLAYED`, `GAME_OVER`, `WIN`.
`GameAction` distinguishes simple actions from complex ones needing
coordinates (`action.set_data({"x": ..., "y": ...})`) and carries a
`reasoning` property. The starter kit's own workflow: edit
`agent/my_agent.py` locally, validate with `make play-local` (which runs the
same `arc-agi` PyPI package's game engine locally that the Kaggle gateway
runs, so "if it works locally, it works on Kaggle"), then `make submit`
auto-builds and pushes a notebook.

This changes the Kaggle adapter (S2) from "build a generic agent-class
wrapper" to "vendor `ARC-AGI-3-Kaggle-Starter` and implement its
`Agent.choose_action`/`is_done` against this project's agent core" -- a
concrete, smaller task than originally scoped, and one with a local
validation loop (`make play-local`) that doesn't need a live Kaggle
submission to check. Note also: this starter kit's local execution path
(caches downloaded games under `environment_files/` and runs fully offline
after that) is the same pattern `play-arc-agi-3` already uses, and is a
different code path from `arc-agi-3-benchmarking`, which calls arcprize.org's
*hosted* API over the internet (fine for slice 1's dev-time benchmarking per
ADR-0004, but not the actual Kaggle-submission code path).

One unresolved discrepancy to check directly rather than assume either way:
the starter kit's own accelerator config was seen listing `cpu`, `p100`,
`rtx6000` with `T4` as the stated default, which doesn't cleanly match this
project's earlier-documented `T4x2 / P100 / RTX 6000` list (PLAN.md). Verify
against `ARC-AGI-3-Kaggle-Starter`'s `build_notebook.py` directly before
slice 3, rather than trusting either list blind.
