# ADR-0013: World-model interface and planning

- Status: Accepted
- Date: 2026-10-31
- Deciders: repo owner (assumed default; settles the interface question left open by ADR-0010)

## Context

ADR-0010 listed five methods for executable models (`parse`, `transition`, `render`, `terminal`,
`legal_actions`). The code that exists (`world_model.py`, `sandbox.py`) has one: `predict(frame, move)`,
and exact replay of recorded history needs nothing more. A planner needs a goal test, and a 27B model
writing five coordinated methods fails more often than one writing a single function.

## Decision

`predict(frame, move) -> frame` stays the only required function. Model-written source may also define,
both optional:

- `is_goal(frame) -> bool`: true for a frame that completes the level. `GoalModel` (predict + is_goal) is
  the Protocol; `SandboxedModel.is_goal` calls it (False plus `last_error` when missing or failing).
- `legal_moves(frame) -> list[dict]`: moves such as `{"name": "ACTION6", "x": 3, "y": 4}` available from a
  frame. When absent, the planner uses the moves it is given. When present it overrides them per state.

`parse`, `render`, `terminal` and `legal_actions` are dropped. Frames cross the sandbox as plain hex rows
(`rows_to_frame` / `frame_to_rows`), never the run-length trace encoding (`decode_frame` now rejects hex
rows with an error naming both).

A plan is a tuple of moves plus the frame predicted after each one (`PlanResult`). It is found entirely
inside the model, never by touching the game.

Search (`planner.plan`) is breadth-first by default, so the first plan found is a shortest one. Options:
`heuristic` alone gives best-first on depth + heuristic, `beam_width` keeps the best W states per depth.
States are de-duplicated by frame. Budgets, all checked before every model call: `max_nodes` (counts
`predict` calls; default 2000), `max_depth` (30), `time_budget_s` (10). A single slow call can overrun
the time budget by its own sandbox timeout. Reasons: `found`, `no plan within depth` (depth or beam cut
something off), `exhausted` (whole reachable space searched, goal unreachable under the model),
`node budget`, `time budget`, `model failed` (exception, a non-frame or wrong-shape result, or a sandbox
`last_error`).

Execution (`PlanExecutor`) plays one real move at a time and compares each actual frame with the
prediction. The first mismatch aborts the plan permanently and yields a report (plan step, cells
differing, bounding box); the agent replans from the actual frame or feeds the counterexample back to the
model. The executor never executes anything.

Replay verification gains two rules: a sandbox error is reported in `ReplayReport.first_error`
(`step N: kind: message`), and replay stops at the first sandbox failure (`stopped_early`, `checked`
counts steps run), so a looping model costs one timeout instead of one per step.

## Consequences

Models stay small and replay-checkable. A model without `is_goal` can still be replayed but not planned
with unless the caller supplies a goal test (for example "levels_completed went up" learned from the
trace). Searching through a sandbox costs one IPC round trip per node, so default budgets are modest and
`legal_moves` doubles the calls per expansion. Optimal-path guarantees hold only for breadth-first and an
admissible heuristic.
