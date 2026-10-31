"""Bounded search over a world model, and the executor that follows a plan (ADR-0013, card ARC-46).

`plan` searches with the model's `predict` only, so it never touches the game. Every budget is checked
before each model call: `max_nodes` counts `predict` calls (the expensive thing when the model is a
sandbox round trip), `max_depth` is the longest plan, `time_budget_s` is wall clock. One slow call can
overrun the time budget by at most that call's own timeout. `PlanExecutor` holds the result and compares
each real frame with the predicted one; it never executes a move itself, the agent loop does.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from solve_arc_agi_3.world_model import (
    Grid,
    History,
    Move,
    WorldModel,
    call_predict,
    takes_history,
)

Reason = Literal[
    "found",
    "no plan within depth",
    "exhausted",
    "node budget",
    "time budget",
    "model failed",
]


@dataclass(frozen=True)
class PlanResult:
    found: bool
    moves: tuple[Move, ...]
    frames: tuple[Grid, ...]  # predicted frame after each move
    reason: Reason
    nodes: int  # predict calls made
    elapsed_s: float = 0.0
    detail: str = ""


class _ModelFailed(Exception):
    pass


def _check_error(model: object) -> None:
    err = getattr(model, "last_error", None)
    if err is not None:
        raise _ModelFailed(str(err))


def _is_frame(out: object, like: Grid) -> bool:
    return (
        isinstance(out, tuple)
        and len(out) == len(like)
        and len(out) > 0
        and all(isinstance(r, tuple) and len(r) == len(like[0]) for r in out)
    )


def plan(
    model: WorldModel,
    frame: Grid,
    moves: Sequence[Move],
    is_goal: Callable[[Grid], bool] | None = None,
    *,
    max_nodes: int = 2000,
    max_depth: int = 30,
    time_budget_s: float = 10.0,
    beam_width: int | None = None,
    heuristic: Callable[[Grid], float] | None = None,
    clock: Callable[[], float] = time.monotonic,
    history: History = (),
) -> PlanResult:
    """Find a move sequence from `frame` to a goal frame.

    Without `beam_width` or `heuristic` this is breadth-first search: the first plan found is a shortest
    one. With `heuristic` alone it is best-first on depth + heuristic (shortest if the heuristic never
    overestimates). With `beam_width` each depth keeps only the best `width` new states (by heuristic, or
    in discovery order without one): fast, not complete. States are de-duplicated by frame, so a frame
    reached first by a longer path is not revisited.

    `is_goal` defaults to the model's own `is_goal` when it has one. When the model defines `legal_moves`
    (and `has_legal_moves` is not false) it overrides `moves` per state. A model that raises, returns
    something that is not a frame of the input shape, or reports a `last_error` ends the search with
    reason "model failed".

    `history` is the real prior transitions of the current level (oldest first); a model whose `predict`
    takes a third argument gets it extended with the predicted steps along the path to each node."""
    start = clock()
    goal = is_goal if is_goal is not None else getattr(model, "is_goal", None)
    if goal is None:
        raise ValueError("no goal test: pass is_goal or use a model that defines it")
    model_moves = getattr(model, "legal_moves", None)
    if not callable(model_moves) or getattr(model, "has_legal_moves", True) is False:
        model_moves = None
    if hasattr(model, "last_error"):
        model.last_error = None

    wants_history = takes_history(model)

    def rollout(node: int) -> list[tuple[Grid, Move]]:
        steps: list[tuple[Grid, Move]] = []
        while node > 0:
            steps.append((frames[parent[node]], move_of[node]))  # type: ignore[arg-type]
            node = parent[node]
        steps.reverse()
        return [*history, *steps]

    nodes = 0
    frames: list[Grid] = [frame]
    parent: list[int] = [-1]
    move_of: list[Move | None] = [None]
    depth: list[int] = [0]
    seen = {frame}
    counter = 0

    def result(reason: Reason, node: int = -1, detail: str = "") -> PlanResult:
        path: list[int] = []
        while node > 0:
            path.append(node)
            node = parent[node]
        path.reverse()
        ms = tuple(m for i in path if (m := move_of[i]) is not None)
        return PlanResult(
            reason == "found",
            ms,
            tuple(frames[i] for i in path),
            reason,
            nodes,
            clock() - start,
            detail,
        )

    def ask_goal(f: Grid) -> bool:
        try:
            value = bool(goal(f))
        except Exception as exc:
            raise _ModelFailed(f"is_goal {type(exc).__name__}: {exc}") from exc
        _check_error(model)
        return value

    try:
        if ask_goal(frame):
            return result("found", 0)
        queue: deque[int] = deque([0])
        heap: list[tuple[float, int, int]] = []
        if heuristic is not None and beam_width is None:
            heapq.heappush(heap, (float(heuristic(frame)), 0, 0))
        level: list[int] = [0]  # beam mode
        cut_by_depth = False
        pruned = False
        while True:
            if beam_width is not None:
                if not level:
                    break
                batch, level = level, []
                nxt: list[int] = []
            else:
                if heap:
                    node = heapq.heappop(heap)[2]
                elif heuristic is None and queue:
                    node = queue.popleft()
                else:
                    break
                batch = [node]
                nxt = []
            for node in batch:
                if depth[node] >= max_depth:
                    cut_by_depth = True
                    continue
                if model_moves is not None:
                    try:
                        options = list(model_moves(frames[node]))
                    except Exception as exc:
                        raise _ModelFailed(
                            f"legal_moves {type(exc).__name__}: {exc}"
                        ) from exc
                    _check_error(model)
                else:
                    options = list(moves)
                for mv in options:
                    if nodes >= max_nodes:
                        return result("node budget")
                    if clock() - start >= time_budget_s:
                        return result("time budget")
                    nodes += 1
                    try:
                        out: Any = call_predict(
                            model,
                            frames[node],
                            mv,
                            rollout(node) if wants_history else None,
                        )
                    except Exception as exc:
                        raise _ModelFailed(
                            f"predict {type(exc).__name__}: {exc}"
                        ) from exc
                    _check_error(model)
                    if not _is_frame(out, frames[node]):
                        raise _ModelFailed(
                            "predict did not return a frame of the input shape"
                        )
                    if out in seen:
                        continue
                    seen.add(out)
                    frames.append(out)
                    parent.append(node)
                    move_of.append(mv)
                    depth.append(depth[node] + 1)
                    child = len(frames) - 1
                    if ask_goal(out):
                        return result("found", child)
                    if heuristic is not None and beam_width is None:
                        counter += 1
                        h = float(heuristic(out))
                        heapq.heappush(heap, (depth[child] + h, counter, child))
                    elif beam_width is not None:
                        nxt.append(child)
                    else:
                        queue.append(child)
            if beam_width is not None:
                if heuristic is not None:
                    nxt.sort(key=lambda i: (float(heuristic(frames[i])), i))
                if len(nxt) > beam_width:
                    pruned = True
                level = nxt[:beam_width]
        if cut_by_depth or pruned:
            return result("no plan within depth")
        return result("exhausted")
    except _ModelFailed as exc:
        return result("model failed", detail=str(exc)[:300])


# Executor ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MismatchReport:
    step: int  # 0-based index of the plan move whose result differed
    move: Move
    cells_differing: int
    bbox: (
        tuple[int, int, int, int] | None
    )  # (y0, x0, y1, x1) inclusive; None on a shape mismatch
    shape_mismatch: bool = False

    def text(self) -> str:
        if self.shape_mismatch:
            return f"plan step {self.step} ({self.move.name}): frame shape differs from the prediction"
        y0, x0, y1, x1 = self.bbox or (0, 0, 0, 0)
        return (
            f"plan step {self.step} ({self.move.name}): {self.cells_differing} cells differ from the "
            f"prediction, within rows {y0}-{y1}, columns {x0}-{x1}"
        )


@dataclass(frozen=True)
class ExecutorStatus:
    state: Literal["continue", "done", "aborted"]
    step: int  # number of plan moves confirmed so far
    report: MismatchReport | None = None


def compare_frames(
    predicted: Grid, actual: Grid
) -> tuple[int, tuple[int, int, int, int] | None, bool]:
    """(cells differing, bounding box (y0, x0, y1, x1), shape mismatch)."""
    if len(predicted) != len(actual) or any(
        len(p) != len(a) for p, a in zip(predicted, actual, strict=True)
    ):
        return max(sum(map(len, predicted)), sum(map(len, actual))), None, True
    ys: list[int] = []
    xs: list[int] = []
    for y, (pr, ar) in enumerate(zip(predicted, actual, strict=True)):
        for x, (p, a) in enumerate(zip(pr, ar, strict=True)):
            if p != a:
                ys.append(y)
                xs.append(x)
    if not ys:
        return 0, None, False
    return len(ys), (min(ys), min(xs), max(ys), max(xs)), False


class PlanExecutor:
    """Follows a found plan one real move at a time. Loop: `m = ex.next_move()`, play `m`, then
    `ex.observe(frame_after)`; stop when the status is not "continue". The first frame that differs from
    the prediction aborts the plan for good (replan from the actual frame). Callers should not call
    `observe` for a move that completed a level: the frame then legitimately differs."""

    def __init__(self, result: PlanResult) -> None:
        if not result.found:
            raise ValueError("cannot execute a plan that was not found")
        self.moves = result.moves
        self.frames = result.frames
        self.confirmed = 0
        self.aborted: MismatchReport | None = None

    @property
    def finished(self) -> bool:
        return self.aborted is not None or self.confirmed >= len(self.moves)

    def next_move(self) -> Move | None:
        return None if self.finished else self.moves[self.confirmed]

    def observe(self, actual: Grid) -> ExecutorStatus:
        if self.aborted is not None:
            return ExecutorStatus("aborted", self.confirmed, self.aborted)
        if self.confirmed >= len(self.moves):
            return ExecutorStatus("done", self.confirmed)
        cells, bbox, shape = compare_frames(self.frames[self.confirmed], actual)
        if cells:
            self.aborted = MismatchReport(
                self.confirmed, self.moves[self.confirmed], cells, bbox, shape
            )
            return ExecutorStatus("aborted", self.confirmed, self.aborted)
        self.confirmed += 1
        if self.confirmed >= len(self.moves):
            return ExecutorStatus("done", self.confirmed)
        return ExecutorStatus("continue", self.confirmed)
