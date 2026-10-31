"""Submit a model-written world model (Python source) to the revision log (ADR-0010, card ARC-123).

`submit_world_model` ties `sandbox.load` to `RevisionLog.submit`: the source is accepted only if it replays
the whole trace exactly, a rejected candidate's worker is always closed, and at most one sandboxed model
stays open per log (the accepted one; the caller may use it through `SubmitResult.model`).
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from solve_arc_agi_3.sandbox import (
    DEFAULT_TIMEOUT,
    SandboxedModel,
    SandboxError,
    SourceRejected,
    load,
)
from solve_arc_agi_3.world_model import (
    Grid,
    History,
    ModelSpec,
    Move,
    ReplayReport,
    RevisionLog,
    Transition,
    transitions_from_trace,
)

MAX_MESSAGE = 600


@dataclass(frozen=True)
class SubmitResult:
    accepted: bool
    spec: ModelSpec | None = None
    report: ReplayReport | None = None
    model: SandboxedModel | None = None  # only when accepted; the caller owns closing
    reason: str = ""

    def message(self) -> str:
        if self.accepted:
            rev = self.spec.revision if self.spec else "?"
            n = self.report.checked if self.report else 0
            text = f"World model r{rev} replays all {n} steps exactly."
        else:
            text = f"World model rejected: {self.reason}"
        return _cut(text, MAX_MESSAGE)

    def close(self) -> None:
        """Close the sandbox worker; safe to call more than once."""
        if self.model is not None:
            self.model.close()


class ModelHolder:
    """The one open sandboxed model of a RevisionLog."""

    def __init__(self) -> None:
        self.model: SandboxedModel | None = None

    def replace(self, model: SandboxedModel | None) -> None:
        old, self.model = self.model, model
        if old is not None and old is not model:
            old.close()


_HOLDERS: weakref.WeakKeyDictionary[RevisionLog, ModelHolder] = (
    weakref.WeakKeyDictionary()
)


def holder_for(log: RevisionLog) -> ModelHolder:
    holder = _HOLDERS.get(log)
    if holder is None:
        holder = _HOLDERS[log] = ModelHolder()
    return holder


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _Recorder:
    """Wraps the sandbox to remember the first failing step (the report only keeps the last error)."""

    def __init__(self, model: SandboxedModel, steps: list[int]) -> None:
        self.model = model
        self.steps = steps
        self.calls = 0
        self.first: tuple[int, SandboxError] | None = None
        self.last_error: SandboxError | None = (
            None  # read by verify_replay, which then stops early
        )

    @property
    def takes_history(self) -> bool:
        return self.model.takes_history

    def predict(self, frame: Grid, move: Move, history: History | None = None) -> Grid:
        step = self.steps[self.calls] if self.calls < len(self.steps) else -1
        self.calls += 1
        self.model.last_error = None
        out = self.model.predict(frame, move, history)
        err = self.model.last_error
        self.last_error = err
        if err is not None and self.first is None:
            self.first = (step, err)
        return out


def _describe(error: SandboxError, step: int | None) -> str:
    where = f" at step {step}" if step is not None else ""
    msg = error.message
    if error.kind == "timeout":
        return f"timeout{where} ({msg}); avoid unbounded loops"
    if error.kind == "crash":
        return f"crash{where}: {msg}"
    if error.kind == "forbidden":
        return f"forbidden import or builtin ({msg}); allowed modules: math, itertools, collections, functools, copy, heapq, bisect, re"
    if error.kind == "bad_output":
        return f"bad output{where}: {msg}"
    return f"exception{where}: {msg}"


def submit_world_model(
    log: RevisionLog,
    source: str,
    records: Iterable[Mapping[str, Any]],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> SubmitResult:
    """Check `source` against the trace and, if it replays every step exactly, make it the active revision."""
    holder = holder_for(log)
    try:
        transitions: list[Transition] = transitions_from_trace(records)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return SubmitResult(
            False, reason=f"trace could not be read ({type(exc).__name__}: {exc})"
        )
    if not transitions:
        return SubmitResult(
            False,
            reason="no recorded steps yet; play at least one move first, a model that was never checked proves nothing",
        )
    if log.active is not None and source == log.active.source:
        return SubmitResult(
            False,
            spec=log.active,
            reason=f"unchanged: identical to active revision r{log.active.revision}; not re-run",
        )
    try:
        model = load(source, timeout=timeout)
    except SourceRejected as exc:
        return SubmitResult(False, reason=_describe(exc.error, None))
    try:
        recorder = _Recorder(model, [t.step for t in transitions])
        spec, report = log.submit(source, recorder, transitions)
    except BaseException:
        model.close()
        raise
    if report.exact:
        holder.replace(model)
        return SubmitResult(True, spec, report, model, "exact replay")
    model.close()
    if recorder.first is not None:
        step, err = recorder.first
        reason = _describe(err, step)
    else:
        reason = (
            f"{len(report.mismatching_steps)} of {report.checked} steps differ, first mismatch at "
            f"step {report.first_mismatch}, {report.mismatch_cells} cells wrong in total"
        )
    return SubmitResult(False, spec, report, None, reason)
