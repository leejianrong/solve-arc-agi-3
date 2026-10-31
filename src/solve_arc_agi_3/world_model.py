"""World-model contract and replay verifier (ADR-0010, card ARC-44).

A world model predicts the next frame from a frame and a move. It is accepted only if it replays the
whole history exactly. Running untrusted model-written code safely is card ARC-51 and is not here:
`verify_replay` takes any in-process object with `predict`. Frames are decoded from the trace encoding
in docs/TRACE.md.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

Grid = tuple[tuple[int, ...], ...]

_HEX = "0123456789abcdef"


@dataclass(frozen=True)
class Move:
    name: str
    x: int | None = None
    y: int | None = None


@dataclass(frozen=True)
class Transition:
    step: int
    before: Grid
    move: Move
    after: Grid
    levels_before: int
    levels_after: int
    state_after: str


History = Sequence[tuple[Grid, Move]]
"""Prior transitions of the current level, oldest first: each item is the frame a move was played from and
that move. The frame the move led to is the next item's frame (or the frame being predicted from)."""


class WorldModel(Protocol):
    """`predict(frame, move)`, optionally `predict(frame, move, history)` for games whose next frame depends
    on hidden state (see `takes_history`)."""

    def predict(self, frame: Grid, move: Move) -> Grid: ...


class GoalModel(WorldModel, Protocol):
    """A world model that can also say whether a frame is a goal state (ADR-0013)."""

    def is_goal(self, frame: Grid) -> bool: ...


def takes_history(model: object) -> bool:
    """True when the model's `predict` wants the optional third `history` argument. A model may set a
    `takes_history` attribute (the sandbox does, from the source); otherwise the signature decides: a third
    positional parameter or `*args`. Never raises."""
    declared = getattr(model, "takes_history", None)
    if isinstance(declared, bool):
        return declared
    try:
        params = list(inspect.signature(model.predict).parameters.values())  # type: ignore[attr-defined]
    except (TypeError, ValueError, AttributeError):
        return False
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3 or any(p.kind is p.VAR_POSITIONAL for p in params)


def call_predict(
    model: WorldModel, frame: Grid, move: Move, history: History | None = None
) -> Grid:
    """`model.predict`, with `history` passed only to models that take it."""
    if history is not None and takes_history(model):
        return model.predict(frame, move, list(history))  # type: ignore[call-arg]
    return model.predict(frame, move)


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    revision: int
    parent: int | None = None


# Frame encoding ------------------------------------------------------------------------------------


def encode_frame(grid: Grid) -> list[str]:
    rows: list[str] = []
    repeats: list[int] = []
    previous: tuple[int, ...] | None = None
    for row in grid:
        if row == previous:
            repeats[-1] += 1
        else:
            rows.append(_encode_row(row))
            repeats.append(0)
        previous = row
    return [r + (f"~{n}" if n else "") for r, n in zip(rows, repeats, strict=True)]


def _encode_row(row: Sequence[int]) -> str:
    tokens: list[str] = []
    i = 0
    while i < len(row):
        j = i
        while j < len(row) and row[j] == row[i]:
            j += 1
        tokens.append(f"{_HEX[row[i]]}{j - i}")
        i = j
    return ".".join(tokens)


_MAX_ROW = 4096
_TOKEN = re.compile(r"[0-9a-f][0-9]{1,4}")


def rows_to_frame(rows: Sequence[str]) -> Grid:
    """Plain hex rows (one hex digit per cell, `"0030"`), as model-written code sees frames, to a Grid."""
    return tuple(tuple(int(c, 16) for c in row) for row in rows)


def frame_to_rows(grid: Grid) -> list[str]:
    """A Grid to plain hex rows; the inverse of `rows_to_frame`. Not the run-length trace encoding."""
    return ["".join(_HEX[c] for c in row) for row in grid]


def decode_frame(encoded: Sequence[str]) -> Grid:
    """Decode run-length rows. `~n` repeats the previous row n more times, whether it is written after the
    row string (`51.363~2`) or as an item of its own (`~2`)."""
    rows: list[tuple[int, ...]] = []
    for item in encoded:
        text, _, repeat = item.partition("~")
        if text:
            rows.append(_decode_row(text))
        if repeat:
            if not rows:
                raise ValueError("repeat before any row")
            rows.extend([rows[-1]] * int(repeat))
    return tuple(rows)


def _decode_row(text: str) -> tuple[int, ...]:
    cells: list[int] = []
    for token in text.split("."):
        if not _TOKEN.fullmatch(token) or int(token[1:]) == 0:
            raise ValueError(_BAD_ROW.format(row=text[:40]))
        cells.extend([_HEX.index(token[0])] * int(token[1:]))
        if len(cells) > _MAX_ROW:
            raise ValueError(_BAD_ROW.format(row=text[:40]))
    return tuple(cells)


_BAD_ROW = (
    "row {row!r} is not run-length encoded. decode_frame takes the trace encoding (hex colour then run "
    "length, joined by '.', e.g. '51.363' for one 5 and 63 threes, '~n' repeats a row); plain hex rows "
    "such as '0030' are a different encoding, use rows_to_frame"
)


def apply_diff(previous: Grid, runs: Iterable[Sequence[int]]) -> Grid:
    """Each run is [y, x0, x1, colour]; x0 to x1 are both included."""
    rows = [list(r) for r in previous]
    for y, x0, x1, colour in runs:
        rows[y][x0 : x1 + 1] = [colour] * (x1 - x0 + 1)
    return tuple(tuple(r) for r in rows)


def decode_trace_frame(frame: Mapping[str, Any], previous: Grid | None) -> Grid:
    if "diff" in frame:
        if previous is None:
            raise ValueError("diff frame without a previous frame")
        return apply_diff(previous, frame["diff"])
    return decode_frame(frame["base"] if "base" in frame else frame["key"])


def transitions_from_trace(records: Iterable[Mapping[str, Any]]) -> list[Transition]:
    """One Transition per step record, starting from the `start` record's frame."""
    out: list[Transition] = []
    frame: Grid | None = None
    levels = 0
    for r in records:
        if r.get("kind") == "start":
            frame = decode_trace_frame(r["frame"], None)
            levels = int(r["levels"])
        elif r.get("kind") == "step":
            if frame is None:
                raise ValueError("step record before the start record")
            after = decode_trace_frame(r["frame"], frame)
            m = r["moves"][0]
            out.append(
                Transition(
                    step=int(r["step"]),
                    before=frame,
                    move=Move(m["name"], m.get("x"), m.get("y")),
                    after=after,
                    levels_before=levels,
                    levels_after=int(r["levels"]),
                    state_after=str(r["state"]),
                )
            )
            frame, levels = after, int(r["levels"])
    return out


# Replay verification ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayReport:
    """exact means every predicted frame equals the observed one cell for cell. Each step is predicted from
    the observed frame before it, so one early error does not hide later ones. A prediction that raises or
    has the wrong shape counts as a mismatch of every cell of the larger frame.

    A model that reports a sandbox error (an object with a `last_error` attribute, as `SandboxedModel`
    has) is rejected at the first such failure: `first_error` names it as `step N: kind: message`,
    `stopped_early` is true and `checked` counts the steps actually run, so a looping model costs one
    timeout, not one per step."""

    exact: bool
    checked: int
    mismatching_steps: tuple[int, ...] = field(default=())
    mismatch_cells: int = 0
    first_error: str = ""
    stopped_early: bool = False

    @property
    def first_mismatch(self) -> int | None:
        return self.mismatching_steps[0] if self.mismatching_steps else None

    @property
    def summary(self) -> str:
        if self.exact:
            return f"exact replay of {self.checked} transitions"
        text = (
            f"{len(self.mismatching_steps)} of {self.checked} transitions differ, first at step "
            f"{self.first_mismatch}, {self.mismatch_cells} cells wrong in total"
        )
        return f"{text}; {self.first_error}" if self.first_error else text


def _cell_diff(predicted: Grid, observed: Grid) -> int:
    if len(predicted) != len(observed) or any(
        len(p) != len(o) for p, o in zip(predicted, observed, strict=True)
    ):
        return max(sum(map(len, predicted)), sum(map(len, observed)))
    return sum(
        p != o
        for pr, orow in zip(predicted, observed, strict=True)
        for p, o in zip(pr, orow, strict=True)
    )


def verify_replay(model: WorldModel, transitions: Sequence[Transition]) -> ReplayReport:
    wrong: list[int] = []
    cells = 0
    first_error = ""
    checked = 0
    stopped = False
    reports_errors = hasattr(model, "last_error")
    wants = takes_history(model)
    past: list[tuple[Grid, Move]] = []
    level: int | None = None
    for t in transitions:
        checked += 1
        if level != t.levels_before:  # history is per level: a new level starts empty
            past, level = [], t.levels_before
        history = list(past) if wants else None
        past.append((t.before, t.move))
        if reports_errors:
            model.last_error = None  # type: ignore[attr-defined]
        sandbox_error = None
        try:
            diff = _cell_diff(call_predict(model, t.before, t.move, history), t.after)
            if reports_errors:
                sandbox_error = model.last_error  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001  untrusted model code: any failure is a mismatch
            diff = sum(map(len, t.after))
            first_error = first_error or f"step {t.step}: {type(exc).__name__}: {exc}"
        if sandbox_error is not None:
            diff = diff or sum(map(len, t.after))
            first_error = first_error or f"step {t.step}: {sandbox_error}"
            stopped = True
        if diff:
            wrong.append(t.step)
            cells += diff
        if stopped:
            break
    return ReplayReport(not wrong, checked, tuple(wrong), cells, first_error, stopped)


class RevisionLog:
    """Accepts a candidate revision only if it replays all history exactly; otherwise the previous
    revision stays active (rollback is simply not switching)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.accepted: list[ModelSpec] = []
        self.rejected: list[tuple[ModelSpec, ReplayReport]] = []
        self._count = 0

    @property
    def active(self) -> ModelSpec | None:
        return self.accepted[-1] if self.accepted else None

    def submit(
        self, source: str, model: WorldModel, transitions: Sequence[Transition]
    ) -> tuple[ModelSpec, ReplayReport]:
        self._count += 1
        parent = self.active.revision if self.active else None
        spec = ModelSpec(
            id=self.model_id, source=source, revision=self._count, parent=parent
        )
        report = verify_replay(model, transitions)
        if report.exact:
            self.accepted.append(spec)
        else:
            self.rejected.append((spec, report))
        return spec, report
