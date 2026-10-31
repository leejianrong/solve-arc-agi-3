"""World-model contract and replay verifier (ADR-0010, card ARC-44).

A world model predicts the next frame from a frame and a move. It is accepted only if it replays the
whole history exactly. Running untrusted model-written code safely is card ARC-51 and is not here:
`verify_replay` takes any in-process object with `predict`. Frames are decoded from the trace encoding
in docs/TRACE.md.
"""

from __future__ import annotations

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


class WorldModel(Protocol):
    def predict(self, frame: Grid, move: Move) -> Grid: ...


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
        cells.extend([_HEX.index(token[0])] * int(token[1:]))
    return tuple(cells)


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
    has the wrong shape counts as a mismatch of every cell of the larger frame."""

    exact: bool
    checked: int
    mismatching_steps: tuple[int, ...] = field(default=())
    mismatch_cells: int = 0
    first_error: str = ""

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
    for t in transitions:
        try:
            diff = _cell_diff(model.predict(t.before, t.move), t.after)
        except Exception as exc:  # noqa: BLE001  untrusted model code: any failure is a mismatch
            diff = sum(map(len, t.after))
            first_error = first_error or f"step {t.step}: {type(exc).__name__}: {exc}"
        if diff:
            wrong.append(t.step)
            cells += diff
    return ReplayReport(not wrong, len(transitions), tuple(wrong), cells, first_error)


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
