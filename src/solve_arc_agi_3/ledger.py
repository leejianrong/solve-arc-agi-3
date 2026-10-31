"""Evidence-linked hypothesis ledger (ADR-0010, card ARC-45).

Claims keep their evidence: every entry points at a trace step and says whether it supports or
counters the claim, so disagreement is preserved instead of overwritten. Many entries are computed from
the trace (`Ledger.from_trace`), the rest are written by the model through `propose` and `observe`.
Status is derived from the evidence tally by `_status`; only `retire` is a manual decision.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Kind = Literal["control", "mechanic", "goal", "counter", "other"]
Status = Literal["open", "supported", "refuted", "retired"]
Polarity = Literal["support", "counter"]
Origin = Literal["computed", "model"]

SUPPORTS_TO_CONFIRM = (
    3  # consistent supports (and no counter) before a claim counts as supported
)
COUNTERS_TO_REFUTE = (
    1  # a counterexample refutes once it is at least as many as the supports
)
NOTE_CHARS = 120
LAST_MOVES = 5  # moves recorded as goal evidence when a level is cleared

_MOTION_CLAIM = re.compile(
    r"(\w+) moves the object \(colour (\d+), size (\d+)\) by \((-?\d+), (-?\d+)\)"
)  # the statement `_motion` writes

GOAL_STATEMENT = "A level is cleared by a short move sequence; the evidence lists the moves before each clear"

_STATUS_RANK: dict[Status, int] = {
    "supported": 0,
    "open": 1,
    "refuted": 2,
    "retired": 3,
}


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    polarity: Polarity
    note: str = ""


class Hypothesis(BaseModel):
    id: str
    kind: Kind
    statement: str
    prediction: str | None = None
    origin: Origin = "model"
    step: int = 0  # step at which the claim was introduced
    status: Status = "open"
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def supports(self) -> int:
        return sum(e.polarity == "support" for e in self.evidence)

    @property
    def counters(self) -> int:
        return sum(e.polarity == "counter" for e in self.evidence)


def _status(h: Hypothesis) -> Status:
    """Rules: retired is sticky. Refuted when counters >= COUNTERS_TO_REFUTE and counters >= supports.
    Supported when supports >= SUPPORTS_TO_CONFIRM and no counter. Everything else (including a claim
    with more supports than counters, which stays contested) is open. Never a one-way door: more
    supports can lift a refuted claim back to open."""
    if h.status == "retired":
        return "retired"
    if h.counters >= COUNTERS_TO_REFUTE and h.counters >= h.supports:
        return "refuted"
    if h.supports >= SUPPORTS_TO_CONFIRM and h.counters == 0:
        return "supported"
    return "open"


def _key(kind: Kind, statement: str) -> tuple[str, str]:
    return kind, " ".join(statement.lower().split())


class Ledger(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)

    def get(self, hypothesis_id: str) -> Hypothesis:
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        raise KeyError(hypothesis_id)

    def propose(
        self,
        statement: str,
        kind: Kind,
        step: int,
        prediction: str | None = None,
        *,
        origin: Origin = "model",
    ) -> str:
        """Add a claim and return its id; an equal statement of the same kind returns the existing id."""
        key = _key(kind, statement)
        for h in self.hypotheses:
            if _key(h.kind, h.statement) == key:
                if h.prediction is None:
                    h.prediction = prediction
                return h.id
        hypothesis_id = f"h{len(self.hypotheses) + 1}"
        self.hypotheses.append(
            Hypothesis(
                id=hypothesis_id,
                kind=kind,
                statement=statement.strip(),
                prediction=prediction,
                origin=origin,
                step=step,
            )
        )
        return hypothesis_id

    def observe(
        self, hypothesis_id: str, step: int, polarity: Polarity, note: str = ""
    ) -> None:
        """Attach evidence; the same (step, polarity) pair is only counted once."""
        h = self.get(hypothesis_id)
        if any(e.step == step and e.polarity == polarity for e in h.evidence):
            return
        h.evidence.append(
            Evidence(step=step, polarity=polarity, note=note[:NOTE_CHARS])
        )
        h.status = _status(h)

    def retire(self, hypothesis_id: str) -> None:
        self.get(hypothesis_id).status = "retired"

    def merge(self, other: Ledger) -> None:
        """Fold another ledger in: equal statements share one entry and pool their evidence."""
        for h in other.hypotheses:
            mine = self.get(
                self.propose(h.statement, h.kind, h.step, h.prediction, origin=h.origin)
            )
            for e in h.evidence:
                self.observe(mine.id, e.step, e.polarity, e.note)
            if h.status == "retired":
                mine.status = "retired"

    def to_json(self) -> str:
        return self.model_dump_json(indent=1)

    @classmethod
    def from_json(cls, text: str) -> Ledger:
        return cls.model_validate_json(text)

    def render(self, budget_chars: int) -> str:
        """Compact prompt block, never longer than budget_chars. Supported claims come first, then open,
        then refuted (so the model does not retry them); within a status, more evidence first. Retired
        claims are left out. A line that does not fit is skipped and shorter ones may still fill the gap."""
        live = [h for h in self.hypotheses if h.status != "retired"]
        live.sort(key=lambda h: (_STATUS_RANK[h.status], -len(h.evidence), h.id))
        lines: list[str] = []
        used = 0
        skipped = 0
        for h in live:
            line = _line(h)
            cost = len(line) + (1 if lines else 0)
            if used + cost > budget_chars:
                skipped += 1
                continue
            lines.append(line)
            used += cost
        footer = f"(+{skipped} more)"
        if skipped and used + len(footer) + 1 <= budget_chars:
            lines.append(footer)
        return "\n".join(lines)

    def expected_motion(self, action: str) -> list[tuple[int, int, int, int]]:
        """Supported computed claims that `action` moves an object: sorted (colour, size, dx, dy)."""
        found: set[tuple[int, int, int, int]] = set()
        for h in self.hypotheses:
            if h.origin != "computed" or h.status != "supported":
                continue
            m = _MOTION_CLAIM.fullmatch(h.statement)
            if m and m.group(1) == action:
                found.add(tuple(int(g) for g in m.groups()[1:]))  # type: ignore[arg-type]
        return sorted(found)

    @classmethod
    def from_trace(cls, records: Iterable[Mapping[str, Any]]) -> Ledger:
        """Computed hypotheses from trace records (docs/TRACE.md). Clicks (a move with x, y) are left
        out of the per-action rules because the same action means different things at different points."""
        steps: list[_Step] = []
        levels = 0
        for r in records:
            if r.get("kind") == "start":
                levels = int(r.get("levels", 0))
            elif r.get("kind") == "step":
                steps.append(_Step(r, levels))
                levels = int(r.get("levels", levels))
        ledger = cls()
        _motion(ledger, steps)
        _unchanged(ledger, steps)
        _counters(ledger, steps)
        _goal(ledger, steps)
        return ledger


def _line(h: Hypothesis) -> str:
    text = f"{h.id} {h.status} {h.kind} +{h.supports}/-{h.counters}: {h.statement}"
    if h.prediction:
        text += f" -> {h.prediction}"
    if (
        h.kind == "goal"
    ):  # the moves before the latest clear are the useful part of a goal claim
        supports = [e for e in h.evidence if e.polarity == "support"]
        if supports:
            text += f" [{supports[-1].note}]"
    counters = [e for e in h.evidence if e.polarity == "counter"]
    if counters:
        text += f" [counter at step {counters[-1].step}: {counters[-1].note}]"
    return text


class _Step:
    def __init__(self, record: Mapping[str, Any], levels_before: int) -> None:
        self.record = record
        self.step = int(record["step"])
        self.levels_before = levels_before
        self.levels_after = int(record.get("levels", levels_before))
        move = record["moves"][0]
        self.move = (
            f"{move['name']}({move['x']},{move['y']})"
            if "x" in move
            else str(move["name"])
        )
        self.action = None if "x" in move else str(move["name"])


def _motion(ledger: Ledger, steps: list[_Step]) -> None:
    seen: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for s in steps:
        if s.action:
            seen[s.action].update(
                (m["color"], m["size"]) for m in s.record.get("motion", [])
            )
    for action in sorted(seen):
        played = [s for s in steps if s.action == action]
        for color, size in sorted(seen[action]):
            deltas_by_step: dict[int, set[tuple[int, int]]] = {}
            for s in played:
                deltas_by_step[s.step] = {
                    (m["to"][0] - m["from"][0], m["to"][1] - m["from"][1])
                    for m in s.record.get("motion", [])
                    if (m["color"], m["size"]) == (color, size)
                }
            for dx, dy in sorted(set().union(*deltas_by_step.values())):
                hid = ledger.propose(
                    f"{action} moves the object (colour {color}, size {size}) by ({dx}, {dy})",
                    "mechanic",
                    played[0].step,
                    f"after {action} the object is shifted by ({dx}, {dy})",
                    origin="computed",
                )
                for s in played:
                    got = deltas_by_step[s.step]
                    if (dx, dy) in got:
                        ledger.observe(hid, s.step, "support", f"moved by ({dx}, {dy})")
                    else:
                        ledger.observe(
                            hid,
                            s.step,
                            "counter",
                            f"moved by {sorted(got)}" if got else "did not move",
                        )


def _unchanged(ledger: Ledger, steps: list[_Step]) -> None:
    groups: dict[tuple[str, int], list[_Step]] = defaultdict(list)
    for s in steps:
        if s.action and "change" in s.record:
            groups[(s.action, s.levels_before)].append(s)
    for (action, level), group in sorted(groups.items()):
        if not any(s.record["change"]["n"] == 0 for s in group):
            continue
        hid = ledger.propose(
            f"{action} changes nothing at level {level}",
            "mechanic",
            group[0].step,
            "no cell changes",
            origin="computed",
        )
        for s in group:
            n = int(s.record["change"]["n"])
            if n == 0:
                ledger.observe(hid, s.step, "support", "no change")
            else:
                ledger.observe(hid, s.step, "counter", f"changed {n} cells")


def _counters(ledger: Ledger, steps: list[_Step]) -> None:
    for s in steps:
        for text in s.record.get("counters", []):
            hid = ledger.propose(
                f"Counter or timer: {text}", "counter", s.step, origin="computed"
            )
            ledger.observe(hid, s.step, "support", "flagged")


def _goal(ledger: Ledger, steps: list[_Step]) -> None:
    for i, s in enumerate(steps):
        if s.levels_after > s.levels_before:
            hid = ledger.propose(GOAL_STATEMENT, "goal", s.step, origin="computed")
            last = ", ".join(t.move for t in steps[max(0, i - LAST_MOVES + 1) : i + 1])
            ledger.observe(
                hid, s.step, "support", f"level {s.levels_after} reached after {last}"
            )
