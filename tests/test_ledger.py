from __future__ import annotations

from typing import Any

from solve_arc_agi_3.ledger import (
    COUNTERS_TO_REFUTE,
    SUPPORTS_TO_CONFIRM,
    Ledger,
)


def test_status_transitions_follow_thresholds() -> None:
    ledger = Ledger()
    hid = ledger.propose("ACTION1 opens the door", "mechanic", 0)
    assert hid == "h1" and ledger.get(hid).status == "open"
    for i in range(SUPPORTS_TO_CONFIRM - 1):
        ledger.observe(hid, i, "support")
    assert ledger.get(hid).status == "open"
    ledger.observe(hid, 10, "support")
    assert ledger.get(hid).status == "supported"
    ledger.observe(hid, 11, "counter", "door stayed shut")
    assert ledger.get(hid).status == "open"  # contested: more supports than counters
    for i in range(20, 20 + SUPPORTS_TO_CONFIRM + 1):
        ledger.observe(hid, i, "counter")
    assert ledger.get(hid).status == "refuted"
    ledger.retire(hid)
    ledger.observe(hid, 99, "support")
    assert ledger.get(hid).status == "retired"


def test_single_counterexample_refutes_an_unproven_claim() -> None:
    assert COUNTERS_TO_REFUTE == 1
    ledger = Ledger()
    hid = ledger.propose("goal is red", "goal", 0)
    ledger.observe(hid, 3, "counter", "reached red, nothing happened")
    assert ledger.get(hid).status == "refuted"


def test_evidence_is_counted_once_per_step_and_polarity_and_notes_are_short() -> None:
    ledger = Ledger()
    hid = ledger.propose("x", "other", 0)
    ledger.observe(hid, 1, "support", "a" * 500)
    ledger.observe(hid, 1, "support", "again")
    ledger.observe(hid, 1, "counter")
    h = ledger.get(hid)
    assert (h.supports, h.counters) == (1, 1) and len(h.evidence[0].note) < 200


def test_dedupe_and_merge() -> None:
    ledger = Ledger()
    a = ledger.propose("ACTION1  moves Left", "control", 0)
    b = ledger.propose("action1 moves left", "control", 5, prediction="x decreases")
    c = ledger.propose("action1 moves left", "mechanic", 6)
    assert a == b and a != c and ledger.get(a).prediction == "x decreases"
    other = Ledger()
    o = other.propose("ACTION1 moves left", "control", 2)
    other.observe(o, 2, "support", "saw it")
    other.propose("new claim", "other", 3)
    ledger.merge(other)
    assert len(ledger.hypotheses) == 3
    assert ledger.get(a).supports == 1 and ledger.get("h3").statement == "new claim"


def test_json_round_trip_is_deterministic() -> None:
    ledger = Ledger()
    hid = ledger.propose("a", "goal", 0, "p")
    ledger.observe(hid, 1, "support", "n")
    ledger.propose("b", "counter", 2, origin="computed")
    text = ledger.to_json()
    again = Ledger.from_json(text)
    assert again == ledger and again.to_json() == text


def test_render_never_exceeds_budget_and_orders_by_value() -> None:
    ledger = Ledger()
    weak = ledger.propose("weak open claim", "other", 0)
    refuted = ledger.propose("refuted claim", "other", 0)
    strong = ledger.propose("strong claim", "mechanic", 0)
    retired = ledger.propose("retired claim", "other", 0)
    ledger.observe(refuted, 1, "counter", "no")
    for i in range(SUPPORTS_TO_CONFIRM):
        ledger.observe(strong, i, "support")
    ledger.retire(retired)
    full = ledger.render(10_000)
    lines = full.splitlines()
    assert [ln.split()[0] for ln in lines] == [strong, weak, refuted]
    assert "retired" not in full and "[counter at step 1: no]" in full
    for budget in range(len(full) + 5):
        out = ledger.render(budget)
        assert len(out) <= budget
    short = ledger.render(len(lines[0]) + 14)
    assert short.startswith(lines[0]) and "(+2 more)" in short


def rec(step: int, name: str, **kw: Any) -> dict[str, Any]:
    move: dict[str, Any] = {"name": name}
    base: dict[str, Any] = {
        "kind": "step",
        "step": step,
        "moves": [move],
        "state": "NOT_FINISHED",
        "levels": 0,
        "change": {"n": 10},
        "motion": [],
        "counters": [],
    }
    base.update(kw)
    return base


def moved(dx: int) -> list[dict[str, Any]]:
    return [{"color": 5, "size": 9, "from": [10, 10], "to": [10 + dx, 10]}]


def test_from_trace_derives_motion_nochange_counter_and_goal() -> None:
    strip = "color 8 size 20 x0-19 y60-61 center (9,60)"
    records = [
        {"kind": "start", "levels": 0},
        rec(0, "ACTION3", motion=moved(5), counters=[strip]),
        rec(1, "ACTION3", motion=moved(5), counters=[strip]),
        rec(2, "ACTION4", change={"n": 0}),
        rec(3, "ACTION3", motion=moved(5), counters=[strip]),
        rec(4, "ACTION4", change={"n": 0}),
        rec(
            5, "ACTION3", motion=[], change={"n": 4}
        ),  # blocked: counter to the move claim
        rec(6, "ACTION4", change={"n": 0}, levels=1),
        {"kind": "unknown"},
    ]
    ledger = Ledger.from_trace(records)
    by = {h.statement: h for h in ledger.hypotheses}
    move = by["ACTION3 moves the object (colour 5, size 9) by (5, 0)"]
    assert (
        (move.supports, move.counters) == (3, 1)
        and move.origin == "computed"
        and move.status == "open"
    )
    assert move.evidence[-1].note == "did not move"
    none = by["ACTION4 changes nothing at level 0"]
    assert (none.supports, none.counters, none.status) == (3, 0, "supported")
    assert by["Counter or timer: " + strip].supports == 3
    assert by["Counter or timer: " + strip].status == "supported"
    goal = next(h for h in ledger.hypotheses if h.kind == "goal")
    assert goal.evidence[0].step == 6
    assert "ACTION3, ACTION4" in goal.evidence[0].note and goal.evidence[
        0
    ].note.startswith("level 1 reached")
    assert Ledger.from_trace(records) == ledger  # deterministic


def test_from_trace_ignores_clicks_and_conflicting_deltas_are_both_kept() -> None:
    records = [
        {"kind": "start", "levels": 2},
        rec(0, "ACTION1", motion=moved(5)),
        rec(1, "ACTION1", motion=moved(-5)),
        {
            **rec(2, "ACTION6", change={"n": 0}),
            "moves": [{"name": "ACTION6", "x": 1, "y": 2}],
        },
    ]
    ledger = Ledger.from_trace(records)
    texts = [h.statement for h in ledger.hypotheses]
    assert len(texts) == 2 and all("ACTION1 moves" in t for t in texts)
    assert all(
        h.status == "refuted" for h in ledger.hypotheses
    )  # 1 support, 1 counter each
    mine = Ledger()
    mine.propose("ACTION1 moves the object (colour 5, size 9) by (5, 0)", "mechanic", 0)
    mine.merge(ledger)
    assert len(mine.hypotheses) == 2 and mine.get("h1").supports == 1


def _motion_step(
    step: int, action: str, delta: tuple[int, int] | None
) -> dict[str, Any]:
    motion = (
        [{"color": 5, "size": 4, "from": [2, 5], "to": [2 + delta[0], 5 + delta[1]]}]
        if delta
        else []
    )
    return {
        "kind": "step",
        "step": step,
        "moves": [{"name": action}],
        "levels": 0,
        "change": {"n": 4 if delta else 0},
        "motion": motion,
        "counters": [],
    }


def test_expected_motion_lists_only_supported_computed_deltas() -> None:
    records = [_motion_step(i, "ACTION3", (5, 0)) for i in range(2)]
    assert (
        Ledger.from_trace(records).expected_motion("ACTION3") == []
    )  # two supports are not enough
    records.append(_motion_step(2, "ACTION3", (5, 0)))
    records.append(_motion_step(3, "ACTION1", (0, -1)))
    ledger = Ledger.from_trace(records)
    assert ledger.expected_motion("ACTION3") == [(5, 4, 5, 0)]
    assert ledger.expected_motion("ACTION1") == []
    assert ledger.expected_motion("ACTION2") == []
    records.append(
        _motion_step(4, "ACTION3", None)
    )  # a counterexample ends "supported"
    assert Ledger.from_trace(records).expected_motion("ACTION3") == []


def test_expected_motion_ignores_model_claims() -> None:
    ledger = Ledger()
    hid = ledger.propose(
        "ACTION3 moves the object (colour 5, size 4) by (5, 0)", "mechanic", 0
    )
    for i in range(3):
        ledger.observe(hid, i, "support")
    assert ledger.get(hid).status == "supported"
    assert ledger.expected_motion("ACTION3") == []
