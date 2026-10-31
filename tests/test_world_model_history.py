"""predict(frame, move, history): the optional third argument (ADR-0013 addendum)."""

from __future__ import annotations

from typing import Any

from solve_arc_agi_3.model_tool import submit_world_model
from solve_arc_agi_3.planner import plan
from solve_arc_agi_3.sandbox import load
from solve_arc_agi_3.step_trace import encode_rows
from solve_arc_agi_3.world_model import (
    Grid,
    Move,
    Transition,
    frame_to_rows,
    takes_history,
    verify_replay,
)

SIZE = 12
START_LEN = 10
MV = Move("ACTION1")


def counter(n: int) -> Grid:
    rows = [[0] * SIZE for _ in range(SIZE)]
    rows[0][:n] = [9] * n
    return tuple(tuple(r) for r in rows)


def length(frame: Grid) -> int:
    return sum(c == 9 for c in frame[0])


def drop(n_moves_so_far: int) -> int:
    return 2 if (n_moves_so_far + 1) % 3 == 0 else 1


class WithHistory:
    def predict(self, frame: Grid, move: Move, history: Any) -> Grid:
        return counter(length(frame) - drop(len(history)))


class FrameOnly:
    def predict(self, frame: Grid, move: Move) -> Grid:
        return counter(length(frame) - 1)


def transitions(n: int = 7, level_break: int | None = None) -> list[Transition]:
    out: list[Transition] = []
    n_len, in_level, level = START_LEN, 0, 0
    for i in range(n):
        if level_break == i:
            n_len, in_level, level = START_LEN, 0, level + 1
        after = n_len - drop(in_level)
        out.append(
            Transition(
                i, counter(n_len), MV, counter(after), level, level, "NOT_FINISHED"
            )
        )
        n_len, in_level = after, in_level + 1
    return out


def test_history_model_replays_exactly_and_frame_only_fails() -> None:
    ts = transitions()
    assert verify_replay(WithHistory(), ts).exact
    bad = verify_replay(FrameOnly(), ts)
    assert not bad.exact and bad.mismatching_steps == (2, 5)


def test_history_is_per_level_oldest_first() -> None:
    seen: list[int] = []

    class Spy:
        def predict(self, frame: Grid, move: Move, history: Any) -> Grid:
            seen.append(len(history))
            if history:
                assert history[0][0] == counter(START_LEN)
            return counter(length(frame) - drop(len(history)))

    ts = transitions(6, level_break=3)
    assert verify_replay(Spy(), ts).exact
    assert seen == [0, 1, 2, 0, 1, 2]


def test_arity_detection() -> None:
    class Star:
        def predict(self, frame: Grid, move: Move, *rest: Any) -> Grid:
            return frame

    class Kw:
        def predict(self, frame: Grid, move: Move, history: Any = None) -> Grid:
            return frame

    assert (
        takes_history(WithHistory()) and takes_history(Star()) and takes_history(Kw())
    )
    assert not takes_history(FrameOnly())


def test_two_arg_models_unchanged_in_planner() -> None:
    res = plan(FrameOnly(), counter(4), [MV], lambda f: length(f) == 0)
    assert res.found and len(res.moves) == 4


def test_planner_rollout_extends_real_history() -> None:
    real = [(counter(10), MV), (counter(9), MV)]  # two real moves already played
    res = plan(WithHistory(), counter(8), [MV], lambda f: length(f) <= 4, history=real)
    # move numbers 3 (drops 2), 4, 5, 6 (drops 2): 8 -> 6 -> 5 -> 4
    assert res.found and [length(f) for f in res.frames] == [6, 5, 4]
    res0 = plan(WithHistory(), counter(8), [MV], lambda f: length(f) <= 4)
    assert [length(f) for f in res0.frames] == [7, 6, 4]


HIST_SRC = """
def predict(frame, move, history):
    n = sum(c == '9' for c in frame[0])
    n -= 2 if (len(history) + 1) % 3 == 0 else 1
    return ['9' * n + '0' * (len(frame[0]) - n)] + list(frame[1:])
"""


def records(ts: list[Transition]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "v": 1,
            "kind": "start",
            "levels": 0,
            "frame": {"base": encode_rows(frame_to_rows(ts[0].before))},
        }
    ]
    for t in ts:
        out.append(
            {
                "v": 1,
                "kind": "step",
                "step": t.step,
                "moves": [{"name": "ACTION1"}],
                "state": "NOT_FINISHED",
                "levels": 0,
                "frame": {"key": encode_rows(frame_to_rows(t.after))},
            }
        )
    return out


def test_sandbox_history_model_accepted_by_submission() -> None:
    from solve_arc_agi_3.world_model import RevisionLog

    res = submit_world_model(RevisionLog("m"), HIST_SRC, records(transitions()))
    try:
        assert res.accepted, res.reason
        assert res.model is not None and res.model.takes_history
    finally:
        res.close()


def test_sandbox_without_history_rejected_when_state_is_hidden() -> None:
    from solve_arc_agi_3.world_model import RevisionLog

    src = """
def predict(frame, move):
    n = sum(c == '9' for c in frame[0]) - 1
    return ['9' * n + '0' * (len(frame[0]) - n)] + list(frame[1:])
"""
    res = submit_world_model(RevisionLog("m"), src, records(transitions()))
    assert not res.accepted


def test_sandbox_history_source_still_blocked_from_escapes() -> None:
    src = "def predict(frame, move, history):\n    import os\n    return frame\n"
    with load(src) as m:
        assert m.takes_history
        m.predict(counter(3), MV, [])
        assert m.last_error is not None and m.last_error.kind == "forbidden"
