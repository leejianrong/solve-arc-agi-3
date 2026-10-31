from __future__ import annotations

import pytest

from solve_arc_agi_3.world_model import (
    Grid,
    Move,
    RevisionLog,
    Transition,
    apply_diff,
    decode_frame,
    decode_trace_frame,
    encode_frame,
    frame_to_rows,
    rows_to_frame,
    transitions_from_trace,
    verify_replay,
)


def grid_with(*cells: tuple[int, int, int], size: int = 8) -> Grid:
    rows = [[0] * size for _ in range(size)]
    for x, y, c in cells:
        rows[y][x] = c
    return tuple(tuple(r) for r in rows)


def test_decode_matches_the_documented_example_and_repeats() -> None:
    assert decode_frame(["51.363"]) == ((5, *([3] * 63)),)
    assert decode_frame(["02.31.03~2", "f8"]) == ((0, 0, 3, 0, 0, 0),) * 3 + (
        (15,) * 8,
    )
    assert (
        decode_frame(["f2~1", "~2"]) == ((15, 15),) * 4
    )  # repeat as an item of its own


def test_encode_decode_round_trip_uses_repeats() -> None:
    g = grid_with((1, 1, 5), (2, 1, 5))
    encoded = encode_frame(g)

    assert any("~" in row for row in encoded)
    assert decode_frame(encoded) == g


def test_diff_and_keyframe_decode() -> None:
    base = grid_with()
    after = apply_diff(base, [[1, 2, 4, 7], [3, 0, 0, 9]])
    assert after[1][1:6] == (0, 7, 7, 7, 0) and after[3][0] == 9 and base[1][2] == 0
    assert decode_trace_frame({"diff": [[0, 0, 0, 1]]}, base)[0][0] == 1
    assert decode_trace_frame({"key": encode_frame(after)}, None) == after
    with pytest.raises(ValueError):
        decode_trace_frame({"diff": []}, None)


def trace() -> list[dict[str, object]]:
    f0 = grid_with((0, 0, 5))
    f1 = grid_with((1, 0, 5))
    return [
        {
            "v": 1,
            "kind": "start",
            "game": "g",
            "state": "NOT_FINISHED",
            "levels": 0,
            "frame": {"base": encode_frame(f0)},
        },
        {
            "v": 1,
            "kind": "step",
            "step": 0,
            "moves": [{"name": "ACTION3"}],
            "state": "NOT_FINISHED",
            "levels": 0,
            "frame": {"diff": [[0, 0, 0, 0], [0, 1, 1, 5]]},
        },
        {
            "v": 1,
            "kind": "step",
            "step": 1,
            "moves": [{"name": "ACTION6", "x": 3, "y": 4}],
            "state": "WIN",
            "levels": 1,
            "frame": {"key": encode_frame(f1)},
        },
        {"v": 1, "kind": "other"},
    ]


def test_transitions_from_trace() -> None:
    ts = transitions_from_trace(trace())
    assert [t.step for t in ts] == [0, 1]
    assert ts[0].before[0][0] == 5 and ts[0].after[0][:2] == (0, 5)
    assert ts[1].before == ts[0].after and ts[1].move == Move("ACTION6", 3, 4)
    assert (ts[1].levels_before, ts[1].levels_after, ts[1].state_after) == (0, 1, "WIN")
    with pytest.raises(ValueError):
        transitions_from_trace(trace()[1:])


class Shift:
    def __init__(self, dx: int) -> None:
        self.dx = dx

    def predict(self, frame: Grid, move: Move) -> Grid:
        return tuple(
            tuple(row[-self.dx :] + row[: -self.dx]) if self.dx else row
            for row in frame
        )


def shifts(n: int) -> list[Transition]:
    out = []
    frame = grid_with((0, 0, 5))
    for i in range(n):
        after = Shift(1).predict(frame, Move("ACTION3"))
        out.append(Transition(i, frame, Move("ACTION3"), after, 0, 0, "NOT_FINISHED"))
        frame = after
    return out


def test_verify_replay_exact_and_mismatching() -> None:
    ts = shifts(3)
    ok = verify_replay(Shift(1), ts)
    assert (
        ok.exact
        and ok.first_mismatch is None
        and ok.mismatch_cells == 0
        and "exact" in ok.summary
    )
    bad = verify_replay(Shift(2), ts)
    assert not bad.exact and bad.first_mismatch == 0
    assert (
        bad.mismatching_steps == (0, 1, 2)
        and bad.mismatch_cells == 6
        and "first at step 0" in bad.summary
    )


def test_verify_replay_reports_errors_and_wrong_shapes() -> None:
    class Boom:
        def predict(self, frame: Grid, move: Move) -> Grid:
            raise RuntimeError("nope")

    class Small:
        def predict(self, frame: Grid, move: Move) -> Grid:
            return ((0,),)

    ts = shifts(1)
    boom = verify_replay(Boom(), ts)
    assert (
        not boom.exact
        and boom.mismatch_cells == 64
        and "RuntimeError: nope" in boom.summary
    )
    assert verify_replay(Small(), ts).mismatch_cells == 64


def test_revision_accept_reject_and_rollback() -> None:
    ts = shifts(3)
    log = RevisionLog("wm")
    assert log.active is None
    bad, report = log.submit("v1", Shift(2), ts)
    assert not report.exact and log.active is None and log.rejected == [(bad, report)]
    good, _ = log.submit("v2", Shift(1), ts)
    assert log.active == good and good.parent is None and good.revision == 2
    worse, _ = log.submit("v3", Shift(0), ts)
    assert log.active == good and len(log.rejected) == 2 and worse.parent == 2
    third, _ = log.submit("v4", Shift(1), ts)
    assert log.active == third and third.parent == 2 and third.revision == 4


def test_decode_frame_rejects_plain_hex_rows_with_a_clear_error() -> None:
    for bad in (["0000000000000000"], ["f" * 70], ["zz1"], ["a0"]):
        with pytest.raises(ValueError, match="run-length") as info:
            decode_frame(bad)
        assert "rows_to_frame" in str(info.value)
    with pytest.raises(ValueError, match="run-length"):
        decode_frame(["1" * 5000 + "1"])  # one huge run that used to overflow


def test_rows_to_frame_and_back_are_plain_hex_not_run_length() -> None:
    g = grid_with((1, 1, 15))
    rows = frame_to_rows(g)
    assert rows[1] == "0f000000" and len(rows) == 8
    assert rows_to_frame(rows) == g
    assert decode_frame(encode_frame(g)) == g != rows_to_frame(["0f"])


class Failing:
    """Looks like a SandboxedModel to verify_replay: an empty frame and a `last_error`."""

    def __init__(self) -> None:
        self.last_error: object = None
        self.calls = 0

    def predict(self, frame: Grid, move: Move) -> Grid:
        self.calls += 1
        self.last_error = "timeout: no reply within 2s"
        return ()


def test_sandbox_error_is_surfaced_and_replay_stops_early() -> None:
    model = Failing()
    report = verify_replay(model, shifts(5))
    assert not report.exact and report.stopped_early
    assert model.calls == 1 and report.checked == 1
    assert report.first_error == "step 0: timeout: no reply within 2s"
    assert "timeout" in report.summary
    # a healthy model still checks everything
    full = verify_replay(Shift(1), shifts(5))
    assert full.checked == 5 and not full.stopped_early
