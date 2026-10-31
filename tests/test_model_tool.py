from __future__ import annotations

import os
import time
from typing import Any

from solve_arc_agi_3.model_tool import MAX_MESSAGE, holder_for, submit_world_model
from solve_arc_agi_3.step_trace import encode_rows
from solve_arc_agi_3.world_model import Grid, Move, RevisionLog

HEX = "0123456789abcdef"
SIZE = 24

GOOD = """
def predict(frame, move):
    if move['name'] != 'ACTION3':
        return list(frame)
    rows = [list(r) for r in frame]
    out = [['0'] * len(r) for r in rows]
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c != '0':
                out[y][x + 5] = c
    return [''.join(r) for r in out]
"""


def block(x: int) -> list[str]:
    rows = [["0"] * SIZE for _ in range(SIZE)]
    for y in (4, 5):
        for xx in (x, x + 1):
            rows[y][xx] = "3"
    return ["".join(r) for r in rows]


def grid_of(rows: list[str]) -> Grid:
    return tuple(tuple(int(c, 16) for c in r) for r in rows)


def records(n: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "v": 1,
            "kind": "start",
            "levels": 0,
            "frame": {"base": encode_rows(block(0))},
        }
    ]
    for i in range(n):
        out.append(
            {
                "v": 1,
                "kind": "step",
                "step": i,
                "moves": [{"name": "ACTION3"}],
                "state": "NOT_FINISHED",
                "levels": 0,
                "frame": {"key": encode_rows(block(5 * (i + 1)))},
            }
        )
    return out


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a zombie still answers kill 0; the sandbox waits on its child, so a live answer is real
    return True


def test_correct_model_accepted_and_usable() -> None:
    log = RevisionLog("m")
    res = submit_world_model(log, GOOD, records(3))
    pid = None
    try:
        assert res.accepted and res.model is not None and res.spec is not None
        assert log.active == res.spec
        assert res.message() == "World model r1 replays all 3 steps exactly."
        pid = res.model.pid
        assert pid is not None
        grid = grid_of(block(0))
        assert res.model.predict(grid, Move("ACTION3")) == grid_of(block(5))
    finally:
        res.close()
        res.close()  # twice is safe
    assert not alive(pid)


def test_wrong_model_rejected_rollback_and_closed(monkeypatch: Any) -> None:
    log = RevisionLog("m")
    good = submit_world_model(log, GOOD, records())
    bad_src = GOOD.replace("x + 5", "x + 4")
    pids: list[int] = []
    import solve_arc_agi_3.model_tool as mt

    real_load = mt.load

    def spy(source: str, **kw: Any) -> Any:
        m = real_load(source, **kw)
        assert m.pid is not None
        pids.append(m.pid)
        return m

    monkeypatch.setattr(mt, "load", spy)
    bad = submit_world_model(log, bad_src, records())
    try:
        assert not bad.accepted and bad.model is None
        assert log.active == good.spec  # rollback
        assert "first mismatch at step 0" in bad.reason
        assert "cells wrong" in bad.message()
        deadline = time.monotonic() + 3
        while alive(pids[0]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(pids[0])
        assert good.model is not None and good.model.pid is not None
    finally:
        good.close()


def reject(src: str, **kw: Any) -> str:
    res = submit_world_model(RevisionLog("m"), src, records(), **kw)
    assert not res.accepted and res.model is None
    assert len(res.message()) <= MAX_MESSAGE
    return res.message()


def test_syntax_error_and_missing_predict() -> None:
    assert "SyntaxError" in reject("def predict(:\n")
    assert "does not define predict" in reject("x = 1\n")


def test_forbidden_import_and_builtin() -> None:
    msg = reject("import os\ndef predict(frame, move):\n    return frame\n")
    assert "forbidden" in msg and "os" in msg
    msg = reject("def predict(frame, move):\n    open('x')\n    return frame\n")
    assert "forbidden" in msg and "open" in msg


def test_infinite_loop_times_out() -> None:
    src = "def predict(frame, move):\n    while True:\n        pass\n"
    msg = reject(src, timeout=0.3)
    assert "timeout at step 0" in msg


def test_memory_bomb_crash() -> None:
    src = "def predict(frame, move):\n    x = [0] * (1 << 40)\n    return frame\n"
    msg = reject(src)
    assert "step 0" in msg and "Memory" in msg


def test_raising_model_names_exception() -> None:
    src = "def predict(frame, move):\n    raise ValueError('boom')\n"
    msg = reject(src)
    assert "exception at step 0" in msg and "ValueError: boom" in msg


def test_empty_history_rejected() -> None:
    log = RevisionLog("m")
    res = submit_world_model(log, GOOD, records(0))
    assert not res.accepted and "no recorded steps" in res.message()
    assert log.active is None and not log.rejected


def test_identical_source_unchanged_and_not_rerun() -> None:
    log = RevisionLog("m")
    first = submit_world_model(log, GOOD, records())
    try:
        again = submit_world_model(log, GOOD, records())
        assert not again.accepted and "unchanged" in again.reason
        assert again.spec == first.spec and again.model is None
        assert len(log.accepted) == 1 and not log.rejected
        assert first.model is not None and first.model.pid is not None
    finally:
        first.close()


def test_second_revision_closes_first_model() -> None:
    log = RevisionLog("m")
    first = submit_world_model(log, GOOD, records())
    assert first.model is not None and first.model.pid is not None
    pid = first.model.pid
    second = submit_world_model(log, GOOD + "\n# v2\n", records())
    try:
        assert second.accepted and second.spec is not None
        assert second.spec.revision == 2 and second.spec.parent == 1
        assert first.model.pid is None and not alive(pid)
        assert holder_for(log).model is second.model
        assert second.model is not None and second.model.pid is not None
    finally:
        second.close()
        first.close()


def test_message_is_capped() -> None:
    msg = reject("def predict(frame, move):\n    raise ValueError('x' * 5000)\n")
    assert len(msg) <= MAX_MESSAGE


def test_looping_model_is_rejected_after_one_timeout() -> None:
    src = "def predict(frame, move):\n    while True:\n        pass\n"
    start = time.monotonic()
    result = submit_world_model(RevisionLog("g"), src, records(4), timeout=0.5)
    assert (
        time.monotonic() - start < 3.0
    )  # eight timeouts would be 3.2s of waiting alone
    assert not result.accepted and result.report is not None
    assert result.report.stopped_early and result.report.checked == 1
    assert result.report.first_error.startswith("step 0: timeout")
    assert "timeout at step 0" in result.reason
