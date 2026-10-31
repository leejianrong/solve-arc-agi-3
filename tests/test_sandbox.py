from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest

from solve_arc_agi_3.sandbox import (
    SandboxedModel,
    SandboxError,
    SourceRejected,
    load,
)
from solve_arc_agi_3.world_model import Grid, Move, Transition, verify_replay

SHIFT = """
def predict(frame, move):
    if move["name"] != "ACTION3":
        return frame
    out = []
    for row in frame:
        cells = list(row)
        moved = ["0"] * len(cells)
        for x, c in enumerate(cells):
            if c != "0":
                moved[x + 5] = c
        out.append("".join(moved))
    return out
"""


def grid(x: int | None, size: int = 16) -> Grid:
    rows = [[0] * size for _ in range(size)]
    if x is not None:
        rows[3][x] = 7
    return tuple(tuple(r) for r in rows)


FRAME = grid(1)
MOVE = Move("ACTION3")


@pytest.fixture
def make() -> Iterator[object]:
    models: list[SandboxedModel] = []

    def factory(source: str, **kw: float) -> SandboxedModel:
        m = load(source, **kw)
        models.append(m)
        return m

    yield factory
    for m in models:
        m.close()


def kind_of(model: SandboxedModel) -> str | None:
    model.predict(FRAME, MOVE)
    return model.last_error.kind if model.last_error else None


def test_correct_model_replays_exactly_and_worker_is_reused(make) -> None:  # type: ignore[no-untyped-def]
    model = make(SHIFT)
    pid = model.pid
    transitions = [
        Transition(i, grid(i), Move("ACTION3"), grid(i + 5), 0, 0, "NOT_FINISHED")
        for i in range(8)
    ] + [Transition(9, grid(2), Move("ACTION1"), grid(2), 0, 0, "NOT_FINISHED")]
    report = verify_replay(model, transitions)
    assert report.exact and report.checked == 9
    assert model.pid == pid and pid is not None and model.last_error is None


def test_timeout_replaces_worker_and_next_call_works(make) -> None:  # type: ignore[no-untyped-def]
    model = make(
        "def predict(frame, move):\n    if move['name'] == 'ACTION1':\n"
        "        while True:\n            pass\n    return frame\n",
        timeout=0.5,
    )
    old = model.pid
    start = time.monotonic()
    assert model.call(FRAME, Move("ACTION1")) == SandboxError(
        "timeout", "no reply within 0.5s"
    )
    assert time.monotonic() - start < 3
    assert model.call(FRAME, Move("ACTION2")) == FRAME
    assert model.pid is not None and model.pid != old


def test_memory_bomb_is_stopped(make) -> None:  # type: ignore[no-untyped-def]
    model = make("def predict(frame, move):\n    return [0] * (10 ** 11)\n", timeout=5)
    result = model.call(FRAME, MOVE)
    assert isinstance(result, SandboxError) and result.kind == "crash"
    assert model.call(FRAME, MOVE) is not None  # worker was replaced, nothing raised


@pytest.mark.parametrize(
    "body",
    [
        "open('/tmp/x', 'w')",
        "import os",
        "import subprocess",
        "import socket",
        "import sys",
        "import ctypes",
        "eval('1')",
        "exec('1')",
        "__import__('os')",
        "frame.__class__",
        "[x for x in ()].__class__",
        "(lambda: (yield))().gi_frame",
        "'{0.__class__}'.format(1)",
        "getattr(frame, 'x')",
    ],
)
def test_forbidden_constructs(make, body: str) -> None:  # type: ignore[no-untyped-def]
    model = make(f"def predict(frame, move):\n    {body}\n    return frame\n")
    assert model.predict(FRAME, MOVE) == ()
    assert model.last_error is not None and model.last_error.kind == "forbidden"


def test_allowed_imports_work(make) -> None:  # type: ignore[no-untyped-def]
    model = make(
        "import math\nfrom collections import Counter\n"
        "def predict(frame, move):\n    return frame if math.floor(1.5) == 1 and Counter('a') else []\n"
    )
    assert model.predict(FRAME, MOVE) == FRAME


def test_child_env_is_clean_and_workdir_empty(make, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("SANDBOX_SECRET_SENTINEL", "hunter2")
    model = make(SHIFT)
    info = model.limits()
    # CPython itself adds LC_CTYPE when coercing the C locale; nothing of ours is inherited
    assert set(info["env"]) <= {"LC_CTYPE"}
    assert "hunter2" not in str(info["env"])
    assert os.listdir(info["cwd"]) == []


def test_limits_are_set_in_the_child(make) -> None:  # type: ignore[no-untyped-def]
    lim = make(SHIFT).limits()["limits"]
    assert lim["RLIMIT_FSIZE"] == [0, 0] and lim["RLIMIT_CORE"] == [0, 0]
    assert lim["RLIMIT_NPROC"] == [1, 1]
    assert lim["RLIMIT_AS"] == [512 << 20, 512 << 20]
    assert lim["RLIMIT_CPU"] == [60, 60]


def test_exception_is_typed_and_message_is_cut(make) -> None:  # type: ignore[no-untyped-def]
    model = make("def predict(frame, move):\n    raise ValueError('x' * 1000)\n")
    assert kind_of(model) == "exception"
    assert model.last_error is not None
    assert model.last_error.message.startswith("ValueError: xxx")
    assert len(model.last_error.message) <= 300


@pytest.mark.parametrize(
    "ret",
    ["frame[:-1]", "['0']", "42", "None", "[[0] * 16] * 16", "['z' * 16] * 16"],
)
def test_bad_output(make, ret: str) -> None:  # type: ignore[no-untyped-def]
    model = make(f"def predict(frame, move):\n    return {ret}\n")
    assert kind_of(model) == "bad_output"


def test_huge_print_is_bad_output_and_worker_survives(make) -> None:  # type: ignore[no-untyped-def]
    model = make(
        "def predict(frame, move):\n    for i in range(10 ** 6):\n        print('x' * 1000)\n    return frame\n"
    )
    assert kind_of(model) == "bad_output"
    assert kind_of(model) == "bad_output"  # same worker still answers


def test_reply_over_the_cap_is_bad_output(make) -> None:  # type: ignore[no-untyped-def]
    model = SandboxedModel(SHIFT, max_reply_bytes=10)
    try:
        assert kind_of(model) == "bad_output"
    finally:
        model.close()


def test_worker_crash_is_typed_and_recovers(make) -> None:  # type: ignore[no-untyped-def]
    model = make(
        "def predict(frame, move):\n    if move['name'] == 'ACTION1':\n"
        "        def f(n):\n            return f(n + 1) + 1\n        f(0)\n    return frame\n"
    )
    assert model.call(FRAME, Move("ACTION1")).kind == "exception"  # type: ignore[union-attr]
    assert model.call(FRAME, MOVE) == FRAME


def test_load_rejects_syntax_error_and_missing_predict() -> None:
    with pytest.raises(SourceRejected) as exc:
        load("def predict(frame, move:\n")
    assert "SyntaxError" in exc.value.error.message
    with pytest.raises(SourceRejected):
        load("x = 1\n")


def test_close_reaps_worker_and_is_idempotent() -> None:
    with load(SHIFT) as model:
        proc = model._proc
        assert proc is not None
        pid = proc.pid
        assert model.predict(FRAME, MOVE) != ()
    model.close()
    assert proc.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert model.pid is None
    assert model.predict(FRAME, MOVE) == ()


# --- regression: allowed modules must not leak other modules (re.enum.sys.modules reached os) ---------


@pytest.mark.parametrize(
    "expr",
    [
        "re.enum.sys.modules['os'].getcwd()",
        "functools.sys.modules['os'].getcwd()",
        "collections.sys.modules['os'].getcwd()",
        "copy.types.sys.modules['os'].getcwd()",
        "heapq.sys.modules['os'].getcwd()",
    ],
)
def test_allowed_modules_do_not_expose_other_modules(expr: str) -> None:
    mod = expr.split(".")[0]
    source = f"import {mod}\ndef predict(frame, move):\n    return [[len({expr})]]\n"
    with SandboxedModel(source, timeout=2.0) as model:
        result = model.call(((0, 0), (0, 0)), Move(name="ACTION1"))
    assert isinstance(result, SandboxError)
    assert (
        result.kind == "exception" and "AttributeError" in result.message
    )  # no such attribute, nothing reached


def test_the_allowed_helpers_still_work_through_the_proxies() -> None:
    source = (
        "import math, itertools, heapq, bisect, re, functools, copy, collections\n"
        "from collections import deque\n"
        "def predict(frame, move):\n"
        "    h = []\n"
        "    heapq.heappush(h, 3); heapq.heappush(h, 1)\n"
        "    ok = (math.isqrt(16) == 4 and next(itertools.permutations([1, 2])) == (1, 2) and heapq.heappop(h) == 1\n"
        "          and bisect.bisect([1, 3, 5], 4) == 2 and re.sub('a', 'b', 'aa') == 'bb'\n"
        "          and functools.reduce(lambda a, b: a + b, [1, 2, 3]) == 6 and copy.deepcopy([1]) == [1]\n"
        "          and len(deque([1, 2])) == 2 and collections.Counter('aab')['a'] == 2)\n"
        "    return frame if ok else [[0]]\n"
    )
    with SandboxedModel(source, timeout=2.0) as model:
        frame = ((1, 2), (3, 4))
        assert model.call(frame, Move(name="ACTION1")) == frame


def test_a_module_that_is_not_on_the_list_cannot_be_imported() -> None:
    source = "import types\ndef predict(frame, move):\n    return frame\n"
    with SandboxedModel(source, timeout=2.0) as model:
        result = model.call(((0,),), Move(name="ACTION1"))
    assert isinstance(result, SandboxError) and result.kind == "forbidden"


def test_replay_reports_the_sandbox_error_and_stops_after_one_timeout(make) -> None:  # type: ignore[no-untyped-def]
    model = make(
        "def predict(frame, move):\n    while True:\n        pass\n", timeout=0.3
    )
    ts = [Transition(i, FRAME, MOVE, FRAME, 0, 0, "NOT_FINISHED") for i in range(20)]
    start = time.monotonic()
    report = verify_replay(model, ts)
    assert time.monotonic() - start < 3.0  # twenty timeouts would be 6s
    assert report.stopped_early and report.checked == 1
    assert report.first_error.startswith("step 0: timeout")


GOAL_SRC = """
def predict(frame, move):
    return frame

def is_goal(frame):
    return frame[3][1] == "7"

def legal_moves(frame):
    return [{"name": "ACTION1"}, {"name": "ACTION6", "x": 2, "y": 3}]
"""


def test_optional_is_goal_and_legal_moves(make) -> None:  # type: ignore[no-untyped-def]
    m = make(GOAL_SRC)
    assert m.has_is_goal and m.has_legal_moves
    assert m.is_goal(grid(1)) is True and m.is_goal(grid(2)) is False
    assert m.legal_moves(FRAME) == [Move("ACTION1"), Move("ACTION6", 2, 3)]
    assert m.last_error is None
    assert m.predict(FRAME, MOVE) == FRAME  # predict still works on the same worker


def test_undefined_or_bad_optional_functions_are_typed_errors(make) -> None:  # type: ignore[no-untyped-def]
    plain = make(SHIFT)
    assert not plain.has_is_goal and plain.is_goal(FRAME) is False
    assert plain.last_error is not None and "is_goal" in plain.last_error.message
    assert plain.legal_moves(FRAME) == []
    bad = make(
        "def predict(frame, move):\n    return frame\n"
        "def is_goal(frame):\n    return 1\n"
        "def legal_moves(frame):\n    return [{'name': 3}]\n"
    )
    assert bad.is_goal(FRAME) is False and bad.last_error is not None
    assert bad.last_error.kind == "bad_output"
    assert bad.legal_moves(FRAME) == [] and bad.last_error.kind == "bad_output"
    loops = make(
        "def predict(frame, move):\n    return frame\n"
        "def is_goal(frame):\n    while True:\n        pass\n",
        timeout=0.3,
    )
    assert loops.is_goal(FRAME) is False and loops.last_error is not None
    assert loops.last_error.kind == "timeout"
    assert loops.predict(FRAME, MOVE) == FRAME  # worker was replaced
