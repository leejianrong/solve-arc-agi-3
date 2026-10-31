from __future__ import annotations

import pytest

from solve_arc_agi_3.planner import PlanExecutor, PlanResult, plan
from solve_arc_agi_3.sandbox import load
from solve_arc_agi_3.world_model import Grid, Move

SIZE = 10
STEP = 3
DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
MOVES = [Move(n) for n in DIRS]


def world(pos: tuple[int, int], walls: tuple[tuple[int, int], ...] = ()) -> Grid:
    rows = [[0] * SIZE for _ in range(SIZE)]
    for x, y in walls:
        rows[y][x] = 8
    rows[pos[1]][pos[0]] = 5
    return tuple(tuple(r) for r in rows)


def find_block(frame: Grid) -> tuple[int, int]:
    for y, row in enumerate(frame):
        if 5 in row:
            return row.index(5), y
    raise AssertionError("no block")


class Toy:
    """A block that moves 3 cells per action; walls and the edge stop it."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame: Grid, move: Move) -> Grid:
        self.calls += 1
        x, y = find_block(frame)
        dx, dy = DIRS[move.name]
        nx, ny = x + dx * STEP, y + dy * STEP
        if not (0 <= nx < SIZE and 0 <= ny < SIZE) or frame[ny][nx] == 8:
            return frame
        rows = [list(r) for r in frame]
        rows[y][x] = 0
        rows[ny][nx] = 5
        return tuple(tuple(r) for r in rows)


def at(target: tuple[int, int]):
    return lambda f: find_block(f) == target


START = world((0, 0), ((3, 0),))
GOAL = at((9, 9))


def manhattan(f: Grid) -> float:
    x, y = find_block(f)
    return (abs(9 - x) + abs(9 - y)) / STEP


def test_bfs_finds_the_shortest_path_and_frames() -> None:
    res = plan(Toy(), START, MOVES, GOAL)
    assert res.found and res.reason == "found"
    assert len(res.moves) == 6 and len(res.frames) == 6
    assert find_block(res.frames[-1]) == (9, 9)
    toy = Toy()
    f = START
    for m, predicted in zip(res.moves, res.frames, strict=True):
        f = toy.predict(f, m)
        assert f == predicted


def test_start_already_at_goal_is_an_empty_plan() -> None:
    res = plan(Toy(), world((9, 9)), MOVES, GOAL)
    assert res.found and res.moves == () and res.nodes == 0


def test_node_budget_is_never_exceeded() -> None:
    toy = Toy()
    res = plan(toy, START, MOVES, GOAL, max_nodes=7)
    assert res.reason == "node budget" and not res.found
    assert res.nodes == 7 == toy.calls


def test_depth_budget() -> None:
    res = plan(Toy(), START, MOVES, GOAL, max_depth=5)
    assert res.reason == "no plan within depth" and not res.found
    assert plan(Toy(), START, MOVES, GOAL, max_depth=6).found


def test_time_budget_with_a_fake_clock() -> None:
    ticks = iter(range(1000))
    toy = Toy()
    res = plan(
        toy, START, MOVES, GOAL, time_budget_s=5, clock=lambda: float(next(ticks))
    )
    assert res.reason == "time budget" and not res.found
    assert 0 < toy.calls <= 5


def test_unreachable_goal_exhausts_the_space() -> None:
    boxed = world((0, 0), ((9, 6), (6, 9)))
    res = plan(Toy(), boxed, MOVES, GOAL)
    assert res.reason == "exhausted" and not res.found
    assert (
        res.nodes == 13 * 4
    )  # 16 block cells minus two walls and the sealed goal, 4 moves each


def test_states_are_deduplicated() -> None:
    toy = Toy()
    plan(toy, START, MOVES, lambda f: False, max_depth=50)
    # at most 16 block positions, each expanded once
    assert toy.calls <= 16 * len(MOVES)


@pytest.mark.parametrize("kw", [{"beam_width": 3}, {"heuristic": manhattan}])
def test_beam_and_heuristic_find_a_path(kw: dict[str, object]) -> None:
    res = plan(Toy(), START, MOVES, GOAL, **kw)  # type: ignore[arg-type]
    assert res.found and len(res.moves) == 6
    assert (
        plan(Toy(), START, MOVES, GOAL, heuristic=manhattan).nodes
        <= plan(Toy(), START, MOVES, GOAL).nodes
    )


def test_beam_too_narrow_reports_no_plan() -> None:
    res = plan(Toy(), START, MOVES, GOAL, beam_width=1, max_depth=2)
    assert not res.found and res.reason == "no plan within depth"


def test_model_failures_stop_the_search() -> None:
    class Bad:
        def predict(self, frame: Grid, move: Move) -> Grid:
            return ()

    class Raises:
        def predict(self, frame: Grid, move: Move) -> Grid:
            raise RuntimeError("boom")

    for model in (Bad(), Raises()):
        res = plan(model, START, MOVES, GOAL)
        assert res.reason == "model failed" and res.detail
    assert plan(Bad(), START, MOVES, GOAL).nodes == 1


def test_missing_goal_is_an_error() -> None:
    with pytest.raises(ValueError):
        plan(Toy(), START, MOVES, None)


def test_executor_continues_on_matching_frames() -> None:
    res = plan(Toy(), START, MOVES, GOAL)
    ex = PlanExecutor(res)
    toy, f = Toy(), START
    states = []
    while (m := ex.next_move()) is not None:
        f = toy.predict(f, m)
        states.append(ex.observe(f))
    assert [s.state for s in states] == ["continue"] * 5 + ["done"]
    assert ex.finished and ex.confirmed == 6


def test_executor_aborts_on_the_first_mismatch_with_a_report() -> None:
    res = plan(Toy(), START, MOVES, GOAL)
    ex = PlanExecutor(res)
    assert ex.observe(res.frames[0]).state == "continue"
    actual = world((0, 0), ((3, 0),))  # the block did not move on step 1
    status = ex.observe(actual)
    assert status.state == "aborted" and status.step == 1
    rep = status.report
    assert rep is not None and rep.step == 1 and rep.cells_differing == 2
    expected = find_block(res.frames[1])
    lo = (min(expected[1], 0), min(expected[0], 0))
    assert rep.bbox == (lo[0], lo[1], max(expected[1], 0), max(expected[0], 0))
    assert not rep.shape_mismatch and "plan step 1" in rep.text()
    assert ex.next_move() is None and ex.observe(res.frames[1]).state == "aborted"


def test_executor_shape_mismatch_and_unfound_plan() -> None:
    res = plan(Toy(), START, MOVES, GOAL)
    ex = PlanExecutor(res)
    status = ex.observe(((0,),))
    assert status.report is not None and status.report.shape_mismatch
    with pytest.raises(ValueError):
        PlanExecutor(PlanResult(False, (), (), "node budget", 3))


SANDBOX_SRC = """
def predict(frame, move):
    rows = [list(r) for r in frame]
    n = len(rows)
    pos = None
    for y in range(n):
        for x in range(n):
            if rows[y][x] == "5":
                pos = (x, y)
    d = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[move["name"]]
    nx, ny = pos[0] + 3 * d[0], pos[1] + 3 * d[1]
    if not (0 <= nx < n and 0 <= ny < n) or rows[ny][nx] == "8":
        return frame
    rows[pos[1]][pos[0]] = "0"
    rows[ny][nx] = "5"
    return ["".join(r) for r in rows]

def is_goal(frame):
    return frame[9][9] == "5"

def legal_moves(frame):
    return [{"name": "RIGHT"}, {"name": "DOWN"}]
"""


def test_sandboxed_model_plan_end_to_end() -> None:
    with load(SANDBOX_SRC, timeout=5) as model:
        assert model.has_is_goal and model.has_legal_moves
        res = plan(model, START, [Move("UP")], max_nodes=40, time_budget_s=20)
        assert res.found and len(res.moves) == 6
        assert {m.name for m in res.moves} <= {
            "RIGHT",
            "DOWN",
        }  # legal_moves overrides the given moves
        assert res.nodes <= 40
        ex = PlanExecutor(res)
        f = START
        toy = Toy()
        while (m := ex.next_move()) is not None:
            f = toy.predict(f, m)
            last = ex.observe(f)
        assert last.state == "done"


def test_sandboxed_model_failure_is_a_model_failed_plan() -> None:
    src = "def predict(frame, move):\n    return 1 + 'a'\n"
    with load(src, timeout=5) as model:
        res = plan(model, START, MOVES, GOAL, max_nodes=10)
    assert res.reason == "model failed" and "exception" in res.detail and res.nodes == 1
