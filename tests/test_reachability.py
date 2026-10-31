import glob
import json
import re
from pathlib import Path

import pytest

from solve_arc_agi_3.observation import Grid, Region, segment
from solve_arc_agi_3.reachability import (
    footprint,
    learn_motion,
    mover_parts,
    objects_reachable,
    reachable_cells,
    summary,
)

STEPS = {"UP": (0, -2), "DOWN": (0, 2), "LEFT": (-2, 0), "RIGHT": (2, 0)}


def make(rows: list[str]) -> Grid:
    return tuple(tuple(int(c, 16) for c in r) for r in rows)


def put(
    grid: Grid, x: int, y: int, size: int = 2, colour: int = 7, floor: int = 0
) -> Grid:
    g = [[floor if c == 7 else c for c in row] for row in grid]
    for yy in range(y, y + size):
        for xx in range(x, x + size):
            g[yy][xx] = colour
    return tuple(tuple(r) for r in g)


MAZE = [
    "000000000000",
    "000000000000",
    "004400000000",
    "004400000000",
    "000044000000",
    "000044000000",
    "000000000000",
    "000000000000",
]


def sprite(grid: Grid) -> Region:
    return next(r for r in segment(grid) if r.color == 7)


def _history() -> list[tuple[Grid, str, Grid]]:
    base = make(MAZE)
    a, b = put(base, 0, 0), put(base, 2, 0)
    c = put(base, 0, 2)
    # DOWN from (2,0) is blocked by the colour-4 wall at x2-3 y2-3
    return [(a, "RIGHT", b), (a, "DOWN", c), (b, "DOWN", b), (b, "LEFT", a)]


def test_learn_motion_steps_blocked_and_colours() -> None:
    m = learn_motion(_history())
    assert m.steps == {"RIGHT": (2, 0), "DOWN": (0, 2), "LEFT": (-2, 0)}
    assert m.blocked == {"DOWN": 1}
    assert m.blocked_colours == frozenset({4})
    assert m.mover == ((7, 4),)


def test_learn_motion_counter_only_change_is_blocked() -> None:
    base = make(MAZE)
    a = put(base, 0, 0)
    rows = [list(r) for r in a]
    rows[-1][0] = 9  # counter strip on the bottom edge
    ticked = tuple(tuple(r) for r in rows)
    m = learn_motion([(a, "RIGHT", put(base, 2, 0)), (a, "DOWN", ticked)])
    assert m.blocked == {"DOWN": 1}


def test_reachable_cells_respects_walls() -> None:
    base = make(MAZE)
    grid = put(base, 0, 0)
    mover = sprite(grid)
    assert mover.color == 7
    cells = reachable_cells(grid, mover, STEPS, blocked_colours={4})
    assert (0, 0) in cells and (2, 0) in cells and (10, 0) in cells
    assert (2, 2) not in cells and (4, 4) not in cells
    assert (0, 2) in cells and (0, 6) in cells
    assert (12, 0) not in cells
    # without learned blocking everything in the lattice is reachable
    assert (2, 2) in reachable_cells(grid, mover, STEPS)
    # passable whitelist behaves as a block on other colours
    assert (2, 2) not in reachable_cells(grid, mover, STEPS, passable_colours={0})
    assert len(reachable_cells(grid, mover, STEPS, limit=3)) <= 4


def test_enclosed_mover_reaches_only_itself() -> None:
    rows = ["000000", "044440", "040040", "040040", "044440", "000000"]
    grid = put(make(rows), 2, 2)
    mover = sprite(grid)
    # ring of 4 around a 2x2 pocket, sprite fills the pocket exactly
    cells = reachable_cells(grid, mover, STEPS, blocked_colours={4})
    assert cells == {(2, 2)}


def test_objects_reachable_and_summary() -> None:
    base = [list(r) for r in make(MAZE)]
    base[0][10] = 9  # goal reachable in the open row
    base[6][4] = 9  # goal behind nothing: reachable too
    grid = put(tuple(tuple(r) for r in base), 0, 0)
    mover = sprite(grid)
    objs = objects_reachable(grid, mover, STEPS, blocked_colours={4})
    assert (9, 1, (10, 0)) in objs
    # the wall itself is a blocked colour and never listed
    assert all(c != 4 for c, _s, _p in objs)
    model = learn_motion(_history())
    line = summary(grid, mover, model)
    assert "positions" in line and "color 9" in line


def test_unreachable_object_not_listed() -> None:
    # a goal in the right half, cut off by a full-height wall of 4s
    sealed = [list(r) for r in make(["0" * 10] * 6)]
    for y in range(6):
        sealed[y][5] = 4
    sealed[2][8] = 9
    g = put(tuple(tuple(r) for r in sealed), 0, 0)
    mover = sprite(g)
    objs = objects_reachable(g, mover, STEPS, blocked_colours={4})
    assert not any(c == 9 for c, _s, _p in objs)
    assert any(c == 9 for c, _s, _p in objects_reachable(g, mover, STEPS))


def _ls20_history() -> list[tuple[Grid, str, Grid]]:
    files = glob.glob("build/dev-runs/v6/traces/*/ls20*.obs.jsonl")
    if not files:
        pytest.skip("ls20 trace not present")
    obs_path = files[0]
    texts = {}
    for line in Path(obs_path).read_text().splitlines():
        r = json.loads(line)
        rows = re.findall(
            r"^[0-9a-f]{20,}$", r["text"].split("Grid (rows", 1)[1], re.MULTILINE
        )
        texts[r["step"]] = tuple(tuple(int(c, 16) for c in row) for row in rows)
    labels = {}
    for line in Path(obs_path.replace(".obs", "")).read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "step" and len(r.get("moves", [])) == 1:
            labels[r["step"]] = r["moves"][0]["name"]
    return [
        (texts[n], labels[n], texts[n + 1])
        for n in sorted(texts)
        if n in labels and n + 1 in texts
    ]


def test_ls20_trace_replay() -> None:
    hist = _ls20_history()
    m = learn_motion(hist)
    assert m.steps["ACTION1"] == (0, -5) and m.steps["ACTION2"] == (0, 5)
    assert m.steps["ACTION3"] == (-5, 0) and m.steps["ACTION4"] == (5, 0)
    assert 4 in m.blocked_colours
    grid = hist[0][0]
    rect = footprint(mover_parts(grid, m.mover))
    assert rect is not None and rect[2] - rect[0] == 4 and rect[3] - rect[1] == 4
    parts = mover_parts(grid, m.mover)
    cells = reachable_cells(grid, parts, m.steps, blocked_colours=m.blocked_colours)
    assert len(cells) > 1
