"""Learn how a controlled sprite moves and which cells it can reach.

`learn_motion` reads (grid_before, move_label, grid_after) history: successful moves give a per-label step
vector, moves that changed nothing but counters count as blocked and teach which colours stop the sprite.
`reachable_cells` then flood-fills sprite positions on a grid. Original code; no I/O.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from solve_arc_agi_3.observation import (
    Grid,
    Region,
    background_color,
    change_boxes,
    changed_count,
    object_motion,
    region_key,
    segment,
)

Step = tuple[int, int]
Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 inclusive
Move = tuple[Grid, str, Grid]


@dataclass
class MotionModel:
    steps: dict[str, Step] = field(default_factory=dict)
    blocked: dict[str, int] = field(default_factory=dict)
    # (colour, size) of each sprite part; empty when no motion was seen
    mover: tuple[tuple[int, int], ...] = ()
    blocked_colours: frozenset[int] = frozenset()


def mover_parts(grid: Grid, mover: Collection[tuple[int, int]]) -> list[Region]:
    wanted = Counter(mover)
    out = []
    for r in segment(grid):
        if wanted[(r.color, r.size)] > 0:
            wanted[(r.color, r.size)] -= 1
            out.append(r)
    return out


def footprint(parts: Iterable[Region]) -> Rect | None:
    ps = list(parts)
    if not ps:
        return None
    return (
        min(p.x0 for p in ps),
        min(p.y0 for p in ps),
        max(p.x1 for p in ps),
        max(p.y1 for p in ps),
    )


def _only_counters_changed(before: Grid, after: Grid) -> bool:
    """Nothing changed, or every changed region is a tiny strip near the border (a step or life counter)."""

    if changed_count(before, after) <= 0:
        return True
    h, w = len(after), len(after[0])
    return all(
        region_key(box, w, h)[0] in ("row", "col")
        for box in change_boxes(before, after)
    )


def _entered_colours(grid: Grid, rect: Rect, step: Step) -> set[int]:
    x0, y0, x1, y1 = rect
    dx, dy = step
    h, w = len(grid), len(grid[0])
    cells: set[int] = set()
    for y in range(y0 + dy, y1 + dy + 1):
        for x in range(x0 + dx, x1 + dx + 1):
            if x0 <= x <= x1 and y0 <= y <= y1:
                continue
            if 0 <= x < w and 0 <= y < h:
                cells.add(grid[y][x])
    return cells


def learn_motion(history: Iterable[Move]) -> MotionModel:
    """Per-move step vectors for the controlled object plus blocked counts and blocking colours."""

    moves = [m for m in history if m[0] and m[2]]
    votes: dict[str, Counter[Step]] = {}
    sigs: Counter[tuple[tuple[int, int], ...]] = Counter()
    for before, label, after in moves:
        pairs = object_motion(before, after, limit=8)
        if not pairs:
            continue
        a, b = pairs[0]
        step: Step = (b.x0 - a.x0, b.y0 - a.y0)
        votes.setdefault(label, Counter())[step] += 1
        group = [p for p in pairs if (p[1].x0 - p[0].x0, p[1].y0 - p[0].y0) == step]
        sigs[tuple(sorted((p[0].color, p[0].size) for p in group))] += 1
    model = MotionModel()
    model.steps = {label: c.most_common(1)[0][0] for label, c in votes.items()}
    if sigs:
        model.mover = sigs.most_common(1)[0][0]
    stopped: set[int] = set()
    passed: set[int] = set()
    for before, label, after in moves:
        known = model.steps.get(label)
        if not model.mover or object_motion(before, after, limit=1):
            if known and model.mover:
                rect = footprint(mover_parts(before, model.mover))
                if rect:
                    passed |= _entered_colours(before, rect, known)
            continue
        if not _only_counters_changed(before, after):
            continue
        model.blocked[label] = model.blocked.get(label, 0) + 1
        rect = footprint(mover_parts(before, model.mover))
        if known and rect:
            stopped |= _entered_colours(before, rect, known)
    model.blocked_colours = frozenset(stopped - passed - {c for c, _ in model.mover})
    return model


def _as_rect(mover_region: Region | Sequence[Region]) -> Rect:
    parts = [mover_region] if isinstance(mover_region, Region) else list(mover_region)
    rect = footprint(parts)
    if rect is None:
        raise ValueError("no mover region")
    return rect


def reachable_cells(
    grid: Grid,
    mover_region: Region | Sequence[Region],
    step_vectors: Mapping[str, Step] | Iterable[Step],
    passable_colours: Collection[int] | None = None,
    limit: int = 5000,
    *,
    blocked_colours: Collection[int] = (),
) -> set[tuple[int, int]]:
    """Top-left positions of the sprite footprint reachable by repeating moves (start included).

    A move is blocked when the footprint would leave the grid or overlap a colour in `blocked_colours`
    (or, when `passable_colours` is given, any colour outside it). The sprite's own colours never block.
    """

    if not grid:
        return set()
    x0, y0, x1, y1 = _as_rect(mover_region)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    own = {grid[y][x] for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}
    stops = set(blocked_colours)
    allowed = None if passable_colours is None else set(passable_colours) | own
    vectors = list(
        step_vectors.values() if isinstance(step_vectors, Mapping) else step_vectors
    )
    vectors = [v for v in dict.fromkeys(vectors) if v != (0, 0)]
    H, W = len(grid), len(grid[0])

    def free(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return False
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                c = grid[yy][xx]
                if c in own and c not in stops:
                    continue
                if c in stops or (allowed is not None and c not in allowed):
                    return False
        return True

    seen = {(x0, y0)}
    queue = deque([(x0, y0)])
    while queue and len(seen) < limit:
        cx, cy = queue.popleft()
        for dx, dy in vectors:
            nxt = (cx + dx, cy + dy)
            if nxt not in seen and free(*nxt):
                seen.add(nxt)
                queue.append(nxt)
    return seen


def objects_reachable(
    grid: Grid,
    mover_region: Region | Sequence[Region],
    step_vectors: Mapping[str, Step] | Iterable[Step],
    passable_colours: Collection[int] | None = None,
    limit: int = 5000,
    *,
    blocked_colours: Collection[int] = (),
    max_size: int | None = None,
) -> list[tuple[int, int, tuple[int, int]]]:
    """(colour, size, centre) of objects the sprite footprint can overlap from some reachable position."""

    x0, y0, x1, y1 = _as_rect(mover_region)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    cap = max_size if max_size is not None else 4 * w * h
    spots = reachable_cells(
        grid,
        mover_region,
        step_vectors,
        passable_colours,
        limit,
        blocked_colours=blocked_colours,
    )
    own = {grid[y][x] for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}
    bg = background_color(grid)
    found = []
    for r in segment(grid, background=bg):
        if r.size > cap or r.color in own or r.color in set(blocked_colours):
            continue
        if x0 <= r.x0 and r.x1 <= x1 and y0 <= r.y0 and r.y1 <= y1:
            continue
        if any(
            sx <= r.x1 and r.x0 <= sx + w - 1 and sy <= r.y1 and r.y0 <= sy + h - 1
            for sx, sy in spots
        ):
            found.append((r.color, r.size, r.center))
    return found


def summary(
    grid: Grid,
    mover_region: Region | Sequence[Region],
    model: MotionModel,
    *,
    max_objects: int = 6,
    max_chars: int = 300,
) -> str:
    """Status-header line: how many positions the sprite can reach and which objects are among them."""

    spots = reachable_cells(
        grid, mover_region, model.steps, blocked_colours=model.blocked_colours
    )
    objs = objects_reachable(
        grid, mover_region, model.steps, blocked_colours=model.blocked_colours
    )
    shown = ", ".join(
        f"color {c} size {s} at {ctr}" for c, s, ctr in objs[:max_objects]
    )
    more = f" (+{len(objs) - max_objects} more)" if len(objs) > max_objects else ""
    text = f"Reachable by repeating moves: {len(spots)} positions; objects reachable: {shown or 'none'}{more}."
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."
