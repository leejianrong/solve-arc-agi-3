"""Compact, deterministic text views of an ARC frame for the model.

The model sees a hex grid, a list of objects (4-connected same-colour regions with their centres, so
click targets are explicit), and a description of what the last action changed. Original code; no state.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

Grid = tuple[tuple[int, ...], ...]

_HEX = "0123456789abcdef"
MAX_OBJECTS = 40
MAX_CHANGE_REGIONS = 5


def ascii_grid(grid: Grid) -> str:
    return "\n".join("".join(_HEX[min(max(c, 0), 15)] for c in row) for row in grid)


@dataclass(frozen=True)
class Region:
    color: int
    size: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2

    def describe(self) -> str:
        cx, cy = self.center
        return f"color {self.color} size {self.size} x{self.x0}-{self.x1} y{self.y0}-{self.y1} center ({cx},{cy})"


def background_color(grid: Grid) -> int:
    counts = Counter(c for row in grid for c in row)
    return counts.most_common(1)[0][0] if counts else 0


def _components(
    grid: Grid, include: Callable[[int, int], bool], *, by_color: bool
) -> list[tuple[list[tuple[int, int]], int]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen = [[False] * width for _ in range(height)]
    found: list[tuple[list[tuple[int, int]], int]] = []
    for y in range(height):
        for x in range(width):
            if seen[y][x] or not include(x, y):
                continue
            color = grid[y][x]
            cells: list[tuple[int, int]] = []
            queue = deque([(x, y)])
            seen[y][x] = True
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and not seen[ny][nx] and include(nx, ny):
                        if by_color and grid[ny][nx] != color:
                            continue
                        seen[ny][nx] = True
                        queue.append((nx, ny))
            found.append((cells, color))
    return found


def segment(grid: Grid, *, background: int | None = None) -> list[Region]:
    """4-connected same-colour regions other than the background, in reading order."""

    if not grid:
        return []
    bg = background_color(grid) if background is None else background
    regions = []
    for cells, color in _components(grid, lambda x, y: grid[y][x] != bg, by_color=True):
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        regions.append(Region(color, len(cells), min(xs), min(ys), max(xs), max(ys)))
    return sorted(regions, key=lambda r: (r.y0, r.x0))


def describe_objects(grid: Grid, *, limit: int = MAX_OBJECTS) -> str:
    if not grid:
        return "(no grid)"
    bg = background_color(grid)
    regions = segment(grid, background=bg)
    shown = regions if len(regions) <= limit else sorted(regions, key=lambda r: -r.size)[:limit]
    shown = sorted(shown, key=lambda r: (r.y0, r.x0))
    lines = [f"background color {bg}; {len(regions)} objects (x = column, y = row):"]
    lines += [f"- {r.describe()}" for r in shown]
    if len(regions) > limit:
        lines.append(f"- ... {len(regions) - limit} smaller objects not listed")
    return "\n".join(lines)


def changed_count(before: Grid, after: Grid) -> int:
    """Number of differing cells, or -1 when the grid size changed."""

    if len(before) != len(after) or (before and len(before[0]) != len(after[0])):
        return -1
    return sum(a != b for rb, ra in zip(before, after, strict=True) for a, b in zip(rb, ra, strict=True))


def describe_change(before: Grid | None, after: Grid) -> str:
    """What differs between two frames: how many cells, where, and the colour transitions."""

    if before is None:
        return "no earlier frame"
    count = changed_count(before, after)
    if count < 0:
        return "the grid size changed"
    if count == 0:
        return "nothing changed"

    def differs(x: int, y: int) -> bool:
        return before[y][x] != after[y][x]

    comps = _components(after, differs, by_color=False)
    comps.sort(key=lambda item: -len(item[0]))
    parts = []
    for cells, _ in comps[:MAX_CHANGE_REGIONS]:
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        moves = Counter((before[y][x], after[y][x]) for x, y in cells)
        (old, new), _n = moves.most_common(1)[0]
        parts.append(f"{len(cells)} cells at x{min(xs)}-{max(xs)} y{min(ys)}-{max(ys)} ({old}->{new})")
    more = f"; {len(comps) - MAX_CHANGE_REGIONS} more regions" if len(comps) > MAX_CHANGE_REGIONS else ""
    return f"{count} cells changed: " + "; ".join(parts) + more
