"""STATUS block for Franzen's Duck harness prompt (ARC3_STATUS_HEADER=1), after our own agent's header.

Self-contained on purpose: run.sh copies this file into the harness tree as inference/agent/status_header.py,
where it cannot import from src/. The logic is a small port of solve_arc_agi_3 (agent.py `_settle_pending`,
`_untried_line`, `_dead_moves`; observation.py `segment`, `object_motion`, `counter_strips`), on letter-coded
frames and (row, col) coordinates. Frames are rows of letters, or rows of ints (the harness `Frame.grid`).
"""

from __future__ import annotations

import re
from collections import Counter, deque
from typing import Any

ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
DEAD_ON_SAME_BOARD = 2
DEAD_CLICK_TRIES = 3
MAX_STRIP_THICKNESS = 2
LOOP_VISITS = (
    3  # the current board seen this often among the recent boards means a cycle
)
Grid = tuple[str, ...]
Region = tuple[str, int, int, int, int, int]  # colour, size, row0, col0, row1, col1
_CLICK = re.compile(
    r"^MOUSE\(\s*(?:ROW\s*=\s*)?(-?\d+)\s*,\s*(?:COL\s*=\s*)?(-?\d+)\s*\)$"
)


def to_grid(rows: Any) -> Grid:
    return tuple(
        r
        if isinstance(r, str)
        else "".join(ARC_COLOR_CHARS[max(0, min(15, int(c)))] for c in r)
        for r in rows
    )


def action_label(action: Any) -> str:
    """UP / MOUSE(4,7) from the harness spellings (`MOUSE(row=4, col=7)`, dicts, any case)."""
    if isinstance(action, dict):
        name = str(action.get("action") or action.get("name") or "").strip().upper()
        if action.get("row") is not None and action.get("col") is not None:
            return f"{name}({action['row']},{action['col']})"
        return name
    text = " ".join(str(action or "").split()).upper()
    m = _CLICK.match(text)
    return f"MOUSE({m.group(1)},{m.group(2)})" if m else text


def _components(
    grid: Grid, include, by_color: bool
) -> list[tuple[list[tuple[int, int]], str]]:
    h, w = len(grid), len(grid[0]) if grid else 0
    seen = [[False] * w for _ in range(h)]
    found = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or not include(r, c):
                continue
            colour, cells, queue = grid[r][c], [], deque([(r, c)])
            seen[r][c] = True
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr][nc]
                        and include(nr, nc)
                        and not (by_color and grid[nr][nc] != colour)
                    ):
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            found.append((cells, colour))
    return found


def background(grid: Grid) -> str:
    counts = Counter(ch for row in grid for ch in row)
    return counts.most_common(1)[0][0] if counts else "B"


def segment(grid: Grid, bg: str | None = None) -> list[Region]:
    """4-connected same-colour objects other than the background, in reading order."""
    bg = background(grid) if bg is None else bg
    out = []
    for cells, colour in _components(grid, lambda r, c: grid[r][c] != bg, True):
        rs, cs = [p[0] for p in cells], [p[1] for p in cells]
        out.append((colour, len(cells), min(rs), min(cs), max(rs), max(cs)))
    return sorted(out, key=lambda o: (o[2], o[3]))


def same_shape(before: Grid, after: Grid) -> bool:
    return len(before) == len(after) and all(
        len(a) == len(b) for a, b in zip(before, after, strict=True)
    )


def change_boxes(before: Grid, after: Grid) -> list[tuple[int, int, int, int, int]]:
    """Boxes (row0, col0, row1, col1, cells) of each connected region that differs."""
    boxes = []
    for cells, _ in _components(after, lambda r, c: before[r][c] != after[r][c], False):
        rs, cs = [p[0] for p in cells], [p[1] for p in cells]
        boxes.append((min(rs), min(cs), max(rs), max(cs), len(cells)))
    return boxes


def thin_edge_strip(box: tuple[int, int, int, int, int], h: int, w: int) -> bool:
    """Touches the border and is at most MAX_STRIP_THICKNESS thick: a timer or step-counter bar."""
    r0, c0, r1, c1, _ = box
    return ((r0 == 0 or r1 == h - 1) and r1 - r0 < MAX_STRIP_THICKNESS) or (
        (c0 == 0 or c1 == w - 1) and c1 - c0 < MAX_STRIP_THICKNESS
    )


def object_motion(
    before: Grid, after: Grid, max_distance: int = 24
) -> list[tuple[Region, Region]]:
    """Objects that kept colour and size but moved, largest move first."""
    bg = background(after)
    old, new = segment(before, bg), segment(after, bg)
    left_new = list((Counter(new) - Counter(old)).elements())
    left_old = list((Counter(old) - Counter(new)).elements())
    pairs = []
    for r in left_new:
        cands = [o for o in left_old if o[:2] == r[:2]]
        if not cands:
            continue
        near = min(cands, key=lambda o: abs(o[2] - r[2]) + abs(o[3] - r[3]))
        if abs(near[2] - r[2]) + abs(near[3] - r[3]) <= max_distance:
            left_old.remove(near)
            pairs.append((near, r))
    return sorted(
        pairs, key=lambda p: -(abs(p[0][2] - p[1][2]) + abs(p[0][3] - p[1][3]))
    )


class StatusTracker:
    """Feed it each (before, action, after) step; render() gives the compact STATUS block."""

    def __init__(self) -> None:
        self.effects: dict[
            str, list[int]
        ] = {}  # action name -> [tries, changed], whole game
        self.clicks: dict[str, list[int]] = {}  # MOUSE(r,c) -> [tries, changed]
        self.click_wins: Counter[str] = Counter()
        self.dead: Counter[tuple[str, int]] = (
            Counter()
        )  # (action, hash of the board it did nothing on)
        self.mover: Region | None = None
        self.new_level = False  # no mover known because the level just changed
        self.last = ""
        self.last_changed = False
        self.noop_streak = 0
        self.level: int | None = None
        self.board: Grid = ()
        self.boards: deque[int] = deque(maxlen=12)

    def reset_board(self, after: Any = None) -> None:
        """A RESET restarts the level: the old position and last result no longer apply."""
        self.mover, self.last, self.noop_streak = None, "", 0
        self.boards.clear()
        if after is not None:
            self.board = to_grid(after)

    def update(
        self, before: Any, action: Any, after: Any, levels: int | None = None
    ) -> None:
        b, a, label = to_grid(before), to_grid(after), action_label(action)
        name = label.split("(")[0]
        new_level = (
            self.level is not None and levels is not None and levels != self.level
        )
        self.level = levels if levels is not None else self.level
        self.board = a
        if new_level:
            self.mover, self.noop_streak, self.new_level = None, 0, True
            self.clicks.clear()
            self.click_wins.clear()
            self.dead.clear()
            self.boards.clear()
            self.last = f"{label} cleared the level"
            return
        self.new_level = False
        shaped = same_shape(b, a)
        boxes = change_boxes(b, a) if shaped else []
        h, w = len(a), len(a[0]) if a else 0
        changed = (not shaped) or any(not thin_edge_strip(x, h, w) for x in boxes)
        pairs = object_motion(b, a) if shaped and boxes else []
        if pairs:
            # keep following the known mover; else the biggest object that moved, not the longest jump
            known = [p for p in pairs if self.mover and p[1][:2] == self.mover[:2]]
            (_, _, r0, c0, _, _), nxt = (
                known[0] if known else max(pairs, key=lambda p: p[1][1])
            )
            self.mover = nxt
            self.last = f"{label} MOVED the object by (row {nxt[2] - r0:+d}, col {nxt[3] - c0:+d})"
        elif not changed:
            why = "nothing changed" if not boxes else "only a counter or timer changed"
            self.last = f"{label} was BLOCKED ({why}); the object did not move"
        else:
            self.last = f"{label} changed the board but no object moved"
        self.last_changed = changed
        stats = self.effects.setdefault(name, [0, 0])
        stats[0] += 1
        stats[1] += changed
        if name == "MOUSE":
            cell = self.clicks.setdefault(label, [0, 0])
            cell[0] += 1
            cell[1] += changed
            if changed:
                self.click_wins[label] += 1
        if not changed:
            self.dead[(label, hash(b))] += 1
        self.noop_streak = 0 if changed else self.noop_streak + 1
        self.boards.append(hash(a))

    def dead_actions(self) -> list[str]:
        here = hash(self.board)
        out = {
            k
            for (k, h), n in self.dead.items()
            if h == here and n >= DEAD_ON_SAME_BOARD
        }
        out |= {
            k for k, (t, c) in self.clicks.items() if t >= DEAD_CLICK_TRIES and c == 0
        }
        return sorted(out)

    def render(self, valid_actions: list[str] | None = None, max_items: int = 8) -> str:
        if not self.last:
            return ""
        m = self.mover
        click_only = bool(valid_actions) and all(
            action_label(v).startswith("MOUSE") or action_label(v) in ("RESET", "UNDO")
            for v in valid_actions or []
        )
        last = self.last
        if click_only and not last.endswith("cleared the level"):
            last = f"{last.split(' ')[0]} {'changed the board' if self.last_changed else 'changed nothing'}"
        where = (
            ""
            if click_only
            else f"Controlled object (colour {m[0]}, size {m[1]}) is at rows {m[2]}-{m[4]} cols {m[3]}-{m[5]}, "
            f"centre ({(m[2] + m[4]) // 2},{(m[3] + m[5]) // 2}). "
            if m
            else (
                "New level: the controlled object has not been located yet. "
                if self.new_level
                else "No object has been seen moving yet. "
            )
        )
        lines = [f"STATUS: {where}Last move: {last}."]
        untried = [
            n
            for n in dict.fromkeys(action_label(v) for v in valid_actions or [])
            if n not in ("RESET", "UNDO", "") and n not in self.effects
        ]
        if untried:
            lines.append(f"Not tried yet in this game: {', '.join(untried)}.")
        wins = self.click_wins.most_common(max_items)
        if wins:
            lines.append(
                "Clicks that changed the board: "
                + ", ".join(f"{k} x{n}" for k, n in wins)
                + " (repeat or vary these)."
            )
        dead = self.dead_actions()[:max_items]
        if dead:
            lines.append(
                f"These moves did nothing here, do not repeat them: {'; '.join(dead)}."
            )
        if self.noop_streak >= 2:
            lines.append(
                f"WARNING: your last {self.noop_streak} actions changed nothing. Try something different."
            )
        elif self.boards and self.boards.count(self.boards[-1]) >= LOOP_VISITS:
            lines.append(
                "WARNING: you keep returning to the same board. Try something different."
            )
        return "\n".join(lines)


def status_lines(
    owner: Any, entries: list[Any], current_frame: Any, valid_actions: list[str] | None
) -> list[str]:
    """Hook for the harness's `_build_user_prompt`: STATUS lines for this turn, [] when there is nothing yet.

    `entries` are the harness HistoryEntry list (action, frame after it); the tracker lives on `owner`
    and only sees the entries it has not seen, rebuilding if the history shrank or changed under it.
    """
    state = getattr(owner, "_status_header_state", None)
    keys = [(e.action, e.frame.step) for e in entries if e.frame is not None]
    if state is None or len(keys) < len(state[1]) or keys[: len(state[1])] != state[1]:
        state = (StatusTracker(), [])
        owner._status_header_state = state
    tracker, seen = state
    frames = [e for e in entries if e.frame is not None]
    for i in range(len(seen), len(frames)):
        e, prev = frames[i], frames[i - 1] if i else None
        if action_label(e.action) == "RESET":
            tracker.reset_board(e.frame.grid)
        elif prev is not None and e.action:
            tracker.update(prev.frame.grid, e.action, e.frame.grid, e.frame.level)
        seen.append(keys[i])
    block = tracker.render(valid_actions)
    return [block] if block else []
