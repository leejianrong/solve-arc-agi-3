"""Per-game step trace, as specified in docs/TRACE.md.

One JSONL file per game plus a ``run.json``. Records are appended and flushed one at a time, so a killed
run still leaves a readable trace. The frame encoding lives here (and its inverse, for readers and tests).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .observation import Grid

SCHEMA_VERSION = 1
HEX = "0123456789abcdef"
KEYFRAME_RUNS = 150  # a diff with more runs than this is stored as a full keyframe
REPLY_CHARS = 2000
REASONING_CHARS = 300
TOOL_CALL_CHARS = 500
ERROR_CHARS = 300


def grid_rows(grid: Grid) -> list[str]:
    return ["".join(HEX[min(max(c, 0), 15)] for c in row) for row in grid]


def encode_rows(rows: Sequence[str]) -> list[str]:
    """Run-length rows ("5" then count, tokens joined by "."), identical neighbours folded into "~n"."""

    encoded: list[str] = []
    previous: str | None = None
    repeats = 0
    for row in rows:
        tokens: list[str] = []
        i = 0
        while i < len(row):
            j = i
            while j < len(row) and row[j] == row[i]:
                j += 1
            tokens.append(f"{row[i]}{j - i}")
            i = j
        text = ".".join(tokens)
        if text == previous:
            repeats += 1
            continue
        if previous is not None:
            encoded.append(previous + (f"~{repeats}" if repeats else ""))
        previous, repeats = text, 0
    if previous is not None:
        encoded.append(previous + (f"~{repeats}" if repeats else ""))
    return encoded


def decode_rows(encoded: Sequence[str]) -> list[str]:
    rows: list[str] = []
    for item in encoded:
        text, repeats = item, 0
        if "~" in item:
            text, count = item.split("~")
            repeats = int(count)
        row = "".join(token[0] * int(token[1:]) for token in text.split("."))
        rows.extend([row] * (1 + repeats))
    return rows


def diff_runs(before: Sequence[str], after: Sequence[str]) -> list[list[int]]:
    """Changed cells as [row, x0, x1, colour] runs of one new colour."""

    runs: list[list[int]] = []
    for y, (old, new) in enumerate(zip(before, after, strict=True)):
        x = 0
        while x < len(new):
            if old[x] == new[x]:
                x += 1
                continue
            x0 = x
            while (
                x + 1 < len(new) and old[x + 1] != new[x + 1] and new[x + 1] == new[x0]
            ):
                x += 1
            runs.append([y, x0, x, HEX.index(new[x0])])
            x += 1
    return runs


def apply_diff(before: Sequence[str], runs: Sequence[Sequence[int]]) -> list[str]:
    cells = [list(row) for row in before]
    for y, x0, x1, colour in runs:
        for x in range(x0, x1 + 1):
            cells[y][x] = HEX[colour]
    return ["".join(row) for row in cells]


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


class GameTrace:
    def __init__(
        self,
        path: Path,
        obs_path: Path | None,
        started: float,
        clock: Callable[[], float],
    ) -> None:
        self._path, self._obs_path = path, obs_path
        self._started, self._clock = started, clock
        self._lock = threading.Lock()
        self._rows: list[str] | None = None
        self._step = 0

    def _append(self, path: Path, record: Mapping[str, Any]) -> None:
        with self._lock, path.open("a", encoding="utf-8") as out:
            out.write(
                json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
            )

    def start(
        self,
        game: str,
        grid: Grid,
        state: str,
        levels: int,
        win_levels: int,
        avail: Sequence[int],
    ) -> None:
        self._rows = grid_rows(grid)
        self._append(
            self._path,
            {
                "v": SCHEMA_VERSION,
                "kind": "start",
                "t": round(self._clock() - self._started, 3),
                "game": game,
                "state": state,
                "levels": levels,
                "win_levels": win_levels,
                "avail": list(avail),
                "frame": {"base": encode_rows(self._rows)},
            },
        )

    def step(
        self,
        *,
        game: str,
        src: str,
        move: Mapping[str, Any],
        grid: Grid,
        state: str,
        levels: int,
        win_levels: int,
        avail: Sequence[int],
        call: Mapping[str, Any] | None = None,
        reply: Mapping[str, Any] | None = None,
        plan: Sequence[Mapping[str, Any]] | None = None,
        plan_idx: int | None = None,
        nudge: bool = False,
        error: str | None = None,
        settled: Mapping[str, Any] | None = None,
        obs_text: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "kind": "step",
            "t": round(self._clock() - self._started, 3),
            "game": game,
            "step": self._step,
            "src": src,
            "moves": [dict(move)],
            "state": state,
            "levels": levels,
            "win_levels": win_levels,
            "avail": list(avail),
        }
        if call is not None:
            record["call"] = dict(call)
        if reply is not None:
            record["reply"] = {
                "content": _cut(str(reply.get("content", "")), REPLY_CHARS),
                "reasoning_tail": str(reply.get("reasoning_tail", ""))[
                    -REASONING_CHARS:
                ],
                "tool_calls": [
                    _cut(str(c), TOOL_CALL_CHARS) for c in reply.get("tool_calls", [])
                ],
            }
        if plan is not None:
            record["plan"], record["plan_idx"] = [dict(m) for m in plan], plan_idx
        if nudge:
            record["nudge"] = True
        if error:
            record["error"] = _cut(error, ERROR_CHARS)
        if settled:
            record["change"] = (
                {
                    "n": settled["n"],
                    "bbox": settled["bbox"],
                    "regions": settled["regions"],
                }
                if settled["n"]
                else {"n": 0}
            )
            record["motion"], record["counters"] = (
                settled["motion"],
                settled["counters"],
            )
            record["effects"] = settled["effects"]
        rows = grid_rows(grid)
        if rows and self._rows is not None and len(rows) == len(self._rows):
            runs = diff_runs(self._rows, rows)
            record["frame"] = (
                {"key": encode_rows(rows)}
                if len(runs) > KEYFRAME_RUNS
                else {"diff": runs}
            )
        elif rows:
            record["frame"] = {"key": encode_rows(rows)}
        if rows:
            self._rows = rows
        self._append(self._path, record)
        if obs_text is not None and self._obs_path is not None:
            self._append(self._obs_path, {"step": self._step, "text": obs_text})
        self._step += 1


class RunTrace:
    def __init__(
        self,
        root: Path,
        *,
        run_id: str | None = None,
        settings: Mapping[str, Any] | None = None,
        keep_obs: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._keep_obs, self._clock, self._started = keep_obs, clock, clock()
        self._meta: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": dict(settings or {}),
            "games": [],
            "scorecard": None,
        }
        self._lock = threading.Lock()
        self._write_meta()

    def _write_meta(self) -> None:
        (self.run_dir / "run.json").write_text(
            json.dumps(self._meta, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    def game(self, game_id: str) -> GameTrace:
        with self._lock:
            if game_id not in self._meta["games"]:
                self._meta["games"].append(game_id)
                self._write_meta()
        obs = self.run_dir / f"{game_id}.obs.jsonl" if self._keep_obs else None
        return GameTrace(
            self.run_dir / f"{game_id}.jsonl", obs, self._started, self._clock
        )

    def finish(
        self,
        scorecard: Mapping[str, Any] | None = None,
        baseline_actions: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        finish_run_dir(self.run_dir, scorecard, baseline_actions)


def finish_run_dir(
    run_dir: Path,
    scorecard: Mapping[str, Any] | None = None,
    baseline_actions: Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Fill in the end of a run in ``run.json`` (callable from code that only knows the directory)."""

    path = run_dir / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["scorecard"] = dict(scorecard) if scorecard else None
    if baseline_actions:
        meta["baseline_actions"] = {k: list(v) for k, v in baseline_actions.items()}
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
