"""Group the games a run did not clear by the mechanism that most plausibly stopped it (slice V5, ARC-54).

Reads step traces (docs/TRACE.md). The rules are deliberately plain and listed here so a ranking can be argued
with: the first matching rule wins, and every number a rule uses is printed in the report.

1. runtime        more than 10% of steps had a failed model call (timeout or server error)
2. tool_syntax    more than 10% of steps were fallbacks because the reply held no usable move
3. exploration    over 30+ steps, fewer than 60% of the offered actions were ever tried, or (click games) under 5 distinct clicks
4. memory         one move repeated 5+ times in a row although it changed nothing
5. goal           every offered action was tried and the board kept changing (30%+ of steps) but no level cleared
6. perception     every action was tried and 70%+ of steps changed nothing: the agent could not find what reacts
7. other          none of the above
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLUSTERS = (
    "runtime",
    "tool_syntax",
    "exploration",
    "memory",
    "goal",
    "perception",
    "other",
)
MIN_STEPS = 30
FAILED_SHARE = 0.10
UNTRIED_SHARE = 0.60
REPEAT_RUN = 5
CHANGING_SHARE = 0.30
IDLE_SHARE = 0.70


@dataclass(frozen=True)
class GameVerdict:
    game: str
    cleared: bool
    steps: int
    cluster: str  # "cleared" for games that cleared a level
    wasted: int  # steps before the first clear, or all steps when nothing cleared
    facts: dict[str, float]


def _move_key(record: Mapping[str, Any]) -> str:
    move = record["moves"][0]
    return (
        f"{move['name']}@{move['x']},{move['y']}" if "x" in move else str(move["name"])
    )


def classify(game: str, records: Iterable[Mapping[str, Any]]) -> GameVerdict:
    start: Mapping[str, Any] = {}
    steps: list[Mapping[str, Any]] = []
    for r in records:
        if r.get("kind") == "start":
            start = r
        elif r.get("kind") == "step":
            steps.append(r)
    levels = int(start.get("levels", 0))
    first_clear = next(
        (i for i, s in enumerate(steps) if int(s.get("levels", 0)) > levels), None
    )
    played = steps if first_clear is None else steps[: first_clear + 1]
    n = max(len(played), 1)
    failed = sum(
        1
        for s in played
        if (s.get("call") or {}).get("ok") is False and s["src"] != "forced_reset"
    )
    syntax = sum(
        1
        for s in played
        if s["src"] == "fallback" and (s.get("call") or {}).get("ok") is not False
    )
    avail = {a for a in start.get("avail", []) if a != 0}
    tried_names = {
        s["moves"][0]["name"] for s in played if s["moves"][0]["name"] != "RESET"
    }
    tried = len(
        {
            name
            for name in tried_names
            if name.startswith("ACTION") and int(name[6:]) in avail
        }
    )
    clicks = len({_move_key(s) for s in played if "x" in s["moves"][0]})
    idle = sum(1 for s in played if s.get("change", {}).get("n", 1) == 0)
    longest = run = 0
    previous = None
    for s in played:
        key = _move_key(s)
        run = (
            run + 1 if (key == previous and s.get("change", {}).get("n", 1) == 0) else 1
        )
        previous = key
        longest = max(longest, run if s.get("change", {}).get("n", 1) == 0 else 0)
    facts = {
        "failed_calls": failed / n,
        "syntax_fallbacks": syntax / n,
        "tried_share": tried / max(len(avail), 1),
        "distinct_clicks": float(clicks),
        "idle_share": idle / n,
        "longest_idle_repeat": float(longest),
    }
    if first_clear is not None:
        return GameVerdict(game, True, len(steps), "cleared", len(played), facts)
    if facts["failed_calls"] > FAILED_SHARE:
        cluster = "runtime"
    elif facts["syntax_fallbacks"] > FAILED_SHARE:
        cluster = "tool_syntax"
    elif len(played) >= MIN_STEPS and (
        facts["tried_share"] < UNTRIED_SHARE or (6 in avail and clicks < 5)
    ):
        cluster = "exploration"
    elif longest >= REPEAT_RUN:
        cluster = "memory"
    elif facts["tried_share"] >= 1.0 and 1 - facts["idle_share"] >= CHANGING_SHARE:
        cluster = "goal"
    elif facts["tried_share"] >= 1.0 and facts["idle_share"] >= IDLE_SHARE:
        cluster = "perception"
    else:
        cluster = "other"
    return GameVerdict(game, False, len(steps), cluster, len(played), facts)


def load_run(run_dir: Path) -> list[GameVerdict]:
    verdicts = []
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name.endswith(".obs.jsonl"):
            continue
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        verdicts.append(classify(path.name.removesuffix(".jsonl"), records))
    return verdicts


def report(verdicts: list[GameVerdict]) -> str:
    by_cluster: dict[str, list[GameVerdict]] = defaultdict(list)
    for v in verdicts:
        by_cluster[v.cluster].append(v)
    lines = [
        (
            f"{len(verdicts)} games, {sum(v.cleared for v in verdicts)} with a cleared level, "
            f"{sum(v.steps for v in verdicts)} actions."
        ),
        "",
        "| cluster | games | actions spent | examples |",
        "|---|---|---|---|",
    ]
    for name in sorted(
        (c for c in by_cluster if c != "cleared"),
        key=lambda c: -sum(v.wasted for v in by_cluster[c]),
    ):
        group = by_cluster[name]
        lines.append(
            f"| {name} | {len(group)} | {sum(v.wasted for v in group)} | {', '.join(v.game[:4] for v in group[:6])} |"
        )
    if "cleared" in by_cluster:
        lines.append(
            f"| cleared | {len(by_cluster['cleared'])} | {sum(v.wasted for v in by_cluster['cleared'])} | "
            f"{', '.join(v.game[:4] for v in by_cluster['cleared'])} |"
        )
    return "\n".join(lines)
