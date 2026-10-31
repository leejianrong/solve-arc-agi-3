#!/usr/bin/env python3
"""Compare trace runs (docs/TRACE.md) of several variants, per label and per game (card ARC-161).

    bench_summary.py base=build/dev-runs/v6 base=build/dev-runs/v6b ledger=build/dev-runs/v7

`label=dir` may repeat; dirs with the same label are repeats and every cell shows `mean (min-max)`.
A dir holds `<game>.jsonl` files, or it is a parent whose sub-dirs hold `run.json` and those (each sub-dir is one run).
Per-step numbers are over every step of the game (not only up to the first clear); the cluster column
is the failure_clusters verdict (rules in that module). Dashes mean "never".
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from solve_arc_agi_3.failure_clusters import classify

SRC_COLUMNS = ("fallback", "override", "probe")
COLUMNS = (
    ("steps", "steps"),
    ("cleared", "levels"),
    ("first_clear", "1st clear"),
    ("idle", "no-chg %"),
    ("clicks", "clicks"),
    ("repeat", "idle rep"),
    *((s, f"{s[:4]} %") for s in SRC_COLUMNS),
)


def _key(step: Mapping[str, Any]) -> str:
    move = step["moves"][0]
    return f"{move['name']}@{move['x']},{move['y']}" if "x" in move else move["name"]


def game_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    start = next((r for r in records if r.get("kind") == "start"), {})
    steps = [r for r in records if r.get("kind") == "step"]
    base = int(start.get("levels", 0))
    first = next(
        (i + 1 for i, s in enumerate(steps) if int(s.get("levels", 0)) > base), None
    )
    n = max(len(steps), 1)
    longest = run = 0
    previous = None
    for s in steps:
        if s.get("change", {}).get("n", 1) != 0:
            run, previous = 0, None
            continue
        key = _key(s)
        run = run + 1 if key == previous else 1
        previous = key
        longest = max(longest, run)
    out: dict[str, Any] = {
        "steps": float(len(steps)),
        "cleared": float(
            max((int(s.get("levels", 0)) for s in steps), default=base) - base
        ),
        "first_clear": None if first is None else float(first),
        "idle": 100 * sum(1 for s in steps if s.get("change", {}).get("n", 1) == 0) / n,
        "clicks": float(len({_key(s) for s in steps if "x" in s["moves"][0]})),
        "repeat": float(longest),
        "cluster": classify("", records).cluster,
    }
    for src in SRC_COLUMNS:
        out[src] = 100 * sum(1 for s in steps if s.get("src") == src) / n
    return out


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _run_dirs(root: Path) -> list[Path]:
    def has(d: Path) -> bool:
        return any(not p.name.endswith(".obs.jsonl") for p in d.glob("*.jsonl"))

    if has(root):
        return [root]
    return sorted({p.parent for p in root.rglob("run.json") if has(p.parent)})


def load_runs(root: Path) -> list[dict[str, dict[str, Any]]]:
    """One {game: stats} per run dir under root."""
    runs = []
    for d in _run_dirs(root):
        games = {
            p.name.removesuffix(".jsonl"): game_stats(_read(p))
            for p in sorted(d.glob("*.jsonl"))
            if not p.name.endswith(".obs.jsonl")
        }
        runs.append(games)
    if not runs:
        raise SystemExit(f"no trace .jsonl files under {root}")
    return runs


def _cell(values: Sequence[float | None], digits: int = 0) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    fmt = f"{{:.{digits}f}}"
    if len(values) == 1:
        return fmt.format(vals[0])
    text = f"{fmt.format(statistics.fmean(vals))} ({fmt.format(min(vals))}-{fmt.format(max(vals))})"
    return text if len(vals) == len(values) else f"{text} n={len(vals)}/{len(values)}"


def _table(rows: list[tuple[str, list[dict[str, Any]]]], first: str) -> list[str]:
    head = [first, *(title for _, title in COLUMNS), "cluster"]
    body = []
    for name, stats in rows:
        cells = [name]
        for key, _ in COLUMNS:
            cells.append(
                _cell(
                    [s[key] for s in stats], 1 if key in ("idle", *SRC_COLUMNS) else 0
                )
            )
        mode = Counter(s["cluster"] for s in stats).most_common(1)[0][0]
        cells.append(
            mode
            if len(stats) == 1
            else f"{mode} {sum(s['cluster'] == mode for s in stats)}/{len(stats)}"
        )
        body.append(cells)
    widths = [max(len(r[i]) for r in [head, *body]) for i in range(len(head))]
    fmt = lambda r: "  ".join(
        c.ljust(w) for c, w in zip(r, widths, strict=True)
    ).rstrip()
    return [fmt(head), *(fmt(r) for r in body)]


def summarize(label_runs: Mapping[str, list[dict[str, dict[str, Any]]]]) -> str:
    out: list[str] = []
    for label, runs in label_runs.items():
        per_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            for game, st in run.items():
                per_game[game].append(st)
        out.append(f"== {label}: {len(runs)} run(s), {len(per_game)} game(s)")
        rows = [(g.split("-")[0], per_game[g]) for g in sorted(per_game)]
        out += _table(rows, "game")
        # aggregate over games, one value per run so min-max is across repeats
        agg: list[dict[str, Any]] = []
        for run in runs:
            games = list(run.values())
            firsts = [g["first_clear"] for g in games if g["first_clear"] is not None]
            a: dict[str, Any] = {
                key: sum(g[key] for g in games)
                / (1 if key in ("steps", "cleared", "clicks") else max(len(games), 1))
                for key, _ in COLUMNS
                if key not in ("first_clear", "repeat")
            }
            a["first_clear"] = statistics.fmean(firsts) if firsts else None
            a["repeat"] = max((g["repeat"] for g in games), default=0.0)
            a["cluster"] = (
                f"{sum(g['cleared'] > 0 for g in games)}/{len(games)} games cleared"
            )
            agg.append(a)
        out.append(
            "ALL (steps/levels/clicks summed, shares and 1st clear averaged, idle rep is the max): "
            + ", ".join(
                f"{title}={_cell([a[k] for a in agg], 1 if k in ('idle', *SRC_COLUMNS) else 0)}"
                for k, title in COLUMNS
            )
            + f", games with a clear={_cell([float(a['cluster'].split('/')[0]) for a in agg])}"
        )
        out.append("")
    return "\n".join(out).rstrip()


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", metavar="label=dir")
    args = ap.parse_args(argv)
    label_runs: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for spec in args.runs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            ap.error(f"expected label=dir, got {spec!r}")
        label_runs.setdefault(label, []).extend(load_runs(Path(path)))
    print(summarize(label_runs))


if __name__ == "__main__":
    main()
