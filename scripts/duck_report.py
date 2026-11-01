#!/usr/bin/env python3
"""Charts and a report for Duck-harness runs (scripts/duck_local), read from each run's benchmark.json.

    duck_report.py base=build/studies/s1/base-lp85-r3 hdr=build/studies/s1/hdr-lp85-r1 --out docs/studies/<name> --title '...'

`label=run_dir` pairs; the same label twice means repeats. Writes levels.png, actions.png and report.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
Runs = dict[str, dict[str, list[dict[str, Any]]]]  # label -> game -> runs


def load(specs: list[str]) -> Runs:
    out: Runs = defaultdict(lambda: defaultdict(list))
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"expected label=run_dir, got {spec!r}")
        for g in json.loads((Path(path) / "benchmark.json").read_text())["game_runs"]:
            out[label][g["game_id"].split("-")[0]].append(
                {
                    "levels": g["levels_completed"],
                    "of": g["number_of_levels"],
                    "score": g["final_score"],
                    "actions": len(g["history"]),
                    "per_level": g["actions_per_level"],
                }
            )
    return out


def _mean(runs: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(r[key] for r in runs)


def first_clear(runs: list[dict[str, Any]]) -> float | None:
    hits = [r["per_level"][0] for r in runs if r["levels"] > 0]
    return statistics.fmean(hits) if hits else None


def charts(data: Runs, out: Path) -> None:
    labels = list(data)
    games = sorted({g for runs in data.values() for g in runs})
    width = 0.8 / len(labels)
    for name, title, fn in (
        (
            "levels.png",
            "Levels cleared (mean of repeats)",
            lambda r: _mean(r, "levels"),
        ),
        (
            "actions.png",
            "Actions to first level clear (0 = none cleared)",
            lambda r: first_clear(r) or 0,
        ),
    ):
        fig, ax = plt.subplots(figsize=(1.6 * len(games) + 3, 3.6))
        for i, label in enumerate(labels):
            xs = [j + i * width for j in range(len(games))]
            ax.bar(
                xs,
                [fn(data[label][g]) if g in data[label] else 0 for g in games],
                width,
                label=label,
                color=COLORS[i % 4],
            )
        ax.set_xticks(
            [j + width * (len(labels) - 1) / 2 for j in range(len(games))], games
        )
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)


def report(data: Runs, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| label | game | runs | levels | first clear (actions) | actions used | score |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, games in data.items():
        for game, runs in sorted(games.items()):
            fc = first_clear(runs)
            lines.append(
                f"| {label} | {game} | {len(runs)} | {_mean(runs, 'levels'):.1f}/{runs[0]['of']} | "
                f"{'-' if fc is None else f'{fc:.0f}'} | {_mean(runs, 'actions'):.0f} | {_mean(runs, 'score'):.2f} |"
            )
    lines += ["", "![levels](levels.png) ![actions](actions.png)", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Duck runs")
    a = p.parse_args()
    data = load(a.runs)
    a.out.mkdir(parents=True, exist_ok=True)
    charts(data, a.out)
    (a.out / "report.md").write_text(report(data, a.title))
    print(a.out / "report.md")


if __name__ == "__main__":
    main()
