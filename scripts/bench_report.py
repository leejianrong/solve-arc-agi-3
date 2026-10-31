#!/usr/bin/env python3
"""Charts and a written report for a Haiku study (docs/studies/<name>/), built on bench_summary.

    bench_report.py base-r1=build/bench/traces/base-r1 --out docs/studies/2026-11-01-base-r1 --title '...'

`label=dir` follows bench_summary (same label twice = repeats). Writes one PNG per chart and report.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location(
    "bench_summary", Path(__file__).resolve().with_name("bench_summary.py")
)
assert _spec and _spec.loader
bs = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("bench_summary", bs)
_spec.loader.exec_module(bs)

COLORS = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
CHARTS = (
    ("levels.png", "Levels cleared per game"),
    ("idle.png", "No-change share of steps per game"),
    ("clicks.png", "Distinct clicks per game"),
    ("first_clear.png", "Step of first clear per game"),
    ("curves.png", "Cumulative levels cleared vs step"),
    ("overview.png", "Overview of headline numbers"),
)

Runs = dict[str, list[dict[str, Any]]]  # label -> runs; run = {game: stats + "curve"}


def load(specs: Sequence[str]) -> Runs:
    out: Runs = {}
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"expected label=dir, got {spec!r}")
        for d in bs._run_dirs(Path(path)):
            run: dict[str, Any] = {}
            for p in sorted(d.glob("*.jsonl")):
                if p.name.endswith(".obs.jsonl"):
                    continue
                recs = bs._read(p)
                st = bs.game_stats(recs)
                start = next((r for r in recs if r.get("kind") == "start"), {})
                base = int(start.get("levels", 0))
                st["curve"] = [
                    int(r.get("levels", 0)) - base
                    for r in recs
                    if r.get("kind") == "step"
                ]
                run[p.name.removesuffix(".jsonl").split("-")[0]] = st
            out.setdefault(label, []).append(run)
        if not out.get(label):
            raise SystemExit(f"no trace .jsonl files under {path}")
    return out


def games_of(data: Runs) -> list[str]:
    return sorted({g for runs in data.values() for r in runs for g in r})


def vals(data: Runs, label: str, game: str, key: str) -> list[float | None]:
    return [r[game][key] for r in data[label] if game in r]


def mean(v: Sequence[float | None]) -> float | None:
    x = [a for a in v if a is not None]
    return statistics.fmean(x) if x else None


def cap_of(data: Runs) -> int:
    return int(
        max(
            (s["steps"] for rs in data.values() for r in rs for s in r.values()),
            default=0,
        )
    )


def _err(v: Sequence[float | None]) -> tuple[float, float]:
    x = [a for a in v if a is not None]
    if len(x) < 2:
        return 0.0, 0.0
    m = statistics.fmean(x)
    return m - min(x), max(x) - m


def grouped(
    data: Runs, key: str, ylabel: str, title: str, path: Path, cap: float | None = None
) -> None:
    games, labels = games_of(data), list(data)
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(
        figsize=(max(6, 2 + 1.6 * len(games) * len(labels) / 2), 4.5)
    )
    for i, lab in enumerate(labels):
        xs = [j + (i - (len(labels) - 1) / 2) * w for j in range(len(games))]
        v = [vals(data, lab, g, key) for g in games]
        ms = [mean(a) for a in v]
        if cap is None:
            hs = [m or 0.0 for m in ms]
            err = list(zip(*[_err(a) for a in v], strict=True))
            bars = ax.bar(
                xs,
                hs,
                w,
                label=lab,
                color=COLORS[i % 6],
                yerr=err if any(map(any, err)) else None,
                capsize=4,
            )
            for b, m in zip(bars, ms, strict=True):
                ax.annotate(
                    f"{m or 0:.1f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        else:
            hit = [(x, m) for x, m in zip(xs, ms, strict=True) if m is not None]
            miss = [x for x, m in zip(xs, ms, strict=True) if m is None]
            ax.scatter(
                [x for x, _ in hit],
                [m for _, m in hit],
                s=90,
                color=COLORS[i % 6],
                label=lab,
                zorder=3,
            )
            ax.scatter(
                miss,
                [cap] * len(miss),
                s=90,
                marker="x",
                color=COLORS[i % 6],
                label=f"{lab} (never, at cap)",
                zorder=3,
            )
    if cap is not None:
        ax.axhline(cap, ls="--", color="grey", lw=1)
        ax.set_ylim(0, cap * 1.1)
    ax.set_xticks(range(len(games)), games)
    ax.set_xlabel("game")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def curves(data: Runs, title: str, path: Path) -> bool:
    games = games_of(data)
    if cap_of(data) < 2:
        return False
    fig, axes = plt.subplots(
        1, len(games), figsize=(4.5 * len(games), 4), sharey=True, squeeze=False
    )
    for ax, g in zip(axes[0], games, strict=True):
        for i, lab in enumerate(data):
            for k, r in enumerate(data[lab]):
                if g in r:
                    c = r[g]["curve"]
                    ax.step(
                        range(1, len(c) + 1),
                        c,
                        where="post",
                        color=COLORS[i % 6],
                        alpha=0.8,
                        label=lab if k == 0 else None,
                    )
        ax.set_title(g)
        ax.set_xlabel("step")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("levels cleared")
    axes[0][-1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def overview(data: Runs, title: str, path: Path) -> None:
    labels, games = list(data), games_of(data)
    panels = (
        (
            "games with a clear",
            lambda lab: float(
                sum((mean(vals(data, lab, g, "cleared")) or 0) > 0 for g in games)
            ),
        ),
        (
            "levels cleared (sum)",
            lambda lab: sum(mean(vals(data, lab, g, "cleared")) or 0 for g in games),
        ),
        (
            "no-change steps (%, mean)",
            lambda lab: statistics.fmean(
                mean(vals(data, lab, g, "idle")) or 0 for g in games
            ),
        ),
        (
            "distinct clicks (sum)",
            lambda lab: sum(mean(vals(data, lab, g, "clicks")) or 0 for g in games),
        ),
    )
    fig, axes = plt.subplots(1, 4, figsize=(max(11, 3.2 * 4), 4))
    for ax, (name, fn) in zip(axes, panels, strict=True):
        ax.bar(
            range(len(labels)),
            [fn(lab) for lab in labels],
            color=[COLORS[i % 6] for i in range(len(labels))],
        )
        for j, lab in enumerate(labels):
            ax.annotate(
                f"{fn(lab):.1f}",
                (j, fn(lab)),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_xticks(range(len(labels)), labels, rotation=20)
        ax.set_ylim(0, max(fn(lab) for lab in labels) * 1.15 or 1)
        ax.set_ylabel(name)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def findings(data: Runs) -> list[str]:
    games, labels = games_of(data), list(data)
    out = []
    for lab in labels:
        lv = {g: mean(vals(data, lab, g, "cleared")) or 0.0 for g in games}
        idle = {g: mean(vals(data, lab, g, "idle")) or 0.0 for g in games}
        best, worst = max(lv, key=lambda g: lv[g]), min(lv, key=lambda g: lv[g])
        out.append(
            f"- {lab}: best game {best} ({lv[best]:.1f} levels), worst {worst} ({lv[worst]:.1f}); "
            f"{sum(v > 0 for v in lv.values())}/{len(games)} games cleared a level; "
            f"highest no-change share {max(idle, key=lambda g: idle[g])} ({max(idle.values()):.0f}% of steps)."
        )
    if len(labels) > 1:
        tot = {
            lab: sum(mean(vals(data, lab, g, "cleared")) or 0 for g in games)
            for lab in labels
        }
        top = max(tot, key=lambda k: tot[k])
        out.append(
            f"- Label difference: {top} cleared the most levels in total ({', '.join(f'{k}={v:.1f}' for k, v in tot.items())}). "
            "Treat this as a hint only: n is tiny and a single run is noisy."
        )
    return out


def report(data: Runs, title: str, out: Path, charts: list[tuple[str, str]]) -> str:
    games = games_of(data)
    table = bs.summarize(
        {lab: [{g: s for g, s in r.items()} for r in rs] for lab, rs in data.items()}
    )
    runs = ", ".join(
        f"{lab} ({len(rs)} run{'s' if len(rs) > 1 else ''})" for lab, rs in data.items()
    )
    lines = [
        f"# {title}",
        "",
        "## Question",
        "",
        (
            "Can the Haiku players clear levels in these games, and how do the variants (labels) differ? "
            "Which games give them trouble and what do the failures look like?"
        ),
        "",
        "## Setup",
        "",
        f"- Labels: {runs}",
        f"- Games: {', '.join(games)}",
        f"- Step cap: {cap_of(data)} actions per game (longest trace)",
        "- Players: Haiku subagents driving the game through semula (docs/SEMULA-HAIKU-LOOP.md); no hosted model API.",
        "",
        "## Results",
        "",
        "```",
        table,
        "```",
        "",
    ]
    for name, cap in charts:
        lines += [f"![{cap}]({name})", ""]
    lines += [
        "## Findings",
        "",
        *findings(data),
        "",
        "## Limits",
        "",
        "- One to a few runs per label and three games: differences are anecdotes, not measurements.",
        "- Haiku players sometimes stopped early, so step counts understate what a full run would do.",
        "- Traces are capped, and clearing earlier is rewarded by the score, which these charts do not compute.",
        "- Distinct clicks and no-change share come from the trace; they say what was tried, not why.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", metavar="label=dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", required=True)
    args = ap.parse_args(argv)
    data = load(args.runs)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cap = cap_of(data)
    grouped(
        data,
        "cleared",
        "levels cleared (mean, bars = min-max)",
        CHARTS[0][1],
        out / "levels.png",
    )
    grouped(
        data, "idle", "steps with no frame change (%)", CHARTS[1][1], out / "idle.png"
    )
    grouped(data, "clicks", "distinct click targets", CHARTS[2][1], out / "clicks.png")
    grouped(
        data,
        "first_clear",
        "step of first clear (x = never)",
        CHARTS[3][1],
        out / "first_clear.png",
        cap=max(cap, 1),
    )
    made = [CHARTS[0], CHARTS[1], CHARTS[2], CHARTS[3]]
    if curves(data, CHARTS[4][1], out / "curves.png"):
        made.append(CHARTS[4])
    overview(data, CHARTS[5][1], out / "overview.png")
    made.append(CHARTS[5])
    (out / "report.md").write_text(report(data, args.title, out, made))
    print(f"wrote {out}/report.md and {len(made)} charts")


if __name__ == "__main__":
    main()
