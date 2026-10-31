#!/usr/bin/env python3
"""Summarise a downloaded Kaggle run folder (recordings, log, optional trace) into one comparable record.

    scripts/summarize_run.py build/dev-runs/ledger                  # one run
    scripts/summarize_run.py build/dev-runs/base build/dev-runs/ledger   # side by side

Ground truth for actions and levels is the recordings (or the trace when there is one): the Kaggle log prints
every agent line twice, so counts taken from log lines are doubled. Writes ``summary.json`` into each folder.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def _log_text(folder: Path) -> str:
    out: list[str] = []
    for path in folder.glob("*.log"):
        if path.name.startswith("vllm"):
            continue
        for raw in path.read_text(errors="replace").splitlines():
            raw = raw.strip().lstrip("[,").rstrip("]")
            try:
                out.append(json.loads(raw).get("data", ""))
            except (json.JSONDecodeError, AttributeError):
                continue
    return "".join(out)


def summarize(folder: Path) -> dict[str, Any]:
    games: dict[str, dict[str, Any]] = {}
    for rec in folder.glob("recordings/*.jsonl"):
        rows = [
            json.loads(line)["data"]
            for line in rec.read_text().splitlines()
            if line.strip()
        ]
        game = rec.name.split(".")[0]
        games[game] = {
            "actions": max(len(rows) - 1, 0),
            "levels": max((r["levels_completed"] for r in rows), default=0),
            "of": rows[-1]["win_levels"] if rows else 0,
            "fallbacks": 0,
        }
    log = _log_text(folder)
    for game, step in {
        (m.group(1), m.group(2))
        for m in re.finditer(r"model step failed for (\S+) \(fallback #(\d+)", log)
    }:
        games.setdefault(game, {"actions": 0, "levels": 0, "of": 0, "fallbacks": 0})[
            "fallbacks"
        ] += 1
    calls = tokens = seconds = 0.0
    stats = re.findall(
        r"model stats: (\d+) calls, mean (\d+) completion tokens, mean ([\d.]+)s", log
    )
    if stats:
        n, tok, sec = stats[-1]
        calls, tokens, seconds = float(n), float(tok), float(sec)
    traces = sorted(folder.glob("traces/*/run.json"))
    plans = {"model": 0, "queued": 0, "fallback": 0, "forced_reset": 0}
    for run_json in traces[:1]:
        for path in run_json.parent.glob("*.jsonl"):
            if path.name.endswith(".obs.jsonl"):
                continue
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("kind") == "step":
                    plans[rec["src"]] = plans.get(rec["src"], 0) + 1
    score = None
    match = re.search(r'SCORECARD \{[^\n]*?"score": ([0-9.]+)', log)
    if match:
        score = float(match.group(1))
    summary_file = folder / "run_summary.json"
    meta = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    actions = sum(g["actions"] for g in games.values())
    result = {
        "run": str(folder),
        "bundle_commit": str(meta.get("bundle_commit", ""))[:7],
        "elapsed_s": meta.get("elapsed_seconds"),
        "games": len(games),
        "actions": actions,
        "levels": sum(g["levels"] for g in games.values()),
        "levels_total": sum(g["of"] for g in games.values()),
        "fallbacks": sum(g["fallbacks"] for g in games.values()),
        "score": score,
        "model_calls": int(calls),
        "moves_per_call": round(actions / calls, 2) if calls else None,
        "mean_tokens": int(tokens),
        "mean_seconds": seconds,
        "step_sources": plans if any(plans.values()) else None,
        "per_game": games,
    }
    (folder / "summary.json").write_text(json.dumps(result, indent=1))
    return result


def render(result: dict[str, Any]) -> str:
    lines = [
        f"{result['run']}  commit {result['bundle_commit']}  elapsed {result['elapsed_s']}s",
        (
            f"  games {result['games']}  actions {result['actions']}  "
            f"levels {result['levels']} of {result['levels_total']}  score {result['score']}"
        ),
        (
            f"  fallbacks {result['fallbacks']}  model calls {result['model_calls']}  "
            f"moves/call {result['moves_per_call']}  mean {result['mean_tokens']} tok, {result['mean_seconds']} s"
        ),
    ]
    if result["step_sources"]:
        lines.append(f"  step sources {result['step_sources']}")
    cleared = {g: v for g, v in result["per_game"].items() if v["levels"]}
    lines.append(
        f"  levels cleared in: {', '.join(f'{g[:4]} {v["levels"]}/{v["of"]} ({v["actions"]} actions)' for g, v in cleared.items()) or 'none'}"
    )
    return "\n".join(lines)


def main() -> None:
    folders = [Path(a) for a in sys.argv[1:]]
    if not folders:
        raise SystemExit(__doc__)
    for folder in folders:
        print(render(summarize(folder)))


if __name__ == "__main__":
    main()
