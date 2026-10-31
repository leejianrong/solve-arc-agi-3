#!/usr/bin/env python3
"""Play real ARC-AGI-3 games offline with the submission agent, against any OpenAI-compatible endpoint.

A fast development loop (minutes, not a one-hour Kaggle run). Needs arc_agi/arcengine, so run it with the
play-arc-agi-3 environment:

    set -a; . ./.env; set +a
    PYTHONPATH=src ../play-arc-agi-3/.venv/bin/python scripts/local_play.py --games 8 --steps 40

Default endpoint is OpenRouter's qwen/qwen3.6-27b: development evidence only, never a performance figure
(see docs/baselines/qwen36-vllm-artifact.md). Plays through the same SubmissionAgent as the Kaggle run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from arc_agi import Arcade, OperationMode

from solve_arc_agi_3.baseline_config import InferenceConfig
from solve_arc_agi_3.inference import OpenAICompatibleClient
from solve_arc_agi_3.kaggle_agent import LoggingClient, RunLimits, make_agent_class
from solve_arc_agi_3.step_trace import (
    RunTrace,
    apply_diff,
    decode_rows,
    finish_run_dir,
    grid_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_DIR = ROOT.parent / "play-arc-agi-3" / "environment_files"


@dataclass
class Frame:
    game_id: str
    frame: list
    state: object
    levels_completed: int
    win_levels: int
    available_actions: list[int]


class Base:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id


class RandomPlanClient:
    """A scripted stand-in for the model: random 1-4 move plans over the actions the prompt says are available.

    Costs nothing and exercises the whole agent (observation encoder, plans, counters, clicks, resets) on real
    frames, so crashes and format bugs show up before a paid or hour-long run. Clicks use the listed object centres.
    """

    def __init__(self, seed: int = 0) -> None:
        import random

        self._rng = random.Random(seed)

    def complete(self, request):
        import re

        from solve_arc_agi_3.inference import FunctionCall, InferenceResponse, ToolCall

        content = request.messages[-1].content
        text = (
            content
            if isinstance(content, str)
            else "".join(getattr(p, "text", "") for p in content)
        )
        avail = [
            int(n)
            for n in re.findall(
                r"ACTION(\d)", text.split("Result of your last action")[0]
            )
        ]
        centres = [
            (int(x), int(y)) for x, y in re.findall(r"center \((\d+),(\d+)\)", text)
        ]
        moves = []
        for _ in range(self._rng.randint(1, 4)):
            a = self._rng.choice(avail or [1])
            if a == 6:
                x, y = self._rng.choice(centres or [(32, 32)])
                moves.append({"action": 6, "x": x, "y": y})
            else:
                moves.append(a)
        call = ToolCall(
            id="c",
            function=FunctionCall(
                name="python", arguments=json.dumps({"code": f"action({moves!r})"})
            ),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


def to_frame(raw) -> Frame:
    return Frame(
        raw.game_id,
        [a.tolist() for a in raw.frame],
        raw.state,
        raw.levels_completed,
        raw.win_levels,
        list(raw.available_actions),
    )


def play(
    game_id: str,
    arc: Arcade,
    card: str,
    agent_cls: type,
    steps: int,
    results: dict,
    truths: dict,
) -> None:
    env = arc.make(game_id, scorecard_id=card)
    agent = agent_cls(game_id)
    frames = [to_frame(env.observation_space)]
    best = 0
    actions: dict[str, int] = {}
    truth = [grid_rows(tuple(map(tuple, frames[0].frame[-1])))]
    for _ in range(steps):
        latest = frames[-1]
        if str(getattr(latest.state, "value", latest.state)) == "WIN":
            break
        action = agent.choose_action(frames, latest)
        actions[action.name] = actions.get(action.name, 0) + 1
        raw = env.step(action, data=action.action_data.model_dump(), reasoning={})
        if raw is None:
            break
        frames.append(to_frame(raw))
        truth.append(grid_rows(tuple(map(tuple, frames[-1].frame[-1]))))
        best = max(best, raw.levels_completed)
    agent.is_done(
        frames, frames[-1]
    )  # as the official loop does; writes the last trace record
    truths[game_id] = truth
    results[game_id] = {
        "levels": best,
        "of": frames[-1].win_levels,
        "fallbacks": agent._fallbacks,
        "steps": len(frames) - 1,
        "actions": actions,
    }


def verify_trace(run_dir: Path, truths: dict) -> None:
    """Decode each game's trace and compare it with the board the game really showed at every step."""

    steps = 0
    for game, truth in truths.items():
        records = [
            json.loads(x) for x in (run_dir / f"{game}.jsonl").read_text().splitlines()
        ]
        rows = decode_rows(records[0]["frame"]["base"])
        assert rows == truth[0], f"{game}: start frame differs"
        for i, rec in enumerate(r for r in records if r["kind"] == "step"):
            frame = rec["frame"]
            rows = (
                decode_rows(frame["key"])
                if "key" in frame
                else apply_diff(rows, frame["diff"])
            )
            assert rows == truth[i + 1], f"{game}: step {i} differs"
            steps += 1
        assert steps and len(records) - 1 == len(truth) - 1, (
            f"{game}: {len(records) - 1} records for {len(truth) - 1} steps"
        )
    print(
        f"TRACE_REPLAY exact: {len(truths)} games, {steps} steps decoded and compared with the real boards"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--games", default="8", help="a count, or comma-separated game id prefixes"
    )
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/baselines/openrouter-qwen36-27b.dev.json",
    )
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--fake",
        action="store_true",
        help="scripted random plans instead of a model (free)",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for --fake")
    ap.add_argument(
        "--nav",
        action="store_true",
        help="state graph and reachability lines in the status header (ARC3_NAV)",
    )
    ap.add_argument(
        "--ledger",
        action="store_true",
        help="put the hypothesis ledger block in the prompt",
    )
    ap.add_argument(
        "--trace-dir",
        type=Path,
        help="write a step trace here and verify it replays exactly",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING, stream=sys.stdout, format="%(levelname)s %(message)s"
    )
    logging.getLogger("solve_arc_agi_3").setLevel(logging.INFO)
    config = InferenceConfig.model_validate_json(args.config.read_text())
    config = config.model_copy(
        update={
            "max_output_tokens": args.max_tokens,
            "request_timeout_seconds": args.timeout,
            "sampling": config.sampling.model_copy(update={"thinking": args.thinking}),
        }
    )
    client = LoggingClient(
        RandomPlanClient(args.seed) if args.fake else OpenAICompatibleClient(config),
        samples=4,
        every=400,
    )
    limits = RunLimits(
        deadline_epoch=time.time() + 3600, max_concurrent_calls=args.concurrency
    )
    trace = (
        RunTrace(args.trace_dir, settings={"fake": args.fake})
        if args.trace_dir
        else None
    )
    agent_cls = make_agent_class(
        Base,
        limits=limits,
        client=client,
        trace=trace,
        use_ledger=args.ledger,
        nav=args.nav,
    )

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(args.env_dir),
        recordings_dir="/tmp/local_play_recordings",
    )
    ids = sorted(
        e.game_id for e in arc.get_environments() if not e.game_id.startswith("r11l")
    )  # held out in play-arc-agi-3
    ids = (
        ids[: int(args.games)]
        if args.games.isdigit()
        else [g for g in ids if g.startswith(tuple(args.games.split(",")))]
    )
    card = arc.open_scorecard(tags=["local"])
    results: dict = {}
    truths: dict = {}
    threads = [
        threading.Thread(
            target=play, args=(g, arc, card, agent_cls, args.steps, results, truths)
        )
        for g in ids
    ]
    t0 = time.time()
    [t.start() for t in threads]
    [t.join() for t in threads]
    scorecard = arc.close_scorecard(card)
    if trace is not None:
        dump = scorecard.model_dump() if scorecard else None
        baselines = {
            env["id"]: run["level_baseline_actions"]
            for env in (dump or {}).get("environments", [])
            for run in env.get("runs", [])[:1]
        }
        finish_run_dir(trace.run_dir, dump, baselines)
        verify_trace(trace.run_dir, truths)
    print(json.dumps(results, indent=1))
    if scorecard is not None:
        for env in scorecard.model_dump().get("environments", []):
            for run in env.get("runs", []):
                if run.get("levels_completed"):
                    print(
                        "SCORED",
                        env["id"],
                        "levels",
                        run["levels_completed"],
                        "actions",
                        run["actions"],
                        "level_scores",
                        run["level_scores"],
                        "level_actions",
                        run["level_actions"],
                        "baseline",
                        run["level_baseline_actions"],
                    )
    print(
        f"games={len(ids)} levels_total={sum(r['levels'] for r in results.values())} "
        f"fallbacks={sum(r['fallbacks'] for r in results.values())} steps={sum(r['steps'] for r in results.values())} "
        f"score={getattr(scorecard, 'score', None)} wall={time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
