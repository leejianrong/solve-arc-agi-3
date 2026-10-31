"""The step trace (docs/TRACE.md): encoding, file layout, and a scripted agent run that must replay exactly."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solve_arc_agi_3.agent import ActionDecision
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    TokenUsage,
    ToolCall,
)
from solve_arc_agi_3.kaggle_agent import RunLimits, make_agent_class
from solve_arc_agi_3.step_trace import (
    KEYFRAME_RUNS,
    RunTrace,
    apply_diff,
    decode_rows,
    diff_runs,
    encode_rows,
    finish_run_dir,
    grid_rows,
)


def random_rows(rng: random.Random, blocks: int = 6) -> list[str]:
    grid = [["5"] * 64 for _ in range(64)]
    for _ in range(blocks):
        x, y, c = rng.randrange(56), rng.randrange(56), rng.choice("0123456789abcdef")
        for dy in range(rng.randrange(1, 8)):
            for dx in range(rng.randrange(1, 8)):
                grid[y + dy][x + dx] = c
    return ["".join(r) for r in grid]


def test_row_encoding_round_trips_and_folds_repeated_rows() -> None:
    rng = random.Random(1)
    for _ in range(20):
        rows = random_rows(rng)
        assert decode_rows(encode_rows(rows)) == rows
    flat = ["5" * 64] * 64
    assert encode_rows(flat) == ["564~63"]
    assert decode_rows(["51.363"]) == ["5" + "3" * 63]


def test_diffs_apply_back_to_the_next_frame_with_inclusive_ends() -> None:
    rng = random.Random(2)
    before, after = random_rows(rng), random_rows(rng)
    runs = diff_runs(before, after)
    assert apply_diff(before, runs) == after
    assert diff_runs(["0000"], ["0110"]) == [
        [0, 1, 2, 1]
    ]  # columns 1 to 2 inclusive, colour 1


@dataclass
class Frame:
    game_id: str
    frame: list[list[list[int]]]
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]


class TinyGame:
    """12x20 board, a 2x2 block that ACTION1 moves right by 3 and ACTION2 moves down; a counter cell flips each move."""

    def __init__(self) -> None:
        self.x, self.y, self.moves, self.over_at = 2, 4, 0, 6
        self.state = "NOT_PLAYED"

    def grid(self) -> list[list[int]]:
        rows = [[0] * 20 for _ in range(12)]
        for dx in range(2):
            for dy in range(2):
                rows[self.y + dy][self.x + dx] = 5
        rows[11][1] = 1 + self.moves % 2
        return rows

    def frame(self) -> Frame:
        return Frame("tiny", [self.grid()], self.state, 0, 2, [1, 2])

    def play(self, decision: ActionDecision) -> None:
        if decision.name == "RESET":
            self.x, self.y, self.state = 2, 4, "NOT_FINISHED"
            return
        self.moves += 1
        if decision.name == "ACTION1":
            self.x = min(self.x + 3, 17)
        elif decision.name == "ACTION2":
            self.y = min(self.y + 3, 8)
        self.state = "GAME_OVER" if self.moves == self.over_at else "NOT_FINISHED"


class Scripted:
    """Replies with fixed codes in order; ``None`` raises as a failed request would."""

    def __init__(self, *codes: str | None) -> None:
        self.codes = list(codes)

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        code = self.codes.pop(0)
        if code is None:
            raise TimeoutError("timed out")
        call = ToolCall(
            id="c",
            function=FunctionCall(name="python", arguments=json.dumps({"code": code})),
        )
        return InferenceResponse(
            content="thinking aloud",
            tool_calls=(call,),
            elapsed_seconds=0.25,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=12, total_tokens=112),
        )


class Base:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.game_id = "tiny"


def run(trace: RunTrace | None, client: Scripted, steps: int = 9) -> TinyGame:
    limits = RunLimits(deadline_epoch=time.time() + 3600)
    agent = make_agent_class(
        Base, limits=limits, client=client, action_mapper=lambda d: d, trace=trace
    )()
    game = TinyGame()
    frames = [
        Frame("tiny", [], "NOT_PLAYED", 0, 2, [])
    ]  # the placeholder frame the official loop starts with
    for _ in range(steps):
        if agent.is_done(frames, frames[-1]):
            break
        decision = agent.choose_action(frames, game.frame())
        game.play(decision)
        frames.append(game.frame())
    agent.is_done(
        frames, frames[-1]
    )  # flushes the last move, as the official loop does
    return game


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_scripted_run_writes_a_trace_that_replays_to_the_true_boards(
    tmp_path: Path,
) -> None:
    trace = RunTrace(tmp_path, run_id="r1", settings={"thinking": False}, keep_obs=True)
    # reset, then a 3-move plan, then a failed request (fallback), then a plan, then game over -> forced reset
    client = Scripted(
        "action([1, 2, 1])", None, "action(2)", "action(1)", "action(1)", "action(1)"
    )
    run(trace, client)
    records = read(tmp_path / "r1" / "tiny.jsonl")
    assert (
        records[0]["kind"] == "start"
        and records[0]["game"] == "tiny"
        and "base" in records[0]["frame"]
    )
    steps = [r for r in records if r["kind"] == "step"]
    assert [r["step"] for r in steps] == list(range(len(steps)))
    srcs = [r["src"] for r in steps]
    assert (
        srcs[0] == "forced_reset"
        and "model" in srcs
        and "queued" in srcs
        and "fallback" in srcs
    )
    assert (
        srcs.count("forced_reset") >= 2
    )  # the opening reset and the one after game over

    # replay: decode the trace and compare with what the game really showed
    game, expected = TinyGame(), []
    rows = decode_rows(records[0]["frame"]["base"])
    assert rows == grid_rows(tuple(map(tuple, game.grid())))
    for r in steps:
        rows = (
            decode_rows(r["frame"]["key"])
            if "key" in r["frame"]
            else apply_diff(rows, r["frame"]["diff"])
        )
        expected.append(rows)
    game = TinyGame()
    seen = []
    for r in steps:
        m = r["moves"][0]
        game.play(ActionDecision(name=m["name"], source="replay"))
        seen.append(grid_rows(tuple(map(tuple, game.grid()))))
    assert seen == expected  # exact, step by step

    model = next(r for r in steps if r["src"] == "model")
    assert model["call"] == {
        "ms": 250,
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "ok": True,
    }
    assert (
        model["reply"]["content"] == "thinking aloud"
        and model["plan"]
        and model["plan_idx"] == 0
    )
    queued = [r for r in steps if r["src"] == "queued"]
    assert [q["plan_idx"] for q in queued[:2]] == [1, 2] and "call" not in queued[0]
    fallback = next(r for r in steps if r["src"] == "fallback")
    assert fallback["error"].startswith("TimeoutError") and fallback["call"] == {
        "ok": False
    }
    moved = [r for r in steps if r.get("motion")]
    assert (
        moved
        and moved[0]["motion"][0]["color"] == 5
        and moved[0]["motion"][0]["size"] == 4
    )
    assert all(r["change"]["n"] >= 0 for r in steps if "change" in r)

    obs = read(tmp_path / "r1" / "tiny.obs.jsonl")
    assert obs and "Grid (rows top to bottom" in obs[0]["text"]
    meta = json.loads((tmp_path / "r1" / "run.json").read_text())
    assert meta["games"] == ["tiny"] and meta["settings"] == {"thinking": False}


def test_a_big_change_is_stored_as_a_keyframe_and_a_small_one_as_a_diff(
    tmp_path: Path,
) -> None:
    trace = RunTrace(tmp_path, run_id="r2")
    game = trace.game("g")
    flat = [[5] * 64 for _ in range(64)]
    game.start("g", tuple(map(tuple, flat)), "NOT_FINISHED", 0, 1, [1])
    small = [row[:] for row in flat]
    small[3][4] = 9
    noisy = [[(x * 7 + y) % 16 for x in range(64)] for y in range(64)]
    for grid in (small, noisy):
        game.step(
            game="g",
            src="model",
            move={"name": "ACTION1"},
            grid=tuple(map(tuple, grid)),
            state="NOT_FINISHED",
            levels=0,
            win_levels=1,
            avail=[1],
        )
    first, second = read(tmp_path / "r2" / "g.jsonl")[1:]
    assert first["frame"] == {"diff": [[3, 4, 4, 9]]}
    assert (
        "key" in second["frame"]
        and len(
            diff_runs(
                grid_rows(tuple(map(tuple, small))), grid_rows(tuple(map(tuple, noisy)))
            )
        )
        > KEYFRAME_RUNS
    )


def test_tracing_off_writes_nothing_and_the_agent_still_plays(tmp_path: Path) -> None:
    game = run(
        None,
        Scripted(
            "action([1, 2, 1])",
            "action(2)",
            "action(1)",
            "action(1)",
            "action(1)",
            "action(1)",
        ),
    )
    assert game.moves > 0 and not list(tmp_path.iterdir())


def test_finishing_a_run_records_the_scorecard_and_baselines(tmp_path: Path) -> None:
    trace = RunTrace(tmp_path, run_id="r3")
    finish_run_dir(trace.run_dir, {"score": 0.5}, {"g": [10, 20]})
    meta = json.loads((trace.run_dir / "run.json").read_text())
    assert (
        meta["scorecard"] == {"score": 0.5}
        and meta["baseline_actions"] == {"g": [10, 20]}
        and "finished_at" in meta
    )


def test_a_killed_run_leaves_every_written_line_readable(tmp_path: Path) -> None:
    trace = RunTrace(tmp_path, run_id="r4")
    game = trace.game("g")
    game.start("g", ((5, 5), (5, 5)), "NOT_FINISHED", 0, 1, [1])
    game.step(
        game="g",
        src="model",
        move={"name": "ACTION1"},
        grid=((5, 5), (5, 9)),
        state="NOT_FINISHED",
        levels=0,
        win_levels=1,
        avail=[1],
    )
    lines = (tmp_path / "r4" / "g.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(line)["v"] == 1 for line in lines)


def test_event_is_written_only_when_given(tmp_path: Path) -> None:
    run = RunTrace(tmp_path, run_id="r")
    game = run.game("g")
    base: dict[str, Any] = {
        "game": "g",
        "src": "model",
        "move": {"name": "ACTION1"},
        "grid": ((0, 1), (1, 0)),
        "state": "NOT_FINISHED",
        "levels": 0,
        "win_levels": 1,
        "avail": (1,),
    }
    game.step(**base)
    game.step(**base, event="surprise")
    lines = [json.loads(x) for x in (run.run_dir / "g.jsonl").read_text().splitlines()]
    assert "event" not in lines[0] and lines[1]["event"] == "surprise"
    assert lines[1]["v"] == 1
