"""Event-driven calls (ARC-49): a contradicted ledger claim drops the plan and says why, off unless asked for."""

from __future__ import annotations

import json
from pathlib import Path

from solve_arc_agi_3.agent import BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)
from solve_arc_agi_3.kaggle_agent import RunLimits, make_agent_class
from solve_arc_agi_3.step_trace import RunTrace


class Scripted:
    def __init__(self, *codes: str) -> None:
        self.codes, self.texts = list(codes), []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        content = request.messages[-1].content
        self.texts.append("".join(getattr(p, "text", "") for p in content))  # type: ignore[union-attr,arg-type]
        call = ToolCall(
            id="c",
            function=FunctionCall(
                name="python", arguments=json.dumps({"code": self.codes.pop(0)})
            ),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


def board(x: int, counter: int = 0, levels: int = 0) -> Observation:
    rows = [[0] * 30 for _ in range(14)]
    for dx in range(2):
        for dy in range(2):
            rows[5 + dy][x + dx] = 5
    rows[13][1] = 1 + counter % 2
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=levels,
        win_levels=3,
        available_actions=(1, 2, 3),
    )


def drive(core: BaselineAgentCore, frames: list[Observation]) -> list[str]:
    """What happened at each frame: 'call' or the queued move name."""
    out = []
    for obs in frames:
        queued = core.queued_decision(obs)
        if queued is None:
            core.choose(obs)
            out.append("call")
        else:
            out.append(queued.name)
    return out


# ACTION3 moves the block by +5 three times (a supported claim), then the fourth move changes nothing.
WALL = [board(2 + 5 * i, counter=i) for i in range(4)] + [
    board(17, counter=4),
    board(17, counter=5),
]


def test_a_contradicted_supported_claim_drops_the_plan_and_names_the_surprise() -> None:
    client = Scripted("action([3, 3, 3, 3, 3])", "action(1)", "action(1)")
    core = BaselineAgentCore(client, events=True)
    moves = drive(core, WALL[:5])
    assert moves == ["call", "ACTION3", "ACTION3", "ACTION3", "call"]
    assert core.last_event == "surprise"
    assert core.last_turn["event"] == "surprise"
    text = client.texts[-1]
    assert (
        "SURPRISE: ACTION3 was expected to move the object (colour 5, size 4) by (5, 0) but nothing moved"
        in text
    )
    assert "rest of your plan was dropped" in text
    drive(core, [board(17, counter=5)])
    assert core.last_event != "surprise"  # the next call has its own reason


def test_without_the_switch_the_plan_keeps_going() -> None:
    client = Scripted("action([3, 3, 3, 3, 3])")
    core = BaselineAgentCore(client)
    moves = drive(core, WALL[:5])
    assert moves == ["call"] + ["ACTION3"] * 4
    assert "event" not in core.last_turn
    assert "SURPRISE" not in client.texts[0]


def test_a_different_delta_is_a_surprise_too() -> None:
    client = Scripted("action([3, 3, 3, 3, 3])", "action(1)")
    core = BaselineAgentCore(client, events=True)
    frames = [board(2 + 5 * i, counter=i) for i in range(4)] + [board(12, counter=4)]
    assert drive(core, frames)[-1] == "call"
    assert "but it moved by [(-5, 0)]" in client.texts[-1]


def test_an_unproven_claim_is_not_a_surprise() -> None:
    client = Scripted("action([3, 3, 3])", "action(1)")
    core = BaselineAgentCore(client, events=True)
    frames = [board(2, 0), board(7, 1), board(7, 2), board(7, 3)]
    assert drive(core, frames) == ["call", "ACTION3", "ACTION3", "call"]
    assert "SURPRISE" not in client.texts[-1]


def test_call_reasons_are_level_start_plan_exhausted_and_no_progress() -> None:
    client = Scripted("action(3)", "action(3)", "action(3)", "action(1)", "action(1)")
    core = BaselineAgentCore(client, events=True)
    seen = []
    for obs in [
        board(2, 0),
        board(7, 1),
        board(12, 2, levels=1),
        board(12, 2, levels=1),
        board(12, 2, levels=1),
    ]:
        drive(core, [obs])
        seen.append(core.last_event)
    assert seen == [
        "level_start",
        "plan_exhausted",
        "level_start",
        "plan_exhausted",
        "no_progress",
    ]


def test_events_are_written_to_the_trace_only_when_on(tmp_path: Path) -> None:
    class Base:
        def __init__(self, *a: object, **k: object) -> None:
            self.game_id = "g"

    class Frame:
        def __init__(self, obs: Observation) -> None:
            self.frame = [[list(r) for r in obs.grid]]
            self.state = obs.state
            self.levels_completed = obs.levels_completed
            self.win_levels = obs.win_levels
            self.available_actions = list(obs.available_actions)
            self.game_id = "g"

    for events, run_id in ((True, "on"), (False, "off")):
        trace = RunTrace(tmp_path, run_id=run_id)
        agent = make_agent_class(
            Base,
            limits=RunLimits(deadline_epoch=1e12),
            client=Scripted("action(3)", "action(3)", "action(3)"),
            action_mapper=lambda d: d,
            trace=trace,
            events=events,
            prompt_chars=0,
            tools=False,
            world_model=False,
        )()
        for obs in WALL[:3]:
            frame = Frame(obs)
            agent.is_done([], frame)
            agent.choose_action([], frame)
        agent.is_done([], Frame(WALL[3]))
        lines = [
            json.loads(x) for x in (trace.run_dir / "g.jsonl").read_text().splitlines()
        ]
        steps = [r for r in lines if r["kind"] == "step"]
        if events:
            assert [r.get("event") for r in steps] == [
                "level_start",
                "plan_exhausted",
                "plan_exhausted",
            ]
        else:
            assert all("event" not in r for r in steps)
