"""Status header at the top of each observation, and one action-bearing tool call per turn."""

from __future__ import annotations

import json

from solve_arc_agi_3.agent import BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)


def _call(code: str) -> ToolCall:
    return ToolCall(
        id="c",
        function=FunctionCall(name="python", arguments=json.dumps({"code": code})),
    )


class Parallel:
    def __init__(self, *replies: tuple[str, ...]) -> None:
        self.replies, self.texts = list(replies), []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.texts.append(
            "".join(getattr(p, "text", "") for p in request.messages[-1].content)  # type: ignore[union-attr,arg-type]
        )
        calls = tuple(_call(c) for c in self.replies.pop(0))
        return InferenceResponse(tool_calls=calls, elapsed_seconds=0.0)


def board(x: int, counter: int = 0) -> Observation:
    rows = [[0] * 24 for _ in range(14)]
    for dx in range(2):
        for dy in range(2):
            rows[5 + dy][x + dx] = 5
    rows[13][1] = 1 + counter % 2
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=1,
        available_actions=(1, 2, 3, 4),
    )


def test_header_reports_position_moved_blocked_and_untried() -> None:
    client = Parallel(("action([4])",), ("action([4])",), ("action([4])",))
    core = BaselineAgentCore(client)
    core.choose(board(3))
    core.choose(board(5))
    assert client.texts[1].startswith(
        "STATUS: Controlled object (colour 5, size 4) is at x5-6 y5-6"
    )
    assert "MOVED the object by (+2,+0)" in client.texts[1]
    assert "Not tried yet in this game: ACTION1, ACTION2, ACTION3." in client.texts[1]
    core.choose(board(5, counter=1))  # only the counter cell changed
    assert "BLOCKED" in client.texts[2].split("\n")[0]
    assert client.texts[2].count("Not tried yet") == 1


def test_parallel_tool_calls_use_only_the_first_action_call() -> None:
    client = Parallel(
        ("x = 1", "action([4])", "action([1])"),
        ("action([4])",),
    )
    core = BaselineAgentCore(client, tools=True)
    turn = core.choose(board(3))
    assert turn.decision.name == "ACTION4"
    assert core._ignored_calls == 1
    core.choose(board(5))
    assert "only the first action-bearing call was used" in client.texts[1]
