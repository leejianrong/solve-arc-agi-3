"""The agent's prompt stays bounded however long the game runs."""

from __future__ import annotations

import json

from solve_arc_agi_3.agent import BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    TextContentPart,
    ToolCall,
)


class RecordingClient:
    def __init__(self) -> None:
        self.requests: list[InferenceRequest] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.requests.append(request)
        call = ToolCall(
            id="c",
            function=FunctionCall(
                name="python", arguments=json.dumps({"code": "action(['ACTION1'])"})
            ),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


def _observation(fill: int, levels: int = 0) -> Observation:
    return Observation(
        game_id="g",
        grid=tuple(
            tuple(fill if (r + c) % 2 else 0 for c in range(8)) for r in range(8)
        ),
        state="NOT_FINISHED",
        levels_completed=levels,
        win_levels=3,
        available_actions=(1, 2),
    )


def _user_text(request: InferenceRequest) -> str:
    content = request.messages[-1].content
    assert not isinstance(content, str)
    return "".join(p.text for p in content if isinstance(p, TextContentPart))


def test_request_size_does_not_grow_with_game_length() -> None:
    client = RecordingClient()
    core = BaselineAgentCore(client, history_lines=5)
    for step in range(60):
        core.choose(_observation(step % 15 + 1))
    sizes = [len(_user_text(r)) for r in client.requests]
    assert (
        len(client.requests[-1].messages) == 2
    )  # system + one user message, never accumulating
    assert max(sizes[10:]) <= sizes[10] * 1.2
    assert len(core.history) == 5


def test_history_describes_effects_and_old_frames_are_not_resent() -> None:
    client = RecordingClient()
    core = BaselineAgentCore(client)
    core.choose(_observation(3))
    core.choose(_observation(5, levels=1))
    text = _user_text(client.requests[-1])
    assert "1. ACTION1 -> NOT_FINISHED, 32 cells changed, levels +1" in text
    assert text.count('"grid"') == 1  # only the current frame is sent in full
