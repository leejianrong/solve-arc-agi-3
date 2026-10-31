"""The hypothesis ledger in the prompt (PLAN.md): computed from the agent's own moves, off unless asked for."""

from __future__ import annotations

import json

from solve_arc_agi_3.agent import BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)


class Recording:
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


def board(x: int, levels: int = 0, counter: int = 0) -> Observation:
    rows = [[0] * 24 for _ in range(14)]
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


def play(core: BaselineAgentCore, frames: list[Observation]) -> None:
    for obs in frames:
        core.choose(obs)


def test_the_ledger_is_off_by_default() -> None:
    client = Recording(*["action(3)"] * 4)
    play(BaselineAgentCore(client), [board(2 + 5 * i, counter=i) for i in range(4)])
    assert "Hypotheses so far" not in client.texts[-1]


def test_computed_claims_with_evidence_appear_once_moves_have_results() -> None:
    client = Recording(*["action(3)"] * 5)
    core = BaselineAgentCore(client, use_ledger=True)
    play(core, [board(2 + 5 * i, counter=i) for i in range(5)])
    assert "Hypotheses so far" not in client.texts[0]  # nothing settled yet
    last = client.texts[-1]
    assert "Hypotheses so far, computed from your moves:" in last
    assert "ACTION3 moves the object (colour 5, size 4) by (5, 0)" in last
    assert (
        "supported" in last
    )  # three or more consistent supports and no counterexample


def test_a_cleared_level_becomes_goal_evidence_with_the_moves_before_it() -> None:
    client = Recording("action(3)", "action(2)", "action(1)", "action(1)")
    core = BaselineAgentCore(client, use_ledger=True)
    play(core, [board(2), board(7), board(7, levels=1), board(7, levels=1)])
    assert "level 1 reached after ACTION3" in client.texts[-1]


def test_the_block_never_exceeds_its_budget() -> None:
    client = Recording(*["action(3)"] * 8)
    core = BaselineAgentCore(client, use_ledger=True, ledger_chars=150)
    play(core, [board(2 + (i % 4) * 3, counter=i) for i in range(8)])
    header = "Hypotheses so far, computed from your moves:\n"
    block = core._ledger_block()
    assert block.startswith(header) and len(block) <= len(header) + 150 + 1
    assert block in client.texts[-1]
