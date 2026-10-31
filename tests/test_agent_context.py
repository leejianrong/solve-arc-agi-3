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
    assert (
        "1. ACTION1 -> NOT_FINISHED, 32 cells changed" in text
        and "LEVEL CLEARED (+1)" in text
    )
    assert (
        text.count("Grid (rows top to bottom") == 1
    )  # only the current frame is sent in full


# Prompt compaction (prompt_chars) ----------------------------------------------------------------------


class ScriptedClient:
    def __init__(self, *codes: str) -> None:
        self.codes, self.texts = list(codes), []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.texts.append(_user_text(request))
        call = ToolCall(
            id="c",
            function=FunctionCall(
                name="python", arguments=json.dumps({"code": self.codes.pop(0)})
            ),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


def _dots(x: int, counter: int = 0) -> Observation:
    rows = [[0] * 24 for _ in range(14)]
    for dx in range(2):
        for dy in range(2):
            rows[8 + dy][x + dx] = 5
    for col in range(0, 24, 2):
        rows[1][col] = 3
        rows[3][col] = 4
    rows[13][1] = 1 + counter % 2
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=3,
        available_actions=(1, 2, 3),
    )


def _final_prompt(prompt_chars: int, *, ledger: bool = True) -> tuple[str, int]:
    """The prompt of the 4th call, after a two-move plan so every trimmable part is present."""

    client = ScriptedClient("action(3)", "action(3)", "action([3, 3])", "action(3)")
    core = BaselineAgentCore(
        client, use_ledger=ledger, prompt_chars=prompt_chars, ledger_chars=600
    )
    for i in range(5):
        obs = _dots(2 + i, counter=i)
        if core.queued_decision(obs) is None:
            core.choose(obs)
    return client.texts[-1], core.compaction_overflows


def test_unlimited_budget_changes_nothing() -> None:
    plain, _ = _final_prompt(0)
    assert _final_prompt(10**9)[0] == plain
    assert "Results of your last plan" in plain and "Recent steps" in plain


def test_budget_is_honoured_and_mandatory_parts_survive() -> None:
    full, _ = _final_prompt(0)
    for budget in (len(full) - 50, len(full) - 400, len(full) - 900):
        text, overflows = _final_prompt(budget)
        assert len(text) <= budget and overflows == 0
        for must in (
            "Game state:",
            "Result of your last action:",
            "What each action has done so far:",
            "Grid (rows top to bottom",
            "background color",
        ):
            assert must in text


def test_trim_order_is_plan_results_then_history_then_ledger_then_objects() -> None:
    full, _ = _final_prompt(0)
    seen = []
    for budget in range(len(full), 600, -20):
        text, overflows = _final_prompt(budget)
        if overflows:
            break
        seen.append(
            (
                "Results of your last plan" in text,
                text.count("\n") and "Recent steps" in text,
                "Hypotheses so far" in text,
                "smaller objects not listed" in text,
            )
        )
    # each part, once gone, stays gone as the budget shrinks, and they go in the documented order
    first_gone = [
        next((i for i, s in enumerate(seen) if not s[k]), 10**6) for k in range(3)
    ]
    assert first_gone[0] <= first_gone[1] <= first_gone[2] < 10**6
    assert seen[-1][3]  # the object list was cut last, and says so
    assert not seen[0][3]


def test_same_input_gives_the_same_prompt() -> None:
    full, _ = _final_prompt(0)
    budget = len(full) - 700
    assert _final_prompt(budget) == _final_prompt(budget)


def test_mandatory_parts_over_budget_are_returned_and_counted() -> None:
    text, overflows = _final_prompt(10)
    assert overflows >= 1
    assert "Grid (rows top to bottom" in text and "Result of your last action:" in text
    assert "Recent steps" not in text and "Results of your last plan" not in text


def test_warnings_are_never_trimmed() -> None:
    client = ScriptedClient(*["action(1)"] * 5)
    core = BaselineAgentCore(client, prompt_chars=900)
    for _ in range(4):
        obs = _observation(5)
        if core.queued_decision(obs) is None:
            core.choose(obs)
    assert "WARNING: your last" in client.texts[-1]
