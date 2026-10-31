"""Multi-move plans, click parsing and per-action memory."""

from __future__ import annotations

import json

import pytest

from solve_arc_agi_3.agent import (
    BaselineAgentCore,
    Observation,
    parse_action_decision,
    parse_action_decisions,
)
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)


def reply(code: str) -> InferenceResponse:
    call = ToolCall(
        id="c",
        function=FunctionCall(name="python", arguments=json.dumps({"code": code})),
    )
    return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


class Scripted:
    def __init__(self, *codes: str) -> None:
        self.codes = list(codes)
        self.calls = 0

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        return reply(self.codes.pop(0))


def obs(
    fill: int = 0,
    *,
    state: str = "NOT_FINISHED",
    levels: int = 0,
    actions: tuple[int, ...] = (1, 2, 3, 6),
) -> Observation:
    grid = tuple(
        tuple(fill if (r, c) == (0, 0) else 1 for c in range(4)) for r in range(4)
    )
    return Observation(
        game_id="g",
        grid=grid,
        state=state,
        levels_completed=levels,
        win_levels=3,
        available_actions=actions,
    )


AV = (1, 2, 3, 4, 5, 6, 7)


@pytest.mark.parametrize(
    "code",
    [
        'action({"action": 6, "x": 12, "y": 30})',
        "action(6, 12, 30)",
        "action(6, x=12, y=30)",
        'action("ACTION6", 12, 30)',
        'action([{"action": "ACTION6", "column": 12, "row": 30}])',
    ],
)
def test_click_forms_all_parse_to_the_same_decision(code: str) -> None:
    decision = parse_action_decision(reply(code), available_actions=AV)
    assert (decision.name, decision.data) == ("ACTION6", {"x": 12, "y": 30})


def test_a_click_without_coordinates_is_not_a_valid_move() -> None:
    for code in ("action(6)", 'action("ACTION6")', "action([6])"):
        with pytest.raises(ValueError):
            parse_action_decision(reply(code), available_actions=AV)


def test_a_plan_keeps_order_drops_illegal_moves_and_is_capped() -> None:
    decisions = parse_action_decisions(
        reply("action([1, 2, 99, 3, 1, 2, 3, 1])"), available_actions=(1, 2, 3)
    )
    assert [d.name for d in decisions] == [
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION1",
        "ACTION2",
    ]


def test_extra_moves_in_a_plan_are_played_without_another_model_call() -> None:
    client = Scripted("action([1, 2, 3])")
    core = BaselineAgentCore(client)
    first = core.choose(obs(0)).decision
    assert first.name == "ACTION1"
    second = core.queued_decision(obs(2))
    third = core.queued_decision(obs(3))
    assert (second.name, third.name) == ("ACTION2", "ACTION3")  # type: ignore[union-attr]
    assert core.queued_decision(obs(4)) is None  # plan spent
    assert client.calls == 1


def test_a_plan_is_dropped_when_a_level_clears_or_the_game_ends() -> None:
    for after in (obs(2, levels=1), obs(2, state="GAME_OVER")):
        core = BaselineAgentCore(Scripted("action([1, 2, 3])"))
        core.choose(obs(0))
        assert core.queued_decision(after) is None


def test_a_plan_is_dropped_after_two_moves_that_changed_nothing() -> None:
    core = BaselineAgentCore(Scripted("action([1, 1, 1, 1])"))
    core.choose(obs(0))
    assert (
        core.queued_decision(obs(0)) is not None
    )  # first move had no effect: still one more allowed
    assert core.queued_decision(obs(0)) is None  # two no-ops in a row


def test_the_model_is_told_what_each_action_has_done_and_warned_about_no_ops() -> None:
    seen: list[str] = []

    class Client(Scripted):
        def complete(self, request: InferenceRequest) -> InferenceResponse:
            content = request.messages[-1].content
            seen.append("".join(getattr(p, "text", "") for p in content))  # type: ignore[arg-type,union-attr]
            return super().complete(request)

    core = BaselineAgentCore(Client("action(1)", "action(1)", "action(1)"))
    core.choose(obs(0))
    core.choose(obs(0))
    core.choose(obs(0))
    last = seen[-1]
    assert "ACTION1 changed the grid 0/2 times (never changed anything)" in last
    assert "WARNING: your last 2 actions changed nothing" in last
    assert "nothing changed" in last
