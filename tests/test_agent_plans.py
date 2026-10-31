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


def _text(request: InferenceRequest) -> str:
    return "".join(getattr(p, "text", "") for p in request.messages[-1].content)  # type: ignore[union-attr,arg-type]


class Recording(Scripted):
    def __init__(self, *codes: str) -> None:
        super().__init__(*codes)
        self.texts: list[str] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.texts.append(_text(request))
        return super().complete(request)


def _frame(tick: int, player_x: int) -> Observation:
    """A 10x4 grid: a counter cell at (9,3) that flips every call, and a player that moves."""
    rows = [[0] * 10 for _ in range(4)]
    rows[3][9] = 1 + tick % 2
    rows[1][player_x] = 5
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=2,
        available_actions=(1, 2, 3, 4),
    )


def test_a_fixed_cell_that_changes_every_action_is_flagged_as_a_counter_but_a_moving_player_is_not() -> (
    None
):
    client = Recording(*["action(4)"] * 6)
    core = BaselineAgentCore(client)
    for tick in range(6):
        core.choose(_frame(tick, player_x=tick))
    last = client.texts[-1]
    assert "probably a step counter or timer" in last
    assert "x9-9 y3-3" in last  # the counter cell
    assert (
        "y1-1" not in last.split("probably a step counter")[1].split("\n")[0]
    )  # the moving player is not flagged


def test_no_counter_warning_before_three_repeats() -> None:
    client = Recording("action(4)", "action(4)")
    core = BaselineAgentCore(client)
    core.choose(_frame(0, 0))
    core.choose(_frame(1, 1))
    assert "step counter" not in client.texts[-1]


def test_rejection_says_which_action_was_not_available() -> None:
    client = Recording("action(5)", "action(2)")
    core = BaselineAgentCore(client)
    o = _frame(0, 0)
    try:
        core.choose(o)
    except ValueError:
        pass
    core.request_nudge()
    core.choose(o)
    assert (
        "used ACTION5, which is not available now (available: ACTION1, ACTION2, ACTION3, ACTION4)"
        in client.texts[-1]
    )


def test_a_click_without_coordinates_is_explained() -> None:
    client = Recording("action(6)", "action(2)")
    core = BaselineAgentCore(client)
    o = Observation(
        game_id="g",
        grid=_frame(0, 0).grid,
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=2,
        available_actions=(2, 6),
    )
    try:
        core.choose(o)
    except ValueError:
        pass
    core.request_nudge()
    core.choose(o)
    assert (
        "click had no coordinates" in client.texts[-1] and '"x": 12' in client.texts[-1]
    )


def test_fifty_small_objects_are_all_listed() -> None:
    from solve_arc_agi_3.observation import describe_objects

    grid = tuple(
        tuple(1 if x % 2 == 0 and y % 2 == 0 else 0 for x in range(20))
        for y in range(10)
    )  # 50 singletons
    assert "not listed" not in describe_objects(grid)
