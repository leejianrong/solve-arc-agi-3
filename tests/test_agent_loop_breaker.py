"""Loop-breaker: a move that did nothing on the identical board is swapped for something untried."""

from __future__ import annotations

import json

from solve_arc_agi_3.agent import BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)
from solve_arc_agi_3.observation import untried_clicks


class Scripted:
    def __init__(self, *codes: str) -> None:
        self.codes = list(codes)
        self.texts: list[str] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.texts.append(
            "".join(getattr(p, "text", "") for p in request.messages[-1].content)  # type: ignore[union-attr,arg-type]
        )
        code = self.codes.pop(0)
        call = ToolCall(
            id="c",
            function=FunctionCall(name="python", arguments=json.dumps({"code": code})),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


def board(actions: tuple[int, ...] = (1, 2, 6)) -> Observation:
    rows = [[0] * 16 for _ in range(16)]
    for x0, color in ((2, 3), (8, 4), (12, 5)):
        for dx in range(2):
            for dy in range(2):
                rows[3 + dy][x0 + dx] = color
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=1,
        available_actions=actions,
    )


CLICK = 'action({"action": 6, "x": 2, "y": 3})'


def test_untried_clicks_skip_clicked_objects_and_clicked_colours_go_last() -> None:
    got = untried_clicks(board().grid, [(2, 3)])
    assert [r.color for r in got] == [4, 5]
    assert untried_clicks(board().grid, [])[0].color == 3


def test_a_click_that_did_nothing_twice_on_the_same_board_is_swapped() -> None:
    core = BaselineAgentCore(Scripted(CLICK, CLICK, CLICK))
    assert core.choose(board()).decision.data == {"x": 2, "y": 3}
    core.choose(board())  # second no-op on the identical board
    turn = core.choose(board())
    assert turn.decision.name == "ACTION6"
    assert turn.decision.data != {"x": 2, "y": 3}  # an untried object instead
    assert "override" in core.last_turn
    assert core.last_turn["move"] == {"name": "ACTION6", **turn.decision.data}


def test_the_prompt_lists_dead_moves_and_unclicked_objects() -> None:
    client = Scripted(CLICK, CLICK, CLICK)
    core = BaselineAgentCore(client)
    for _ in range(3):
        core.choose(board())
    text = client.texts[2]
    assert "do not repeat them: ACTION6 at x=2,y=3" in text
    assert "Objects not clicked yet" in text
    assert "colour 4 at (8, 3)" in text


def test_a_plain_action_is_swapped_for_an_untried_one() -> None:
    core = BaselineAgentCore(Scripted("action([1])", "action([1])", "action([1])"))
    for _ in range(2):
        core.choose(board((1, 2)))
    turn = core.choose(board((1, 2)))
    assert turn.decision.name == "ACTION2"


def test_a_dead_move_is_not_swapped_when_nothing_else_exists() -> None:
    core = BaselineAgentCore(Scripted("action([1])", "action([1])", "action([1])"))
    for _ in range(3):
        turn = core.choose(board((1,)))
    assert turn.decision.name == "ACTION1"
    assert "override" not in core.last_turn
