"""Click guidance (ARC-155) and the LEVEL CLEARED note (ARC-160)."""

from __future__ import annotations

import json

from solve_arc_agi_3.agent import BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)

BUTTONS = {(4, 32): 3, (58, 32): 4}  # the only two clicks that do anything


class Scripted:
    def __init__(self, *points: tuple[int, int]) -> None:
        self.points = list(points)
        self.texts: list[str] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.texts.append(
            "".join(getattr(p, "text", "") for p in request.messages[-1].content)  # type: ignore[union-attr,arg-type]
        )
        x, y = self.points.pop(0)
        code = json.dumps({"code": f'action({{"action": 6, "x": {x}, "y": {y}}})'})
        call = ToolCall(id="c", function=FunctionCall(name="python", arguments=code))
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


def toy(presses: int = 0, levels: int = 0) -> Observation:
    """Two big side buttons, a few other blocks; the left strip row shows how often a button was pressed."""
    rows = [[0] * 64 for _ in range(64)]
    for x0, y0, c in ((2, 30, 3), (56, 30, 4), (20, 10, 5), (30, 50, 6), (40, 20, 7)):
        for dx in range(5):
            for dy in range(5):
                rows[y0 + dy][x0 + dx] = c
    for i in range(presses):
        rows[60][i] = 9
    return Observation(
        game_id="lp",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=levels,
        win_levels=3,
        available_actions=(6,),
    )


def play(points: list[tuple[int, int]], clear_at: int | None = None):
    client = Scripted(*points)
    core = BaselineAgentCore(client)
    presses = levels = 0
    turns = []
    for i, p in enumerate(points):
        turns.append(core.choose(toy(presses, levels)))
        if p in BUTTONS:
            presses += 1
        if clear_at == i:
            levels += 1
    core.settle(toy(presses, levels))
    return core, client, turns


def test_effective_clicks_lead_and_the_sweep_list_disappears() -> None:
    pts = [(10, 10), (4, 32), (58, 32), (4, 32), (4, 32)]
    _, client, _ = play(pts)
    text = client.texts[4]
    assert (
        "Clicks that changed the board: (4,32) x2, (58,32) x1 (repeat or vary these)."
        in text
    )
    assert "Objects not clicked yet" not in text
    assert text.index("Clicks that changed") < text.index("Game state")


def test_untried_list_still_shown_before_any_click_worked_capped_at_8() -> None:
    _, client, _ = play([(10, 10), (10, 10)])
    text = client.texts[1]
    assert "Clicks that changed" not in text
    listed = text.split("Objects not clicked yet")[1].split("\n")[0]
    assert listed.count("colour") <= 8


def test_effective_click_is_not_overridden_unless_dead_on_this_board() -> None:
    # (4,32) changed the board once; clicking it again on a board where it did nothing twice is allowed
    # to be swapped, but a live effective click is never replaced.
    core, _, turns = play([(4, 32), (4, 32), (4, 32)])
    assert all(t.decision.data == {"x": 4, "y": 32} for t in turns)
    assert "override" not in core.last_turn


def test_dead_effective_click_is_swapped_for_another_effective_click() -> None:
    client = Scripted((4, 32), (58, 32), (4, 32), (4, 32), (4, 32))
    core = BaselineAgentCore(client)
    core.choose(toy(0))
    core.choose(toy(1))  # both buttons changed the board
    same = toy(2)
    core.choose(same)  # (4,32) on `same`: no change
    core.choose(same)  # no change again -> dead on this exact board
    turn = core.choose(same)
    assert turn.decision.data == {"x": 58, "y": 32}
    assert "override" in core.last_turn
    assert "Objects not clicked yet" not in client.texts[4]


def test_untried_list_returns_when_every_effective_click_is_dead_here() -> None:
    client = Scripted((4, 32), (4, 32), (4, 32), (10, 10))
    core = BaselineAgentCore(client)
    core.choose(toy(0))
    same = toy(1)
    core.choose(same)
    core.choose(same)
    core.choose(same)
    text = client.texts[3]
    assert "Clicks that changed the board: (4,32) x1" in text
    assert "Objects not clicked yet" in text


def test_level_cleared_note_appears_once_and_is_traced() -> None:
    pts = [(4, 32), (58, 32), (4, 32), (58, 32)]
    client = Scripted(*pts)
    core = BaselineAgentCore(client)
    core.choose(toy(0, 0))
    core.choose(toy(1, 0))
    core.choose(toy(2, 0))
    assert "cleared" not in core.last_turn
    core.choose(toy(3, 1))  # levels_completed rose after the third move
    text = client.texts[3]
    assert "LEVEL CLEARED by: " in text
    assert "ACTION6 at x=4,y=32" in text.split("LEVEL CLEARED by: ")[1]
    assert "board change that triggered it:" in text
    assert core.last_turn["cleared"].startswith("LEVEL CLEARED by:")  # type: ignore[union-attr]
    core2 = BaselineAgentCore(Scripted((4, 32), (4, 32)))
    core2.choose(toy(0, 0))
    core2.choose(toy(1, 1))
    assert "cleared" in core2.last_turn


def _hint_text(mode: str) -> str:
    client = Scripted((4, 32), (10, 10), (10, 10))
    core = BaselineAgentCore(client, click_hints=mode)
    for presses in (0, 1, 1):
        core.choose(toy(presses))
    return client.texts[2]


def test_click_hint_modes() -> None:
    both = ("Clicks that changed", "Objects not clicked yet")
    for mode, want in (
        ("auto", (True, False)),
        ("wins", (True, False)),
        ("sweep", (False, True)),
        ("none", (False, False)),
    ):
        text = _hint_text(mode)
        assert tuple(s in text for s in both) == want, mode


def test_unknown_click_hint_mode_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        BaselineAgentCore(Scripted(), click_hints="bogus")
