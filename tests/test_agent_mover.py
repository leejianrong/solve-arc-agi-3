"""The controlled-object header must not flip between twins or to a drifting object."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solve_arc_agi_3.agent import BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)


class Client:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.texts.append(
            "".join(getattr(p, "text", "") for p in request.messages[-1].content)  # type: ignore[union-attr,arg-type]
        )
        call = ToolCall(
            id="c",
            function=FunctionCall(
                name="python", arguments=json.dumps({"code": "action([4])"})
            ),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


def board(x: int, twin_x: int, drift_y: int) -> Observation:
    rows = [[0] * 40 for _ in range(30)]

    def box(x0: int, y0: int, w: int, h: int, colour: int) -> None:
        for dy in range(h):
            for dx in range(w):
                rows[y0 + dy][x0 + dx] = colour

    box(x, 5, 3, 3, 5)  # controlled piece
    box(twin_x, 20, 3, 3, 5)  # identical twin, mirrored
    box(30, drift_y, 2, 2, 7)  # a second object drifting on its own
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=1,
        available_actions=(1, 2, 3, 4),
    )


def test_mover_does_not_flip_to_twin_or_drifter() -> None:
    client = Client()
    core = BaselineAgentCore(client)
    for i in range(6):
        core.choose(board(3 + 4 * i, 30 - 4 * i, 2 + i))
    for text in client.texts[1:]:
        assert "colour 5, size 9" in text
    assert core._mover is not None and core._mover.x0 == 3 + 4 * 5
    assert "identical twin at x" in client.texts[-1]
    assert core._mover.y0 == 5  # the twin sits at y20, never reported


def test_last_move_and_plan_results_agree() -> None:
    client = Client()
    core = BaselineAgentCore(client)
    seen: list[str] = []
    orig = core._plan_results
    core._plan_results = lambda: (seen.extend(core._since_call[-1:]), orig())[1]  # type: ignore[method-assign]
    for i in range(3):
        core.choose(board(3 + 4 * i, 30 - 4 * i, 2 + i))
    assert "MOVED the object" in core._move_status
    assert seen[-1].startswith(core._move_status)


TRACE = Path("build/dev-runs/v6/traces/dry-1793457012/ls20-9607627b.jsonl")


@pytest.mark.skipif(not TRACE.exists(), reason="real ls20 trace not available")
def test_real_ls20_sprite_regions_keep_one_type() -> None:
    core = BaselineAgentCore(Client())
    types = []
    for line in TRACE.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") != "step" or not row.get("motion"):
            continue
        from solve_arc_agi_3.observation import Region

        pairs = [
            (
                Region(m["color"], m["size"], *m["from"], *m["from"]),
                Region(m["color"], m["size"], *m["to"], *m["to"]),
            )
            for m in row["motion"]
        ]
        if core._pick_mover(pairs):
            assert core._mover is not None
            types.append((core._mover.color, core._mover.size))
    assert len(types) > 20
    assert len(set(types)) == 1
