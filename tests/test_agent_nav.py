"""ARC3_NAV header (state graph, reachability, loop warning) and the run(move, n) macro."""

from __future__ import annotations

import json

from solve_arc_agi_3.agent import (
    SYSTEM_PROMPT,
    BaselineAgentCore,
    Observation,
    parse_action_decisions,
)
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)


def _resp(code: str) -> InferenceResponse:
    call = ToolCall(
        id="c",
        function=FunctionCall(name="python", arguments=json.dumps({"code": code})),
    )
    return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


class Scripted:
    def __init__(self, *codes: str) -> None:
        self.codes, self.texts = list(codes), []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.texts.append(
            "".join(getattr(p, "text", "") for p in request.messages[-1].content)  # type: ignore[union-attr,arg-type]
        )
        return _resp(self.codes.pop(0))


def board(x: int, goal: bool = True) -> Observation:
    rows = [[0] * 24 for _ in range(14)]
    for dx in range(2):
        for dy in range(2):
            rows[5 + dy][x + dx] = 5
    if goal:
        rows[5][20] = 3
        rows[6][20] = 3
    rows[13][1] = 1
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=1,
        available_actions=(1, 2, 3, 4),
    )


def _play(core: BaselineAgentCore, xs: list[int]) -> None:
    for x in xs:
        core.choose(board(x))


def test_nav_header_has_graph_reachability_and_loop_warning() -> None:
    client = Scripted(*["action([4])"] * 5, *["action([3])"] * 5)
    core = BaselineAgentCore(client, nav=True)
    _play(core, [2, 4, 6, 8])
    text = client.texts[-1]
    assert "This board seen 1 time." in text
    assert "Reachable by repeating moves:" in text
    assert "WARNING: a board repeated" not in text
    _play(core, [6, 8])  # walks back and forth: boards 6 and 8 repeat
    assert "WARNING: a board repeated within your last 8 moves" in client.texts[-1]
    assert "seen 2 times" in client.texts[-1] or "seen 3 times" in client.texts[-1]


def test_nav_off_header_is_unchanged() -> None:
    client = Scripted(*["action([4])"] * 4)
    core = BaselineAgentCore(client)
    _play(core, [2, 4, 6])
    for marker in ("This board seen", "Reachable by", "WARNING: a board"):
        assert marker not in client.texts[-1]
    assert client.texts[-1].startswith("STATUS: Controlled object")


def test_run_macro_expands_inside_a_list() -> None:
    got = parse_action_decisions(
        _resp("action([run(1, 8), 2])"), available_actions=(1, 2, 3, 4)
    )
    assert [d.name for d in got] == ["ACTION1"] * 8 + ["ACTION2"]
    assert len({d.source for d in got[:8]}) == 1
    capped = parse_action_decisions(
        _resp("action([run('ACTION3', 99)])"), available_actions=(1, 2, 3, 4)
    )
    assert len(capped) == 12
    plain = parse_action_decisions(
        _resp("action([1, 1, 1, 1, 1, 1, 1])"), available_actions=(1, 2, 3, 4)
    )
    assert len(plain) == 5


def test_run_macro_aborts_on_the_first_blocked_move() -> None:
    client = Scripted("action([run(4, 8), 2])")
    core = BaselineAgentCore(client)
    first = core.choose(board(2)).decision
    assert first.name == "ACTION4"
    # after move 1 and 2 the board changed; after move 3 it did not, so the other run moves are dropped
    assert core.queued_decision(board(4)).name == "ACTION4"  # type: ignore[union-attr]
    assert core.queued_decision(board(6)).name == "ACTION4"  # type: ignore[union-attr]
    nxt = core.queued_decision(board(6))
    assert nxt is not None and nxt.name == "ACTION2"
    assert core.queued_decision(board(6)) is None


def test_prompts_mention_run_and_stay_stable() -> None:
    assert "run(1, 8)" in SYSTEM_PROMPT
    client = Scripted("action([4])", "action([4])")
    core = BaselineAgentCore(client)
    core.choose(board(2))
    core.choose(board(4))
    assert "Controlled object (colour 5, size 4) is at x4-5 y5-6" in client.texts[1]
