"""Model-callable ledger tools and the world-model tool (V3): parsed, never executed, off unless asked for."""

from __future__ import annotations

import json
import os
import time

import pytest

from solve_arc_agi_3.agent import (
    PYTHON_TOOL,
    SYSTEM_PROMPT,
    BaselineAgentCore,
    Observation,
)
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)


class Scripted:
    def __init__(self, *codes: str) -> None:
        self.codes, self.texts, self.requests = list(codes), [], []

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        content = request.messages[-1].content
        self.requests.append(request)
        self.texts.append("".join(getattr(p, "text", "") for p in content))  # type: ignore[union-attr,arg-type]
        call = ToolCall(
            id="c",
            function=FunctionCall(
                name="python", arguments=json.dumps({"code": self.codes.pop(0)})
            ),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


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
        win_levels=3,
        available_actions=(1, 2, 3),
    )


def play(core: BaselineAgentCore, frames: list[Observation]) -> list[str]:
    out = []
    for obs in frames:
        decision = core.choose(obs).decision
        out.append(decision.name)
    return out


def test_everything_is_off_by_default() -> None:
    client = Scripted(
        'hypothesis("ACTION3 moves right", "mechanic")\naction(3)', "action(3)"
    )
    core = BaselineAgentCore(client, use_ledger=True)
    assert play(core, [board(2), board(7, 1)]) == ["ACTION3", "ACTION3"]
    assert "ACTION3 moves right" not in client.texts[1]
    request = client.requests[0]
    assert request.tools == (PYTHON_TOOL,)
    assert request.messages[0].content == SYSTEM_PROMPT
    assert (
        "hypothesis" not in SYSTEM_PROMPT
        and "hypothesis" not in PYTHON_TOOL.description
    )


def test_the_tool_text_changes_only_with_the_switch() -> None:
    client = Scripted("action(3)")
    play(BaselineAgentCore(client, tools=True), [board(2)])
    request = client.requests[0]
    assert "hypothesis(" in request.tools[0].description
    assert "evidence(" in str(request.messages[0].content)
    assert "submit_world_model" not in request.tools[0].description
    assert "submit_world_model" not in str(request.messages[0].content)


def test_world_model_requires_tools() -> None:
    with pytest.raises(ValueError):
        BaselineAgentCore(Scripted(), world_model=True)


def test_claims_and_evidence_come_back_with_ids() -> None:
    client = Scripted(
        'hypothesis("ACTION3 moves the block right", "mechanic")\n'
        'hypothesis("touch the green cell", "goal", "level clears")\n'
        "action(3)",
        'evidence("h1", "support", "moved by 5")\nevidence("h2", "counter", "no")\naction(3)',
        "action(3)",
    )
    core = BaselineAgentCore(client, tools=True)
    play(core, [board(2), board(7, 1), board(12, 2)])
    assert "Hypotheses so far" not in client.texts[0]
    assert "h1 open mechanic +0/-0: ACTION3 moves the block right" in client.texts[1]
    assert "h2 open goal +0/-0: touch the green cell -> level clears" in client.texts[1]
    last = client.texts[2]
    assert "h1 open mechanic +1/-0: ACTION3 moves the block right" in last
    assert "h2 refuted goal +0/-1" in last and "counter at step" in last


def test_model_claims_keep_their_ids_next_to_computed_ones() -> None:
    client = Scripted(
        'hypothesis("the block is the player", "mechanic")\naction(3)',
        "action(3)",
        "action(3)",
        "action(3)",
    )
    core = BaselineAgentCore(client, tools=True, use_ledger=True)
    play(core, [board(2 + 5 * i, i) for i in range(4)])
    last = client.texts[-1]
    assert "h1 open mechanic +0/-0: the block is the player" in last
    assert "ACTION3 moves the object (colour 5, size 4) by (5, 0)" in last
    assert "h1 open mechanic +0/-0" in last and last.count("h1 ") == 1


def test_malformed_calls_are_ignored_and_the_move_still_plays() -> None:
    code = """hypothesis()
hypothesis(5, 'mechanic')
hypothesis('ok but bad kind', 'wish')
hypothesis('fine', 'mechanic')
evidence('h9', 'support')
evidence('h1', 'maybe')
evidence(1, 2, 3)
hypothesis(some_variable, 'goal')
action(3)"""
    client = Scripted(code, "action(3)")
    core = BaselineAgentCore(client, tools=True)
    assert play(core, [board(2), board(7, 1)]) == ["ACTION3", "ACTION3"]
    ids = [h.id for h in core._model_ledger.hypotheses]
    assert (
        ids == ["h1"] or ids == []
    )  # cap of 3 calls: only the first three attempts count
    broken = Scripted(
        "hypothesis('x', 'mechanic'\naction(3)"
    )  # a syntax error plays nothing
    core = BaselineAgentCore(broken, tools=True)
    with pytest.raises(ValueError):
        core.choose(board(2))
    assert core._model_ledger.hypotheses == []


def test_at_most_three_calls_per_reply_and_statements_are_cut() -> None:
    calls = [f"hypothesis('claim {i} ' + 'x', 'mechanic')" for i in range(5)]
    literal = [f"hypothesis('claim {i}', 'mechanic')" for i in range(5)]
    client = Scripted("\n".join(calls + literal) + "\naction(3)", "action(3)")
    core = BaselineAgentCore(client, tools=True)
    play(core, [board(2), board(7, 1)])
    # the first three attempts (non-literal arguments) are malformed and use up the cap
    assert core._model_ledger.hypotheses == []
    client = Scripted("\n".join(literal) + "\naction(3)")
    core = BaselineAgentCore(client, tools=True)
    play(core, [board(2)])
    assert [h.statement for h in core._model_ledger.hypotheses] == [
        "claim 0",
        "claim 1",
        "claim 2",
    ]
    long = "a" * 300
    client = Scripted(f"hypothesis('{long}', 'goal', '{long}')\naction(3)")
    core = BaselineAgentCore(client, tools=True)
    play(core, [board(2)])
    claim = core._model_ledger.hypotheses[0]
    assert len(claim.statement) == 200 and claim.prediction is not None
    assert len(claim.prediction) == 200


def test_the_cap_counts_across_python_calls_in_one_reply() -> None:
    def call(code: str) -> ToolCall:
        return ToolCall(
            id="c",
            function=FunctionCall(name="python", arguments=json.dumps({"code": code})),
        )

    core = BaselineAgentCore(Scripted(), tools=True)
    core._apply_tool_calls(
        InferenceResponse(
            tool_calls=(
                call("hypothesis('a', 'goal')\nhypothesis('b', 'goal')"),
                call("hypothesis('c', 'goal')\nhypothesis('d', 'goal')"),
            ),
            elapsed_seconds=0.0,
        )
    )
    assert [h.statement for h in core._model_ledger.hypotheses] == ["a", "b", "c"]


# World model -----------------------------------------------------------------------------------------

GOOD = """def predict(frame, move):
    if move['name'] != 'ACTION3':
        return list(frame)
    rows = [list(r) for r in frame]
    out = [['0'] * len(r) for r in rows]
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c != '0':
                out[y][x + 5] = c
    return [''.join(r) for r in out]
"""
WRONG = GOOD.replace("x + 5", "x + 4")


def wboard(x: int) -> Observation:
    rows = [[0] * 24 for _ in range(14)]
    for dx in range(2):
        for dy in range(2):
            rows[5 + dy][x + dx] = 5
    return Observation(
        game_id="g",
        grid=tuple(map(tuple, rows)),
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=3,
        available_actions=(1, 2, 3),
    )


def submit(source: str) -> str:
    return f'submit_world_model("""{source}""")\naction(3)'


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_a_correct_then_a_wrong_world_model() -> None:
    client = Scripted(
        "action(3)",
        submit(GOOD),
        submit(WRONG),
        submit(GOOD + "# same behaviour, new text\n"),
        "action(3)",
    )
    core = BaselineAgentCore(client, tools=True, world_model=True)
    frames = [wboard(2), wboard(7), wboard(12), wboard(17), wboard(22 - 4)]
    play(core, frames[:2])
    assert "World model" not in client.texts[1]
    play(core, frames[2:3])
    assert "World model r1 replays all 1 steps exactly." in client.texts[2]
    first = core._wm_result
    assert first is not None and first.accepted and first.model is not None
    pid = first.model.pid
    assert alive(pid)
    assert "World model rejected" in core._wm_message and "mismatch" in core._wm_message
    assert len(core._wm_message) <= 600
    assert core._wm_result is first and alive(pid)  # the wrong one never replaced it
    play(core, frames[3:4])
    assert "World model rejected" in client.texts[3]  # shown in the very next prompt
    assert core._wm_result is not first  # the later correct revision replaced it
    play(core, frames[4:5])
    deadline = time.monotonic() + 3
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)
    second = core._wm_result
    deadline = time.monotonic() + 3
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)
    assert second is not None and second.model is not None
    pid2 = second.model.pid
    assert alive(pid2)
    core.close()
    core.close()
    deadline = time.monotonic() + 3
    while alive(pid2) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid2)


def test_world_model_message_is_shown_once_and_a_bad_source_never_fails_the_move() -> (
    None
):
    client = Scripted(
        "action(3)",
        submit("def predict(:\n"),
        "action(3)",
        "action(3)",
    )
    core = BaselineAgentCore(client, tools=True, world_model=True)
    assert play(core, [wboard(2), wboard(7), wboard(12), wboard(17)]) == ["ACTION3"] * 4
    assert "World model rejected" in client.texts[2]
    assert "World model" not in client.texts[3]


def test_submit_world_model_is_ignored_without_its_switch() -> None:
    client = Scripted("action(3)", submit(GOOD), "action(3)")
    core = BaselineAgentCore(client, tools=True)
    play(core, [wboard(2), wboard(7), wboard(12)])
    assert "World model" not in client.texts[2]
    assert core._wm_records == []
