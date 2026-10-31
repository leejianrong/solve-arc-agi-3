"""V3 wiring: a verified world model plans, the agent follows the plan, a mismatch aborts it and tells the model."""

from __future__ import annotations

import json

from solve_arc_agi_3.agent import ActionDecision, BaselineAgentCore, Observation
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)

WIDTH = 12
GOAL_X = 6

# Toy game: one pixel on row 0. ACTION3 moves it right by 1, except from x=3 where it jumps by 2.
# Reaching GOAL_X completes the level.


class ToyGame:
    def __init__(self) -> None:
        self.x = 0
        self.levels = 0
        self.played: list[str] = []

    def obs(self) -> Observation:
        rows = [[0] * WIDTH for _ in range(4)]
        rows[0][self.x] = 5
        return Observation(
            game_id="toy",
            grid=tuple(map(tuple, rows)),
            state="NOT_FINISHED",
            levels_completed=self.levels,
            win_levels=2,
            available_actions=(1, 2, 3),
        )

    def apply(self, decision: ActionDecision) -> None:
        self.played.append(decision.name)
        if decision.name == "ACTION3":
            self.x = min(self.x + (2 if self.x == 3 else 1), WIDTH - 1)
            if self.x == GOAL_X:
                self.levels += 1


TEMPLATE = """def predict(frame, move):
    if move['name'] != 'ACTION3':
        return list(frame)
    x = frame[0].index('5')
    x = min(x + STEP, 11)
    return ['0' * x + '5' + '0' * (11 - x)] + list(frame[1:])

def is_goal(frame):
    return frame[0][6] == '5'
"""
V1 = TEMPLATE.replace("STEP", "1")
V2 = TEMPLATE.replace("STEP", "(2 if x == 3 else 1)")


class Scripted:
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


def submit(source: str, move: str) -> str:
    return f'submit_world_model("""{source}""")\naction({move})'


def drive(core: BaselineAgentCore, game: ToyGame, steps: int) -> list[str]:
    sources = []
    for _ in range(steps):
        obs = game.obs()
        decision = core.queued_decision(obs)
        if decision is None:
            decision = core.choose(obs).decision
        sources.append(str(core.last_turn["src"]))
        game.apply(decision)
    return sources


def test_agent_follows_the_plan_aborts_on_mismatch_and_replans_after_a_fix() -> None:
    client = Scripted("action(3)", submit(V1, "3"), submit(V2, "2"))
    core = BaselineAgentCore(client, tools=True, world_model=True)
    game = ToyGame()
    try:
        sources = drive(core, game, 6)
        assert sources == ["model", "model", "planned", "planned", "model", "planned"]
        assert game.played == ["ACTION3"] * 4 + ["ACTION2", "ACTION3"]
        # The model was told what went wrong, once, in the prompt of the call that followed.
        assert "PLAN ABORTED" in client.texts[2]
        assert "world model r1 predicted" in client.texts[2]
        assert "happened at step 3" in client.texts[2]
        assert core.last_mismatch["step"] == 3 and core.last_mismatch["revision"] == 1
        # It is recorded as counter evidence against the model.
        counters = [
            e
            for h in core._model_ledger.hypotheses
            for e in h.evidence
            if e.polarity == "counter"
        ]
        assert len(counters) == 1
        # The corrected model (r2) planned the last move, which cleared the level.
        assert game.levels == 1
        assert core.last_turn["plan"] == [{"name": "ACTION3"}]
    finally:
        core.close()


def test_no_planning_without_the_switch_or_a_goal() -> None:
    no_goal = V1.split("def is_goal")[0]
    client = Scripted("action(3)", submit(no_goal, "3"), "action(3)", "action(3)")
    core = BaselineAgentCore(client, tools=True, world_model=True)
    game = ToyGame()
    try:
        assert drive(core, game, 4) == ["model"] * 4
    finally:
        core.close()
    client = Scripted("action(3)", submit(V1, "3"), "action(3)")
    core = BaselineAgentCore(client, tools=True)
    assert drive(core, ToyGame(), 3) == ["model"] * 3


def test_a_host_side_goal_predicate_plans_without_is_goal() -> None:
    no_goal = V1.split("def is_goal")[0]
    client = Scripted("action(3)", submit(no_goal, "3"))
    core = BaselineAgentCore(client, tools=True, world_model=True)
    core.set_goal_predicate(lambda g: g[0][3] == 5)
    game = ToyGame()
    try:
        assert drive(core, game, 3) == ["model", "model", "planned"]
        assert game.x == 3
    finally:
        core.close()
