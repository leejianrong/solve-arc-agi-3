from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from solve_arc_agi_3.agent import ActionDecision
from solve_arc_agi_3.inference import (
    FunctionCall,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
)
from solve_arc_agi_3.kaggle_agent import RunLimits, fallback_decision, make_agent_class


@dataclass
class Frame:
    frame: list[list[list[int]]] = field(default_factory=lambda: [[[0, 1], [1, 0]]])
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    win_levels: int = 3
    available_actions: tuple[int, ...] = (1, 2, 6)
    game_id: str = "g"


class Base:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.game_id = "g"


def mapper(decision: ActionDecision) -> object:
    return decision


class OkClient:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay, self.in_flight, self.peak = delay, 0, 0
        self.lock = threading.Lock()

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(self.delay)
        with self.lock:
            self.in_flight -= 1
        call = ToolCall(
            id="c",
            function=FunctionCall(
                name="python", arguments=json.dumps({"code": "action(['ACTION2'])"})
            ),
        )
        return InferenceResponse(tool_calls=(call,), elapsed_seconds=0.0)


class BrokenClient:
    def complete(self, request: InferenceRequest) -> InferenceResponse:
        raise RuntimeError("server down")


def limits(**kw: object) -> RunLimits:
    return RunLimits(deadline_epoch=time.time() + 3600, **kw)  # type: ignore[arg-type]


def test_model_move_is_used_and_game_over_forces_reset_without_a_model_call() -> None:
    client = OkClient()
    agent = make_agent_class(
        Base, limits=limits(), client=client, action_mapper=mapper
    )()
    assert agent.choose_action([Frame()], Frame()).name == "ACTION2"
    reset = agent.choose_action([Frame()], Frame(state="GAME_OVER"))
    assert reset.name == "RESET" and reset.source == "forced"


def test_model_failure_falls_back_instead_of_raising() -> None:
    agent = make_agent_class(
        Base, limits=limits(), client=BrokenClient(), action_mapper=mapper
    )()
    first = agent.choose_action([Frame()], Frame())
    second = agent.choose_action([Frame()], Frame())
    assert first.source == second.source == "fallback"
    assert first.name != second.name  # cycles through what the game offers


def test_done_on_win_or_deadline() -> None:
    cls = make_agent_class(
        Base, limits=limits(), client=OkClient(), action_mapper=mapper
    )
    assert cls().is_done([], Frame(state="WIN")) is True
    assert cls().is_done([], Frame()) is False
    past = RunLimits(deadline_epoch=time.time() - 1)
    expired = make_agent_class(
        Base, limits=past, client=OkClient(), action_mapper=mapper
    )
    assert expired().is_done([], Frame()) is True


def test_concurrent_model_calls_are_capped_across_all_games() -> None:
    client = OkClient(delay=0.05)
    cls = make_agent_class(
        Base, limits=limits(max_concurrent_calls=3), client=client, action_mapper=mapper
    )
    agents = [cls() for _ in range(20)]
    threads = [
        threading.Thread(target=agent.choose_action, args=([Frame()], Frame()))
        for agent in agents
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert 1 < client.peak <= 3


def test_fallback_clicks_the_centre_for_action6() -> None:
    d = fallback_decision((6,), 0)
    assert d.name == "ACTION6" and d.data == {"x": 32, "y": 32}


def test_limits_from_env() -> None:
    lim = RunLimits.from_env(
        {"ARC3_DEADLINE_EPOCH": "100", "ARC3_MAX_CONCURRENT_CALLS": "8"}
    )
    assert lim.max_concurrent_calls == 8 and lim.expired(101) and not lim.expired(99)


def test_first_action_needs_no_model_and_model_calls_wait_for_the_ready_flag(
    tmp_path: Path,
) -> None:
    flag = tmp_path / "ready"
    lim = RunLimits(deadline_epoch=time.time() + 3600, ready_flag=str(flag))
    client = OkClient()
    agent = make_agent_class(Base, limits=lim, client=client, action_mapper=mapper)()
    first = agent.choose_action([Frame()], Frame(state="NOT_PLAYED"))
    assert first.name == "RESET" and client.peak == 0  # no model, no waiting

    results: list[object] = []
    worker = threading.Thread(
        target=lambda: results.append(agent.choose_action([Frame()], Frame()))
    )
    worker.start()
    time.sleep(0.2)
    assert results == []  # still waiting for the model server
    flag.write_text("ok")
    worker.join(5)
    assert results and results[0].name == "ACTION2"  # type: ignore[attr-defined]


def test_wait_gives_up_at_the_deadline(tmp_path: Path) -> None:
    lim = RunLimits(
        deadline_epoch=time.time() + 0.2, ready_flag=str(tmp_path / "never")
    )
    assert lim.wait_ready(poll_seconds=0.05) is False


def test_failed_boot_degrades_to_fallback_moves_instead_of_hanging(
    tmp_path: Path,
) -> None:
    flag = tmp_path / "ready"
    (tmp_path / "ready.failed").write_text("vllm exited")
    lim = RunLimits(deadline_epoch=time.time() + 3600, ready_flag=str(flag))
    agent = make_agent_class(
        Base, limits=lim, client=OkClient(), action_mapper=mapper
    )()
    move = agent.choose_action([Frame()], Frame())
    assert move.source == "fallback"
