from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from solve_arc_agi_3.agent import ActionDecision
from solve_arc_agi_3.inference import (
    ChatMessage,
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


def test_fallback_scans_the_grid_for_click_only_games() -> None:
    points = {tuple(fallback_decision((6,), i).data.values()) for i in range(40)}
    assert len(points) == 40  # a different cell each time
    assert all(0 <= v <= 63 for p in points for v in p)


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


def test_logging_client_reports_sample_replies_and_stats(caplog: object) -> None:
    import logging

    from solve_arc_agi_3.kaggle_agent import LoggingClient

    client = LoggingClient(OkClient(), samples=1, every=2)
    request = InferenceRequest(messages=(ChatMessage(role="user", content="x"),))
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        client.complete(request)
        client.complete(request)
    text = caplog.text  # type: ignore[attr-defined]
    assert "model sample #1" in text and "model stats: 2 calls" in text


def test_agent_class_from_env_applies_the_run_tunables(monkeypatch: object) -> None:
    from solve_arc_agi_3.kaggle_agent import LoggingClient, agent_class_from_env

    root = Path(__file__).parents[1]
    env = {
        "ARC3_MANIFEST": str(root / "configs/baselines/duck-control-v1.json"),
        "ARC3_DEADLINE_EPOCH": str(time.time() + 60),
        "ARC3_THINKING": "0",
        "ARC3_MAX_OUTPUT_TOKENS": "777",
        "ARC3_REQUEST_TIMEOUT": "55",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)  # type: ignore[attr-defined]
    cls = agent_class_from_env(Base)
    client = cls().__dict__["_adapter"]._core._client
    assert isinstance(client, LoggingClient)
    config = client._inner.config
    assert (
        config.max_output_tokens,
        config.request_timeout_seconds,
        config.sampling.thinking,
    ) == (777, 55.0, False)


def test_a_prose_reply_gets_one_plain_retry_before_falling_back() -> None:
    replies = [
        InferenceResponse(
            content="Let me analyze the grid first...", elapsed_seconds=0.0
        ),
        InferenceResponse(
            tool_calls=(
                ToolCall(
                    id="c",
                    function=FunctionCall(
                        name="python", arguments=json.dumps({"code": "action(2)"})
                    ),
                ),
            ),
            elapsed_seconds=0.0,
        ),
    ]
    seen: list[str] = []

    class Client:
        def complete(self, request: InferenceRequest) -> InferenceResponse:
            content = request.messages[-1].content
            seen.append("".join(getattr(p, "text", "") for p in content))  # type: ignore[arg-type,union-attr]
            return replies.pop(0)

    agent = make_agent_class(
        Base, limits=limits(), client=Client(), action_mapper=mapper
    )()
    move = agent.choose_action([Frame()], Frame())
    assert move.name == "ACTION2" and move.source == "python_tool"  # type: ignore[attr-defined]
    assert (
        "previous reply contained no action" in seen[1]
        and "previous reply" not in seen[0]
    )


def test_switches_come_from_the_environment_and_default_to_off() -> None:
    from solve_arc_agi_3.kaggle_agent import agent_switches

    assert agent_switches({}) == {
        "prompt_chars": 0,
        "events": False,
        "tools": False,
        "world_model": False,
    }
    assert agent_switches(
        {
            "ARC3_PROMPT_CHARS": "4000",
            "ARC3_EVENTS": "1",
            "ARC3_TOOLS": "1",
            "ARC3_WORLD_MODEL": "1",
        }
    ) == {"prompt_chars": 4000, "events": True, "tools": True, "world_model": True}
    assert (
        agent_switches({"ARC3_PROMPT_CHARS": "", "ARC3_EVENTS": "0"})["events"] is False
    )


def test_world_model_without_tools_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC3_WORLD_MODEL", "1")
    monkeypatch.delenv("ARC3_TOOLS", raising=False)
    with pytest.raises(ValueError):
        make_agent_class(Base, limits=limits(), client=OkClient(), action_mapper=mapper)


def test_cleanup_closes_the_core_and_calls_the_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class WithCleanup(Base):
        def cleanup(self, *args: object, **kwargs: object) -> None:
            calls.append("base")

    for name in ("ARC3_EVENTS", "ARC3_TOOLS", "ARC3_WORLD_MODEL", "ARC3_PROMPT_CHARS"):
        monkeypatch.delenv(name, raising=False)
    agent = make_agent_class(
        WithCleanup, limits=limits(), client=OkClient(), action_mapper=mapper
    )()
    closed: list[int] = []
    agent._adapter.core.close = lambda: closed.append(1)  # type: ignore[method-assign]
    agent.cleanup()
    assert calls == ["base"] and closed == [1]
