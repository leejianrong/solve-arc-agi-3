"""Deterministic no-network baseline smoke episode."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .agent import BaselineAgentCore, observation_from_frame
from .baseline_config import InferenceConfig
from .fake_openai import DeterministicFakeServer, assert_openai_request_shape
from .inference import InferenceClient, OpenAICompatibleClient
from .trace import TraceWriter


@dataclass
class SmokeFrame:
    game_id: str
    frame: object
    state: object
    levels_completed: int
    win_levels: int
    available_actions: list[int]


@dataclass
class SmokeEnvironment:
    actions: list[str] = field(default_factory=list)

    def reset(self) -> SmokeFrame:
        return SmokeFrame(
            game_id="deterministic-smoke",
            frame=[[[0, 1], [0, 0]]],
            state="NOT_FINISHED",
            levels_completed=0,
            win_levels=1,
            available_actions=[1],
        )

    def step(self, action_name: str) -> SmokeFrame:
        if action_name != "ACTION1":
            raise ValueError(f"smoke environment rejects {action_name}")
        self.actions.append(action_name)
        return SmokeFrame(
            game_id="deterministic-smoke",
            frame=[[[0, 0], [0, 1]]],
            state="WIN",
            levels_completed=1,
            win_levels=1,
            available_actions=[],
        )


class SmokeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    run_dir: Path
    trace_path: Path
    terminal_state: str
    actions: tuple[str, ...]
    model_requests: int
    network_requests: int


def _run_smoke_episode(
    *,
    config: InferenceConfig,
    runs_root: Path,
    client: InferenceClient,
    request_capture: list[dict[str, object]] | None,
    network_requests: int,
    run_id_factory: Callable[[], str] | None = None,
) -> SmokeResult:
    make_run_id = run_id_factory or (lambda: str(uuid.uuid4()))
    run_id = make_run_id()
    trace = TraceWriter(runs_root, run_id)
    agent = BaselineAgentCore(client)
    environment = SmokeEnvironment()

    trace.append("configuration", config.model_dump(mode="json"))
    try:
        initial = environment.reset()
        observation = observation_from_frame(initial)
        trace.append("observation", observation.model_dump(mode="json"))
        # Trace the request/response before parsing an action from it: parsing
        # can reject an otherwise well-formed reply, and losing visibility
        # into what the model actually said makes that failure undebuggable.
        model_request, model_response = agent.request_completion(observation)
        request_payload: dict[str, Any]
        if request_capture is None:
            request_payload = {
                "model": config.model,
                "messages": [
                    message.model_dump(mode="json")
                    for message in model_request.messages
                ],
                "tools": [tool.model_dump(mode="json") for tool in model_request.tools],
            }
        else:
            request_payload = request_capture[-1]
            assert_openai_request_shape(request_payload)
        trace.append("model_request", request_payload)
        trace.append("model_response", model_response.model_dump(mode="json"))
        decision = agent.record_decision(observation, model_response)
        trace.append("action_decision", decision.model_dump(mode="json"))
        terminal = environment.step(decision.name)
        terminal_observation = observation_from_frame(terminal)
        trace.append(
            "environment_transition",
            {
                "action": decision.model_dump(mode="json"),
                "observation": terminal_observation.model_dump(mode="json"),
            },
        )
        trace.append(
            "terminal",
            {"state": terminal_observation.state, "levels_completed": 1},
        )
    except Exception as exc:
        trace.append("error", {"type": type(exc).__name__, "message": str(exc)})
        raise

    return SmokeResult(
        run_id=run_id,
        run_dir=trace.run_dir,
        trace_path=trace.path,
        terminal_state=terminal_observation.state,
        actions=tuple(environment.actions),
        model_requests=1,
        network_requests=network_requests,
    )


def run_deterministic_smoke(
    *,
    config: InferenceConfig,
    runs_root: Path,
    run_id_factory: Callable[[], str] | None = None,
) -> SmokeResult:
    """Exercise the entire episode through a fake without opening a socket."""

    fake_server = DeterministicFakeServer(config.model)
    client = OpenAICompatibleClient(config, transport=fake_server)
    return _run_smoke_episode(
        config=config,
        runs_root=runs_root,
        client=client,
        request_capture=fake_server.requests,
        network_requests=fake_server.network_requests,
        run_id_factory=run_id_factory,
    )


def run_provider_smoke(*, config: InferenceConfig, runs_root: Path) -> SmokeResult:
    """Run the same episode against OpenRouter, RunPod, or local vLLM."""

    return _run_smoke_episode(
        config=config,
        runs_root=runs_root,
        client=OpenAICompatibleClient(config),
        request_capture=None,
        network_requests=1,
    )
