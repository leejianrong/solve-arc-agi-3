"""Competition glue: the clean-room agent as an official ``Agent`` subclass.

The official ``Swarm`` runs one thread per game, all at once (110 on the real rerun), each calling
``choose_action`` synchronously. This module adds what that needs and the bare adapter does not:
a wall-clock deadline, a cap on in-flight model calls, a per-game action cap, an automatic RESET
after a game over, and a deterministic fallback move when the model call or its parsing fails, so
one bad reply never ends a game early.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import (
    ActionDecision,
    BaselineAgentCore,
    OfficialAgentAdapter,
    _state_text,
    default_arcengine_action_mapper,
)
from .inference import InferenceClient

logger = logging.getLogger(__name__)

_RESET_STATES = {"NOT_PLAYED", "NOT_STARTED", "GAME_OVER"}


@dataclass(frozen=True)
class RunLimits:
    deadline_epoch: float
    max_concurrent_calls: int = 32
    max_actions_per_game: int = 1500
    # The notebook writes this file once the model server answers. Until then games take only their
    # forced RESET (Kaggle wants an action within 15 minutes of the run starting, and loading the
    # model takes most of that) and wait before their first model call.
    ready_flag: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RunLimits:
        source = os.environ if env is None else env
        return cls(
            deadline_epoch=float(source["ARC3_DEADLINE_EPOCH"]),
            max_concurrent_calls=int(source.get("ARC3_MAX_CONCURRENT_CALLS", "32")),
            max_actions_per_game=int(source.get("ARC3_MAX_ACTIONS", "1500")),
            ready_flag=source.get("ARC3_READY_FLAG") or None,
        )

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.deadline_epoch

    def ready(self) -> bool:
        return self.ready_flag is None or Path(self.ready_flag).exists()

    def wait_ready(self, poll_seconds: float = 2.0) -> bool:
        """Block until the model server is up; False if the deadline passed first."""

        while not self.ready():
            if self.expired():
                return False
            time.sleep(poll_seconds)
        return True


def fallback_decision(available_actions: tuple[int, ...], step: int) -> ActionDecision:
    """Deterministic non-RESET move that cycles through what the game offers."""

    choices = [a for a in available_actions if a != 0] or [1]
    action_id = choices[step % len(choices)]
    data = {"x": 32, "y": 32} if action_id == 6 else {}
    return ActionDecision(name=f"ACTION{action_id}", data=data, source="fallback")


def make_agent_class(
    base: type,
    *,
    limits: RunLimits,
    client: InferenceClient,
    action_mapper: Callable[[ActionDecision], object] = default_arcengine_action_mapper,
) -> type:
    """Build the submission agent on top of the official ``Agent`` (or a test double)."""

    limiter = threading.BoundedSemaphore(limits.max_concurrent_calls)

    class SubmissionAgent(base):  # type: ignore[misc]
        MAX_ACTIONS = limits.max_actions_per_game

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._adapter = OfficialAgentAdapter(
                BaselineAgentCore(client), action_mapper=action_mapper
            )
            self._fallbacks = 0

        def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
            return _state_text(latest_frame.state) == "WIN" or limits.expired()

        def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
            if _state_text(latest_frame.state) in _RESET_STATES:
                return action_mapper(ActionDecision(name="RESET", source="forced"))
            try:
                if not limits.wait_ready():
                    raise TimeoutError(
                        "model server never became ready before the deadline"
                    )
                with limiter:
                    return self._adapter.choose_action(frames, latest_frame)
            except Exception:  # model/parse failure must not end the game
                self._fallbacks += 1
                logger.exception(
                    "model step failed for %s (fallback #%d)",
                    self.game_id,
                    self._fallbacks,
                )
                decision = fallback_decision(
                    tuple(latest_frame.available_actions), self._fallbacks
                )
                return action_mapper(decision)

    SubmissionAgent.__name__ = "MyAgent"
    return SubmissionAgent


def agent_class_from_env(base: type) -> type:
    """Build the submission agent from ``ARC3_*`` environment variables (used by the notebook shim)."""

    from .baseline_config import config_from_manifest
    from .baseline_manifest import load_control_manifest
    from .inference import OpenAICompatibleClient

    manifest_path = Path(os.environ["ARC3_MANIFEST"])
    config = config_from_manifest(
        load_control_manifest(manifest_path),
        manifest_path=manifest_path,
        model_dir=Path("/unused/model"),
        wheelhouse_dir=Path("/unused/wheelhouse"),
    )
    return make_agent_class(
        base,
        limits=RunLimits.from_env(),
        client=OpenAICompatibleClient(config.inference),
    )
