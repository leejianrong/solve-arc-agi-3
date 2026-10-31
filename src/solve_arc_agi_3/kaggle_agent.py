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
from .inference import InferenceClient, InferenceRequest, InferenceResponse
from .step_trace import RunTrace

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

    def failed(self) -> bool:
        """The notebook writes ``<ready_flag>.failed`` when the model server could not start."""

        return (
            self.ready_flag is not None and Path(self.ready_flag + ".failed").exists()
        )

    def ready(self) -> bool:
        return self.ready_flag is None or Path(self.ready_flag).exists()

    def wait_ready(self, poll_seconds: float = 2.0) -> bool:
        """Block until the model server is up; False if the deadline passed first."""

        while not self.ready():
            if self.expired() or self.failed():
                return False
            time.sleep(poll_seconds)
        return True


def fallback_decision(available_actions: tuple[int, ...], step: int) -> ActionDecision:
    """Deterministic non-RESET move that cycles through what the game offers.

    Clicks scan the grid (co-prime strides) instead of repeating one cell, so click-only games at least
    probe different places.
    """

    choices = [a for a in available_actions if a != 0] or [1]
    action_id = choices[step % len(choices)]
    data = {"x": (step * 13) % 64, "y": (step * 29) % 64} if action_id == 6 else {}
    return ActionDecision(name=f"ACTION{action_id}", data=data, source="fallback")


def _reply_fields(
    response: InferenceResponse | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The ``call`` and ``reply`` parts of a trace record for the latest model response."""

    if response is None:
        return {"ok": False}, None
    call = {
        "ms": round(response.elapsed_seconds * 1000),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "ok": True,
    }
    reply = {
        "content": response.content,
        "reasoning_tail": response.reasoning or "",
        "tool_calls": [c.function.arguments for c in response.tool_calls],
    }
    return call, reply


def agent_switches(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The optional behaviours from the environment; every one defaults to off."""

    source = os.environ if env is None else env
    return {
        "prompt_chars": int(source.get("ARC3_PROMPT_CHARS") or "0"),
        "events": source.get("ARC3_EVENTS") == "1",
        "tools": source.get("ARC3_TOOLS") == "1",
        "world_model": source.get("ARC3_WORLD_MODEL") == "1",
    }


def make_agent_class(
    base: type,
    *,
    limits: RunLimits,
    client: InferenceClient,
    action_mapper: Callable[[ActionDecision], object] = default_arcengine_action_mapper,
    trace: RunTrace | None = None,
    use_ledger: bool = False,
    prompt_chars: int | None = None,
    events: bool | None = None,
    tools: bool | None = None,
    world_model: bool | None = None,
) -> type:
    """Build the submission agent on top of the official ``Agent`` (or a test double).

    The four optional switches (see `agent_switches`) fall back to ``ARC3_PROMPT_CHARS``, ``ARC3_EVENTS``,
    ``ARC3_TOOLS`` and ``ARC3_WORLD_MODEL`` when left as None; all off is the plain baseline loop.
    """

    env = agent_switches()
    chars = env["prompt_chars"] if prompt_chars is None else prompt_chars
    use_events = env["events"] if events is None else events
    use_tools = env["tools"] if tools is None else tools
    use_world_model = env["world_model"] if world_model is None else world_model
    if use_world_model and not use_tools:
        raise ValueError("ARC3_WORLD_MODEL=1 requires ARC3_TOOLS=1")
    limiter = threading.BoundedSemaphore(limits.max_concurrent_calls)

    class SubmissionAgent(base):  # type: ignore[misc]
        MAX_ACTIONS = limits.max_actions_per_game

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._adapter = OfficialAgentAdapter(
                BaselineAgentCore(
                    client,
                    use_ledger=use_ledger,
                    prompt_chars=chars,
                    events=use_events,
                    tools=use_tools,
                    world_model=use_world_model,
                ),
                action_mapper=action_mapper,
            )
            self._fallbacks = 0
            self._trace = trace.game(self.game_id) if trace is not None else None
            self._started = False
            self._awaiting: dict[str, Any] | None = None

        def _write_trace(self, frames: list[Any], latest_frame: Any) -> None:
            """Write the start record, and the record of the last move now that its result is visible."""

            if self._trace is None:
                return
            obs = self._adapter.observation(frames, latest_frame)
            if not obs.grid:
                return
            if not self._started:
                self._started = True
                self._trace.start(
                    self.game_id,
                    obs.grid,
                    obs.state,
                    obs.levels_completed,
                    obs.win_levels,
                    obs.available_actions,
                )
                return
            turn, self._awaiting = self._awaiting, None
            if turn is None:
                return
            core = self._adapter.core
            core.settle(obs)
            self._trace.step(
                game=self.game_id,
                src=turn["src"],
                move=turn["move"],
                grid=obs.grid,
                state=obs.state,
                levels=obs.levels_completed,
                win_levels=obs.win_levels,
                avail=obs.available_actions,
                call=turn.get("call"),
                reply=turn.get("reply"),
                plan=turn.get("plan"),
                plan_idx=turn.get("plan_idx"),
                nudge=bool(turn.get("nudge")),
                error=turn.get("error"),
                settled=core.last_settled or None,
                obs_text=turn.get("obs_text"),
                event=turn.get("event"),
            )

        def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
            self._write_trace(frames, latest_frame)
            done = _state_text(latest_frame.state) == "WIN" or limits.expired()
            if done:
                self._adapter.core.close()
            return done

        def cleanup(self, *args: Any, **kwargs: Any) -> Any:
            self._adapter.core.close()
            parent = getattr(super(), "cleanup", None)
            return parent(*args, **kwargs) if parent is not None else None

        def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
            self._write_trace(frames, latest_frame)
            action, turn = self._decide(frames, latest_frame)
            if self._trace is not None:
                self._awaiting = turn
            return action

        def _decide(
            self, frames: list[Any], latest_frame: Any
        ) -> tuple[Any, dict[str, Any]]:
            core = self._adapter.core
            core.last_response = (
                None  # so a failed request is never blamed on an older reply
            )
            if _state_text(latest_frame.state) in _RESET_STATES:
                decision = ActionDecision(name="RESET", source="forced")
                core.note_move(
                    self._adapter.observation(frames, latest_frame), decision
                )
                return action_mapper(decision), {
                    "src": "forced_reset",
                    "move": {"name": "RESET"},
                }
            try:
                if not limits.wait_ready():
                    raise TimeoutError(
                        "model server never became ready before the deadline"
                    )
                with limiter:
                    try:
                        action = self._adapter.choose_action(frames, latest_frame)
                    except (
                        ValueError
                    ):  # the reply held no usable move: ask once more, plainly
                        self._adapter.request_nudge()
                        action = self._adapter.choose_action(frames, latest_frame)
                turn = dict(core.last_turn)
                if turn.get("src") == "model":
                    turn["call"], turn["reply"] = _reply_fields(core.last_response)
                    turn["obs_text"] = core.last_observation_text
                return action, turn
            except Exception as exc:  # model/parse failure must not end the game
                self._fallbacks += 1
                logger.error(
                    "model step failed for %s (fallback #%d): %s",
                    self.game_id,
                    self._fallbacks,
                    describe_reply(core.last_response),
                    exc_info=self._fallbacks <= 2,
                )
                decision = fallback_decision(
                    tuple(latest_frame.available_actions), self._fallbacks
                )
                core.note_move(
                    self._adapter.observation(frames, latest_frame), decision
                )
                call, reply = _reply_fields(core.last_response)
                return action_mapper(decision), {
                    "src": "fallback",
                    "move": core._move(decision),
                    "call": call,
                    "reply": reply,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    SubmissionAgent.__name__ = "MyAgent"
    return SubmissionAgent


def describe_reply(response: InferenceResponse | None) -> str:
    """One line saying what the model sent back (for the log when no move could be read from it)."""

    if response is None:
        return "no reply (request failed or timed out)"
    calls = [c.function.arguments[:200] for c in response.tool_calls]
    return (
        f"tool_calls={calls} content={response.content[:200]!r} "
        f"reasoning_tail={(response.reasoning or '')[-120:]!r}"
    )


class LoggingClient:
    """Wraps a client to log sample replies and rolling stats (tokens, latency, empty answers)."""

    def __init__(
        self, inner: InferenceClient, *, samples: int = 12, every: int = 50
    ) -> None:
        self._inner = inner
        self._samples = samples
        self._every = every
        self._lock = threading.Lock()
        self._calls = 0
        self._tokens = 0
        self._seconds = 0.0
        self._no_tool = 0
        self._errors = 0

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        try:
            response = self._inner.complete(request)
        except Exception as exc:
            with self._lock:
                self._errors += 1
                if self._errors <= 10:
                    logger.warning(
                        "model request failed: %s: %s",
                        type(exc).__name__,
                        str(exc)[:200],
                    )
            raise
        with self._lock:
            self._calls += 1
            self._tokens += response.usage.completion_tokens
            self._seconds += response.elapsed_seconds
            self._no_tool += 0 if response.tool_calls else 1
            n = self._calls
            if n <= self._samples:
                calls = [c.function.arguments[:200] for c in response.tool_calls]
                logger.info(
                    "model sample #%d: %.0fs, %d completion tokens, tool_calls=%s content=%r reasoning=%r",
                    n,
                    response.elapsed_seconds,
                    response.usage.completion_tokens,
                    calls,
                    response.content[:200],
                    (response.reasoning or "")[-200:],
                )
            if n % self._every == 0:
                logger.info(
                    "model stats: %d calls, mean %.0f completion tokens, mean %.1fs, %.0f%% without a tool call",
                    n,
                    self._tokens / n,
                    self._seconds / n,
                    100 * self._no_tool / n,
                )
        return response


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
    # Tunables for the Kaggle run: the pinned 4096-token / 120 s budget cannot finish a thinking answer at
    # ~25 tokens/s per stream (the first dry runs: 354 of 506 steps timed out or ran out of tokens).
    inference = config.inference.model_copy(
        update={
            "max_output_tokens": int(
                os.environ.get(
                    "ARC3_MAX_OUTPUT_TOKENS", config.inference.max_output_tokens
                )
            ),
            "request_timeout_seconds": float(
                os.environ.get(
                    "ARC3_REQUEST_TIMEOUT", config.inference.request_timeout_seconds
                )
            ),
            "sampling": config.inference.sampling.model_copy(
                update={"thinking": os.environ.get("ARC3_THINKING", "1") == "1"}
            ),
        }
    )
    client = LoggingClient(OpenAICompatibleClient(inference))
    trace = None
    if os.environ.get("ARC3_TRACE_DIR"):
        trace = RunTrace(
            Path(os.environ["ARC3_TRACE_DIR"]),
            run_id=os.environ.get("ARC3_RUN_ID") or None,
            keep_obs=os.environ.get("ARC3_TRACE_OBS") == "1",
            settings={
                "model": inference.model,
                "thinking": inference.sampling.thinking,
                "max_output_tokens": inference.max_output_tokens,
                "request_timeout_seconds": inference.request_timeout_seconds,
                "max_concurrent_calls": RunLimits.from_env().max_concurrent_calls,
                "bundle_commit": os.environ.get("ARC3_BUNDLE_COMMIT", ""),
            },
        )
    return make_agent_class(
        base,
        limits=RunLimits.from_env(),
        client=client,
        trace=trace,
        use_ledger=os.environ.get("ARC3_LEDGER") == "1",
    )
