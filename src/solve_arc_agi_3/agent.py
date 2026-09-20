"""Original clean-room baseline loop and official ARC agent adapter."""

from __future__ import annotations

import ast
import base64
import binascii
import importlib
import json
import re
import struct
import zlib
from collections.abc import Callable, Sequence
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from .inference import (
    ChatMessage,
    ImageContentPart,
    ImageUrl,
    InferenceClient,
    InferenceRequest,
    InferenceResponse,
    TextContentPart,
    ToolDefinition,
)

_ACTION_PATTERN = re.compile(r"\b(RESET|ACTION[1-7])\b", re.IGNORECASE)
_COORDINATE_SUFFIX = re.compile(r"^[\s:(]*([0-9]{1,2})[\s,]+([0-9]{1,2})")


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Observation(AgentModel):
    game_id: str
    grid: tuple[tuple[int, ...], ...]
    state: str
    levels_completed: int = Field(ge=0)
    win_levels: int = Field(ge=0)
    available_actions: tuple[int, ...]


class ActionDecision(AgentModel):
    name: str = Field(pattern=r"^(RESET|ACTION[1-7])$")
    data: dict[str, int] = Field(default_factory=dict)
    source: str

    @property
    def action_id(self) -> int:
        return 0 if self.name == "RESET" else int(self.name.removeprefix("ACTION"))


class AgentTurn(AgentModel):
    observation: Observation
    request: InferenceRequest
    response: InferenceResponse
    decision: ActionDecision


_ARC_PALETTE = (
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
    (255, 255, 255),
    (89, 89, 89),
    (178, 223, 138),
    (157, 124, 216),
    (255, 183, 178),
    (163, 140, 63),
)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def grid_png_data_url(grid: tuple[tuple[int, ...], ...]) -> str | None:
    """Encode an ARC grid as a dependency-free RGB PNG data URL."""

    if not grid or not grid[0]:
        return None
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("observation grid is not rectangular")
    scale = 8
    scanlines = bytearray()
    for row in grid:
        encoded_row = bytearray()
        for cell in row:
            if not 0 <= cell < len(_ARC_PALETTE):
                raise ValueError(f"ARC palette index is out of range: {cell}")
            encoded_row.extend(_ARC_PALETTE[cell] * scale)
        for _ in range(scale):
            scanlines.append(0)
            scanlines.extend(encoded_row)
    header = struct.pack(">IIBBBBB", width * scale, len(grid) * scale, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class FrameLike(Protocol):
    game_id: str
    frame: object
    state: object
    levels_completed: int
    win_levels: int
    available_actions: list[int]


def _state_text(state: object) -> str:
    value = getattr(state, "value", state)
    return str(value)


def _normalize_grid(raw_frame: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw_frame, list) or not raw_frame:
        return ()
    possible_grid = raw_frame[-1]
    if not isinstance(possible_grid, list):
        return ()
    rows: list[tuple[int, ...]] = []
    for raw_row in possible_grid:
        if not isinstance(raw_row, list):
            return ()
        if any(not isinstance(cell, int) for cell in raw_row):
            return ()
        rows.append(tuple(cast(list[int], raw_row)))
    return tuple(rows)


def observation_from_frame(frame: FrameLike) -> Observation:
    return Observation(
        game_id=frame.game_id,
        grid=_normalize_grid(frame.frame),
        state=_state_text(frame.state),
        levels_completed=frame.levels_completed,
        win_levels=frame.win_levels,
        available_actions=tuple(frame.available_actions),
    )


def _action_name_from_id(action_id: int) -> str | None:
    if action_id == 0:
        return "RESET"
    if 1 <= action_id <= 7:
        return f"ACTION{action_id}"
    return None


def _decision_from_literal(value: object) -> ActionDecision | None:
    # The observation's available_actions are raw ints (e.g. [1]); a model
    # reading that field back often calls action(1) rather than the string
    # tool-description name, and that is just as valid a legal move.
    if isinstance(value, int) and not isinstance(value, bool):
        name = _action_name_from_id(value)
        return None if name is None else ActionDecision(name=name, source="python_tool")
    if isinstance(value, str):
        name = value.upper()
        if _ACTION_PATTERN.fullmatch(name):
            return ActionDecision(name=name, source="python_tool")
        return None
    if not isinstance(value, dict):
        return None
    raw_name = value.get("action", value.get("action_type"))
    if isinstance(raw_name, int) and not isinstance(raw_name, bool):
        raw_name = _action_name_from_id(raw_name)
    if not isinstance(raw_name, str):
        return None
    name = raw_name.upper()
    if not _ACTION_PATTERN.fullmatch(name):
        return None
    if name != "ACTION6":
        return ActionDecision(name=name, source="python_tool")
    x = value.get("x", value.get("column"))
    y = value.get("y", value.get("row"))
    if type(x) is not int or type(y) is not int:
        return None
    if not 0 <= x <= 63 or not 0 <= y <= 63:
        return None
    return ActionDecision(name=name, data={"x": x, "y": y}, source="python_tool")


def _actions_from_python(code: str) -> list[ActionDecision]:
    """Parse the narrow ``action([...])`` tool surface without executing code."""

    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return []
    actions: list[ActionDecision] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "action":
            continue
        if not node.args:
            continue
        try:
            value = ast.literal_eval(node.args[0])
        except (ValueError, TypeError, SyntaxError):
            continue
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for candidate in candidates:
            decision = _decision_from_literal(candidate)
            if decision is not None:
                actions.append(decision)
    return actions


def parse_action_decision(
    response: InferenceResponse, *, available_actions: Sequence[int]
) -> ActionDecision:
    for tool_call in response.tool_calls:
        if tool_call.function.name != "python":
            continue
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict) or not isinstance(
            arguments.get("code"), str
        ):
            continue
        for decision in _actions_from_python(arguments["code"]):
            if decision.action_id in available_actions:
                return decision

    for match in reversed(list(_ACTION_PATTERN.finditer(response.content))):
        decision = ActionDecision(name=match.group(1).upper(), source="assistant_text")
        if decision.action_id not in available_actions:
            continue
        if decision.name != "ACTION6":
            return decision
        coordinates = _COORDINATE_SUFFIX.match(response.content[match.end() :])
        if coordinates is None:
            continue
        x, y = (int(value) for value in coordinates.groups())
        if 0 <= x <= 63 and 0 <= y <= 63:
            return decision.model_copy(update={"data": {"x": x, "y": y}})
    raise ValueError("model response did not contain an available ARC action")


PYTHON_TOOL = ToolDefinition(
    name="python",
    description=(
        "Inspect the supplied observation and choose an environment move by calling "
        "action with a list containing an available ACTION name. For ACTION6, use "
        "an object with action, row, and column integer fields from 0 through 63."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
)

SYSTEM_PROMPT = """You control an unfamiliar deterministic grid world. Build a compact,
testable account of its mechanics from observed transitions. Prefer small experiments when
the goal is unclear. When it is clear, choose a short safe move. Use the Python tool to call
action with one available action name. Never assume state carries between games."""


class BaselineAgentCore:
    """Provider-independent prompt, model call, and action parsing."""

    def __init__(self, client: InferenceClient) -> None:
        self._client = client
        self._conversation: list[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT)
        ]

    @property
    def conversation(self) -> tuple[ChatMessage, ...]:
        return tuple(self._conversation)

    def request_completion(
        self, observation: Observation
    ) -> tuple[InferenceRequest, InferenceResponse]:
        """Send one model call and return it, without parsing an action yet.

        Split from `choose` so a caller that wants trace visibility even on a
        parse failure -- action parsing can reject a well-formed response --
        can log the request/response before calling `record_decision`, which
        is the step that can raise.
        """

        observation_payload = observation.model_dump(mode="json")
        observation_payload["ascii_grid"] = "\n".join(
            "".join("0123456789ABCDEF"[cell] for cell in row)
            for row in observation.grid
        )
        serialized_observation = json.dumps(
            observation_payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        content_parts: list[TextContentPart | ImageContentPart] = [
            TextContentPart(text=serialized_observation)
        ]
        image_url = grid_png_data_url(observation.grid)
        if image_url is not None:
            content_parts.append(
                ImageContentPart(image_url=ImageUrl(url=image_url, detail="low"))
            )
        user_message = ChatMessage(
            role="user",
            content=tuple(content_parts),
        )
        request = InferenceRequest(
            messages=(*self._conversation, user_message), tools=(PYTHON_TOOL,)
        )
        response = self._client.complete(request)
        self._conversation = list(request.messages)
        return request, response

    def record_decision(
        self, observation: Observation, response: InferenceResponse
    ) -> ActionDecision:
        """Parse an action from a completion and extend conversation history.

        Raises if no available action can be parsed; the caller is
        responsible for having already traced `response` if that matters,
        since this step can fail on an otherwise well-formed reply.
        """

        decision = parse_action_decision(
            response, available_actions=observation.available_actions
        )
        self._conversation.append(
            ChatMessage(
                role="assistant",
                content=response.content or f"Selected {decision.name} via tool.",
            )
        )
        return decision

    def choose(self, observation: Observation) -> AgentTurn:
        request, response = self.request_completion(observation)
        decision = self.record_decision(observation, response)
        return AgentTurn(
            observation=observation,
            request=request,
            response=response,
            decision=decision,
        )


ActionMapper = Callable[[ActionDecision], object]


class _OfficialAction(Protocol):
    def set_data(self, data: dict[str, int]) -> object: ...


class _OfficialGameActionType(Protocol):
    def from_name(self, name: str) -> _OfficialAction: ...


def default_arcengine_action_mapper(decision: ActionDecision) -> object:
    """Map to the official ``arcengine.GameAction`` only when it is installed."""

    try:
        module = importlib.import_module("arcengine")
    except ImportError as exc:  # pragma: no cover - exercised in the Kaggle runtime
        raise RuntimeError(
            "arcengine is required by the official agent adapter"
        ) from exc
    game_action = cast(_OfficialGameActionType, module.GameAction)
    action = game_action.from_name(decision.name)
    if decision.data:
        action.set_data(decision.data)
    return action


class OfficialAgentAdapter:
    """Implements the official starter's ``choose_action``/``is_done`` contract."""

    def __init__(
        self,
        core: BaselineAgentCore,
        *,
        action_mapper: ActionMapper = default_arcengine_action_mapper,
    ) -> None:
        self._core = core
        self._action_mapper = action_mapper

    def choose_action(self, frames: list[FrameLike], latest_frame: FrameLike) -> object:
        source_frame = latest_frame
        if not _normalize_grid(latest_frame.frame):
            for previous in reversed(frames):
                if _normalize_grid(previous.frame):
                    source_frame = previous
                    break
        observation = observation_from_frame(source_frame).model_copy(
            update={
                "state": _state_text(latest_frame.state),
                "available_actions": tuple(latest_frame.available_actions),
                "levels_completed": latest_frame.levels_completed,
                "win_levels": latest_frame.win_levels,
            }
        )
        return self._action_mapper(self._core.choose(observation).decision)

    def is_done(self, frames: list[FrameLike], latest_frame: FrameLike) -> bool:
        del frames
        return _state_text(latest_frame.state) == "WIN"
