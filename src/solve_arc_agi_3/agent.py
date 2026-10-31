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
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

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
from .ledger import Ledger
from .observation import (
    MAX_COUNTER_CELLS,
    ascii_grid,
    change_boxes,
    changed_count,
    describe_change,
    describe_key,
    describe_motion,
    describe_objects,
    object_motion,
    region_key,
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
    # ACTION6 (a click) is never valid without coordinates.
    if isinstance(value, int) and not isinstance(value, bool):
        name = _action_name_from_id(value)
        if name is None or name == "ACTION6":
            return None
        return ActionDecision(name=name, source="python_tool")
    if isinstance(value, str):
        name = value.upper()
        if _ACTION_PATTERN.fullmatch(name) and name != "ACTION6":
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
    return _click(x, y)


def _click(x: object, y: object) -> ActionDecision | None:
    if type(x) is not int or type(y) is not int:
        return None
    if not 0 <= x <= 63 or not 0 <= y <= 63:
        return None
    return ActionDecision(name="ACTION6", data={"x": x, "y": y}, source="python_tool")


def _is_click_name(value: object) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return value == 6
    return isinstance(value, str) and value.upper() == "ACTION6"


def _actions_from_python(code: str) -> list[ActionDecision]:
    """Parse the narrow ``action(...)`` tool surface without executing code.

    Accepted: ``action(1)``, ``action([1, 2, 4])``, ``action("ACTION3")``, a list mixing those with
    ``{"action": 6, "x": 3, "y": 9}``, and the flat click forms ``action(6, 3, 9)`` / ``action(6, x=3, y=9)``.
    """

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
        if _is_click_name(value):
            coords: list[object] = []
            for extra in node.args[1:3]:
                try:
                    coords.append(ast.literal_eval(extra))
                except (ValueError, TypeError, SyntaxError):
                    coords.append(None)
            keywords = {}
            for kw in node.keywords:
                try:
                    keywords[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
            x = (
                coords[0]
                if len(coords) > 0
                else keywords.get("x", keywords.get("column"))
            )
            y = coords[1] if len(coords) > 1 else keywords.get("y", keywords.get("row"))
            click = _click(x, y)
            if click is not None:
                actions.append(click)
            continue
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for candidate in candidates:
            decision = _decision_from_literal(candidate)
            if decision is not None:
                actions.append(decision)
    return actions


MAX_PLAN = 5


def parse_action_decisions(
    response: InferenceResponse,
    *,
    available_actions: Sequence[int],
    limit: int = MAX_PLAN,
) -> list[ActionDecision]:
    """Every legal action in the reply, in order (at most ``limit``); raises if there is none."""

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
        legal = [
            d
            for d in _actions_from_python(arguments["code"])
            if d.action_id in available_actions
        ]
        if legal:
            return legal[:limit]

    for call in re.finditer(r"action\((?:[^()]|\([^()]*\))*\)", response.content):
        written = [
            d
            for d in _actions_from_python(call.group(0))
            if d.action_id in available_actions
        ]
        if written:
            return written[:limit]

    for match in reversed(list(_ACTION_PATTERN.finditer(response.content))):
        decision = ActionDecision(name=match.group(1).upper(), source="assistant_text")
        if decision.action_id not in available_actions:
            continue
        if decision.name != "ACTION6":
            return [decision]
        coordinates = _COORDINATE_SUFFIX.match(response.content[match.end() :])
        if coordinates is None:
            continue
        x, y = (int(value) for value in coordinates.groups())
        if 0 <= x <= 63 and 0 <= y <= 63:
            return [decision.model_copy(update={"data": {"x": x, "y": y}})]
    raise ValueError("model response did not contain an available ARC action")


def parse_action_decision(
    response: InferenceResponse, *, available_actions: Sequence[int]
) -> ActionDecision:
    return parse_action_decisions(
        response, available_actions=available_actions, limit=1
    )[0]


PYTHON_TOOL = ToolDefinition(
    name="python",
    description=(
        "Play moves. Call action(...) once with a list of up to 5 moves played in order, for example "
        "action([1, 1, 4]) or action([{'action': 6, 'x': 12, 'y': 30}, 2]). Moves are action numbers "
        "(1-5, 7) or click objects {'action': 6, 'x': column, 'y': row} with coordinates 0-63. "
        "Only actions listed as available can be used."
    ),
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
)

SYSTEM_PROMPT = """You are playing an unfamiliar turn-based grid game. No rules are given: learn them by acting and
watching what changes, then use them to clear the level (levels_completed goes up when you do). Cells are colours
written as hex digits 0-f.

Actions: 1-5 are plain moves or interactions (often up, down, left, right and interact, but verify by trying).
Action 6 clicks one cell and needs x (column) and y (row): choose them from the object list, whose centres are
given. Action 7 is undo in some games. Only actions listed as available can be used.

How to work: in a new game try each available action once and read what changed. Form a guess about the goal and
the controls, test it with cheap moves, and keep what you learn. Never repeat an action that changed nothing: if
two tries changed nothing, do something different. A level is cleared by reaching its goal, not by changing the
most cells.

Reply with one python tool call and nothing else: call action(...) with a list of up to 5 moves to play in order,
for example action([1, 1, 4]) or action([{"action": 6, "x": 12, "y": 30}]). Use one move while exploring and a
longer list only when you are sure of the controls."""


HISTORY_LINES = 40
_ACTION_LABELS = {
    "RESET": "RESET",
    **{f"ACTION{i}": f"ACTION{i}" for i in range(1, 8)},
}


class BaselineAgentCore:
    """Provider-independent prompt, model call, and action parsing.

    Context stays bounded: every call is the system prompt plus one user message holding a short
    text history (what was tried and what it changed), what each action has done so far, and the
    *current* observation (objects and grid). Old frames and images are never resent, so a long
    game cannot overflow the model's context window. A reply may carry up to five moves; the rest
    are played without another model call while the game stays in the same state.
    """

    def __init__(
        self,
        client: InferenceClient,
        *,
        history_lines: int = HISTORY_LINES,
        use_ledger: bool = False,
        ledger_chars: int = 1200,
    ) -> None:
        self._use_ledger = use_ledger
        self._ledger_chars = ledger_chars
        self._ledger_records: list[dict[str, Any]] = []
        self._pending_move: dict[str, object] = {}
        self._client = client
        self._history_lines = history_lines
        self._history: list[str] = []
        self._pending: tuple[str, str] | None = None  # (label, action key)
        self._previous: Observation | None = None
        self._queue: list[ActionDecision] = []
        self._plan_levels = 0
        self._effects: dict[
            str, list[int]
        ] = {}  # action key -> [tries, tries that changed the grid]
        self._noop_streak = 0
        self._last_change = "no action taken yet"
        self._steps = 0
        self.last_response: InferenceResponse | None = None
        self._nudge = False
        self._last_available: tuple[int, ...] = ()
        self._box_streaks: dict[tuple[object, ...], int] = {}
        self._motions: dict[str, Counter[str]] = {}
        self._since_call: list[str] = []
        # What the last settled move did, for the trace writer (change, motion, counters, effects).
        self.last_settled: dict[str, object] = {}
        # Which move was just chosen and why, for the trace writer.
        self.last_turn: dict[str, object] = {}
        self.last_observation_text = ""
        self._plan_full: list[ActionDecision] = []
        self._plan_pos = 0
        self._call_nudged = False

    def _counter_regions(self) -> list[str]:
        """Small regions that changed on each of the last 3+ actions: probably a counter or timer."""

        return [
            describe_key(key)
            for key, streak in sorted(
                self._box_streaks.items(), key=lambda kv: str(kv[0])
            )
            if streak >= 3
        ]

    @staticmethod
    def _move(decision: ActionDecision) -> dict[str, object]:
        move: dict[str, object] = {"name": decision.name}
        if decision.data:
            move.update(x=decision.data["x"], y=decision.data["y"])
        return move

    def settle(self, observation: Observation) -> None:
        """Record the result of the last move now (idempotent; the next call would do it anyway)."""

        self._settle_pending(observation)

    def note_move(self, observation: Observation, decision: ActionDecision) -> None:
        """Record a move the model did not choose (a reset or a fallback) so history and the trace include it."""

        self._queue.clear()
        self._previous = observation
        self._note_pending(decision)

    def request_nudge(self) -> None:
        """Ask for the move properly on the next call (the last reply had no usable action)."""

        self._nudge = True

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def _settle_pending(self, observation: Observation) -> None:
        """Turn the last action into a history line now that its result is visible."""

        if self._pending is None or self._previous is None:
            return
        label, key = self._pending
        before, after = self._previous.grid, observation.grid
        change = describe_change(before, after)
        gained = observation.levels_completed - self._previous.levels_completed
        raw = changed_count(before, after)
        height = len(after)
        width = len(after[0]) if height else 0
        boxes = change_boxes(before, after) if raw > 0 else []
        flagged = {k for k, streak in self._box_streaks.items() if streak >= 3}
        keys = [region_key(box, width, height) for box in boxes]
        small = {
            k for k, box in zip(keys, boxes, strict=True) if box[4] <= MAX_COUNTER_CELLS
        }
        self._box_streaks = {k: self._box_streaks.get(k, 0) + 1 for k in small}
        # A change confined to regions already flagged as counters does not count as the action doing anything.
        changed = raw < 0 or any(k not in flagged for k in keys)
        pairs = object_motion(before, after)
        if pairs:
            moved_from, moved_to = pairs[0]
            kind = (
                f"moves color {moved_from.color} size {moved_from.size} "
                f"by ({moved_to.x0 - moved_from.x0:+d},{moved_to.y0 - moved_from.y0:+d})"
            )
            self._motions.setdefault(key, Counter())[kind] += 1
        tries = self._effects.setdefault(key, [0, 0])
        tries[0] += 1
        tries[1] += 1 if changed else 0
        self._noop_streak = 0 if changed else self._noop_streak + 1
        level = f"; LEVEL CLEARED (+{gained})" if gained > 0 else ""
        note = (
            " (only a counter or timer region changed)"
            if raw > 0 and not changed
            else ""
        )
        moved = f"; motion: {describe_motion(pairs)}" if pairs else ""
        self._last_change = f"{label}: {change}{note}{moved}{level}"
        self._history.append(
            f"{self._steps}. {label} -> {observation.state}, {change[:150]}{note}{level}"
        )
        self._since_call.append(f"{label}: {change[:110]}{note}{level}")
        del self._history[: -self._history_lines]
        self.last_settled = {
            "n": max(raw, 0),
            "bbox": (
                [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ]
                if boxes
                else None
            ),
            "regions": [list(b) for b in boxes[:6]],
            "motion": [
                {
                    "color": a.color,
                    "size": a.size,
                    "from": list(a.center),
                    "to": list(b.center),
                }
                for a, b in pairs
            ],
            "counters": self._counter_regions(),
            "effects": {k: list(v) for k, v in self._effects.items()},
        }
        if self._use_ledger:
            self._ledger_records.append(
                {
                    "kind": "step",
                    "step": self._steps - 1,
                    "moves": [self._pending_move],
                    "levels": observation.levels_completed,
                    "change": {"n": max(raw, 0)},
                    "motion": self.last_settled["motion"],
                    "counters": self.last_settled["counters"],
                }
            )
        self._pending = None

    def _effects_text(self) -> str:
        if not self._effects:
            return "(none yet)"
        parts = []
        for key in sorted(self._effects):
            tries, changed = self._effects[key]
            part = f"{key} changed the grid {changed}/{tries} times" + (
                " (never changed anything)" if changed == 0 else ""
            )
            seen = self._motions.get(key)
            if seen:
                text, n = seen.most_common(1)[0]
                part += f"; {text} ({n}x)"
            parts.append(part)
        return "; ".join(parts)

    def _observation_text(self, observation: Observation) -> str:
        names = ", ".join(
            _ACTION_LABELS.get(_name_or_none(a) or "", str(a))
            for a in observation.available_actions
        )
        warning = (
            f"\nWARNING: your last {self._noop_streak} actions changed nothing. Try something different."
            if self._noop_streak >= 2
            else ""
        )
        history = "\n".join(self._history[-12:]) or "(no actions yet)"
        reminder = ""
        self._call_nudged = self._nudge
        if self._nudge:
            self._nudge = False
            reminder = self._nudge_text(names)
        text = (
            reminder
            + f"Game state: {observation.state}; levels completed {observation.levels_completed} of "
            f"{observation.win_levels}. Available actions: {names}.\n"
            f"Result of your last action: {self._last_change}.{warning}\n"
            f"{self._plan_results()}"
            f"What each action has done so far: {self._effects_text()}\n"
            f"{self._untried_line(observation)}"
            f"{self._ledger_block()}"
            f"{self._counter_line()}"
            f"Recent steps:\n{history}\n\n"
            f"{describe_objects(observation.grid)}\n\n"
            f"Grid (rows top to bottom, x is the column, hex colours):\n{ascii_grid(observation.grid)}"
        )
        self.last_observation_text = text
        return text

    def _ledger_block(self) -> str:
        """What the moves so far show, as claims with evidence (computed from this game's own steps)."""

        if not self._use_ledger or not self._ledger_records:
            return ""
        text = Ledger.from_trace(self._ledger_records).render(self._ledger_chars)
        return f"Hypotheses so far, computed from your moves:\n{text}\n" if text else ""

    def _plan_results(self) -> str:
        """Move-by-move effects of the last multi-move plan (a single move is already in the line above)."""

        if len(self._since_call) < 2:
            return ""
        lines = "\n".join(
            f"  {i}) {text}" for i, text in enumerate(self._since_call, 1)
        )
        return f"Results of your last plan, move by move:\n{lines}\n"

    def _untried_line(self, observation: Observation) -> str:
        if not self._effects:
            return ""
        untried = [
            f"ACTION{a}"
            for a in observation.available_actions
            if a != 0 and f"ACTION{a}" not in self._effects
        ]
        return f"Not tried yet in this game: {', '.join(untried)}.\n" if untried else ""

    def _counter_line(self) -> str:
        regions = self._counter_regions()
        if not regions:
            return ""
        return (
            "Regions that changed on every recent action (probably a step counter or timer, not part of the "
            f"puzzle): {', '.join(regions)}\n"
        )

    def _nudge_text(self, available_names: str) -> str:
        """Why the last reply was unusable, as specifically as we can tell."""

        response = self.last_response
        illegal: list[str] = []
        click_without_coordinates = False
        if response is not None:
            for call in response.tool_calls:
                try:
                    code = json.loads(call.function.arguments).get("code", "")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if not isinstance(code, str):
                    continue
                for decision in _actions_from_python(code):
                    if decision.action_id not in self._last_available:
                        illegal.append(decision.name)
                if re.search(r"\bACTION6\b|\b6\b", code) and "x" not in code.lower():
                    click_without_coordinates = True
        if illegal:
            return (
                f"Your previous reply used {', '.join(sorted(set(illegal)))}, which is not available now "
                f"(available: {available_names}). Reply with only the python tool call, using an available "
                "action.\n"
            )
        if click_without_coordinates:
            return (
                "Your previous click had no coordinates. A click needs x (column) and y (row), for example "
                'action([{"action": 6, "x": 12, "y": 30}]). Reply with only the python tool call.\n'
            )
        return (
            "Your previous reply contained no action. Reply now with only the python tool call, "
            "for example action([1]).\n"
        )

    def queued_decision(self, observation: Observation) -> ActionDecision | None:
        """The next move of the current plan, or None when the plan is spent or no longer safe."""

        if not self._queue:
            return None
        unsafe = (
            observation.state != "NOT_FINISHED"
            or observation.levels_completed != self._plan_levels
            or self._noop_streak >= 2
        )
        self._settle_pending(observation)
        if unsafe or self._noop_streak >= 2:
            self._queue.clear()
            return None
        decision = self._queue.pop(0)
        if decision.action_id not in observation.available_actions:
            self._queue.clear()
            return None
        self._previous = observation
        self._note_pending(decision)
        self._plan_pos += 1
        self.last_turn = {
            "src": "queued",
            "move": self._move(decision),
            "plan": [self._move(d) for d in self._plan_full],
            "plan_idx": self._plan_pos,
            "nudge": False,
        }
        return decision

    def _note_pending(self, decision: ActionDecision) -> None:
        self._steps += 1
        self._pending_move = self._move(decision)
        coords = (
            f" at x={decision.data['x']},y={decision.data['y']}"
            if decision.data
            else ""
        )
        self._pending = (f"{decision.name}{coords}", decision.name)

    def request_completion(
        self, observation: Observation
    ) -> tuple[InferenceRequest, InferenceResponse]:
        """Send one model call and return it, without parsing an action yet.

        Split from `choose` so a caller that wants trace visibility even on a
        parse failure -- action parsing can reject a well-formed response --
        can log the request/response before calling `record_decision`, which
        is the step that can raise.
        """

        self._settle_pending(observation)
        self._queue.clear()
        self._last_available = observation.available_actions
        content_parts: list[TextContentPart | ImageContentPart] = [
            TextContentPart(text=self._observation_text(observation))
        ]
        self._since_call = []
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
            messages=(ChatMessage(role="system", content=SYSTEM_PROMPT), user_message),
            tools=(PYTHON_TOOL,),
        )
        response = self._client.complete(request)
        self.last_response = response
        self._previous = observation
        return request, response

    def record_decision(
        self, observation: Observation, response: InferenceResponse
    ) -> ActionDecision:
        """Parse the moves from a completion; play the first now and queue the rest.

        Raises if no available action can be parsed; the caller is
        responsible for having already traced `response` if that matters,
        since this step can fail on an otherwise well-formed reply.
        """

        decisions = parse_action_decisions(
            response, available_actions=observation.available_actions
        )
        self._queue = decisions[1:]
        self._plan_levels = observation.levels_completed
        self._note_pending(decisions[0])
        self._plan_full, self._plan_pos = decisions, 0
        self.last_turn = {
            "src": "model",
            "move": self._move(decisions[0]),
            "plan": [self._move(d) for d in decisions],
            "plan_idx": 0,
            "nudge": self._call_nudged,
        }
        return decisions[0]

    def choose(self, observation: Observation) -> AgentTurn:
        request, response = self.request_completion(observation)
        decision = self.record_decision(observation, response)
        return AgentTurn(
            observation=observation,
            request=request,
            response=response,
            decision=decision,
        )


def _name_or_none(action_id: int) -> str | None:
    return _action_name_from_id(action_id)


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

    def observation(
        self, frames: list[FrameLike], latest_frame: FrameLike
    ) -> Observation:
        """The model-facing view of the latest frame (falls back to the last non-empty grid)."""

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
        return observation

    def choose_action(self, frames: list[FrameLike], latest_frame: FrameLike) -> object:
        observation = self.observation(frames, latest_frame)
        queued = self._core.queued_decision(observation)
        if queued is not None:
            return self._action_mapper(queued)
        return self._action_mapper(self._core.choose(observation).decision)

    @property
    def last_response(self) -> InferenceResponse | None:
        return self._core.last_response

    @property
    def core(self) -> BaselineAgentCore:
        return self._core

    def request_nudge(self) -> None:
        self._core.request_nudge()

    def is_done(self, frames: list[FrameLike], latest_frame: FrameLike) -> bool:
        del frames
        return _state_text(latest_frame.state) == "WIN"
