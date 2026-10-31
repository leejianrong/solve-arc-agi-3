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
from .ledger import Kind, Ledger
from .observation import (
    MAX_COUNTER_CELLS,
    MAX_OBJECTS,
    Region,
    ascii_grid,
    change_boxes,
    changed_count,
    describe_change,
    describe_key,
    describe_motion,
    describe_objects,
    object_motion,
    region_key,
    segment,
    untried_clicks,
)
from .step_trace import encode_rows, grid_rows

# Loop-breaker: a move is "dead" after this many no-ops on the identical board, or (clicks only) this many
# tries anywhere without ever changing anything. The agent then swaps it for something untried.
DEAD_ON_SAME_BOARD = 2
DEAD_CLICK_TRIES = 3

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


MAX_TOOL_CALLS = 3  # hypothesis()/evidence() calls read from one reply
MAX_STATEMENT = 200
MAX_WORLD_MODEL_SOURCE = 20000
MAX_WORLD_MODEL_MESSAGE = 600
_KINDS = ("control", "mechanic", "goal", "counter", "other")

_TOOLS_TOOL_TEXT = (
    " The same code may first record what you believe: hypothesis('statement', 'mechanic'), "
    "hypothesis('statement', 'goal', 'what you predict'), evidence('h3', 'support' or 'counter', 'note')."
)
_TOOLS_PROMPT = """

The python call may also keep your notes on the game, before action(...): hypothesis("statement", "mechanic")
records a claim (kind is one of control, mechanic, goal, counter, other; an optional third argument is what it
predicts); evidence("h3", "support", "note") adds support, or "counter", evidence to your claim h3. At most 3
such calls per reply, statements up to 200 characters. Your claims come back with their ids in the hypotheses
block."""
_WORLD_MODEL_TOOL_TEXT = (
    " submit_world_model('''def predict(frame, move): ...''') offers a model of the game; it is checked "
    "against every move so far."
)
_WORLD_MODEL_PROMPT = '''
submit_world_model("""def predict(frame, move): ...""") offers Python source that predicts the next frame (a
list of hex-digit row strings) from the current frame and the move ({'name': 'ACTION3'}, plus 'x' and 'y' for a
click). It is accepted only if it reproduces every move played so far exactly; the answer comes back in the next
message. Only math, itertools, collections, functools, copy, heapq, bisect and re may be imported.'''


def tool_texts(tools: bool, world_model: bool) -> tuple[ToolDefinition, str]:
    """The python tool and system prompt; the extras appear only when their switch is on."""

    if not tools:
        return PYTHON_TOOL, SYSTEM_PROMPT
    description = PYTHON_TOOL.description + _TOOLS_TOOL_TEXT
    prompt = SYSTEM_PROMPT + _TOOLS_PROMPT
    if world_model:
        description += _WORLD_MODEL_TOOL_TEXT
        prompt += _WORLD_MODEL_PROMPT
    return PYTHON_TOOL.model_copy(update={"description": description}), prompt


_MISSING = object()


def _literal(node: ast.expr) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return _MISSING


def _tool_calls_from_python(code: str) -> list[tuple[str, list[object]]]:
    """hypothesis(...), evidence(...) and submit_world_model(...) calls in source order (nothing runs)."""

    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return []
    found: list[tuple[int, int, str, list[object]]] = []
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("hypothesis", "evidence", "submit_world_model")
        ):
            args = [_literal(a) for a in node.args]
            found.append((node.lineno, node.col_offset, node.func.id, args))
    found.sort(key=lambda f: (f[0], f[1]))
    return [(name, args) for _, _, name, args in found]


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
        prompt_chars: int = 0,
        events: bool = False,
        tools: bool = False,
        world_model: bool = False,
    ) -> None:
        if world_model and not tools:
            raise ValueError("the world_model switch requires the tools switch")
        self._prompt_chars = prompt_chars
        self._events = events
        self._tools = tools
        self._world_model = world_model
        self._tool, self._system_prompt = tool_texts(tools, world_model)
        # Prompt text that was cut to fit prompt_chars even with every trimmable part gone.
        self.compaction_overflows = 0
        self._surprise = ""
        self._last_call_levels: int | None = None
        self.last_event: str | None = None
        self._model_ledger = Ledger()
        self._game_id = "game"
        self._wm_records: list[dict[str, Any]] = []
        self._wm_log: Any = None
        self._wm_result: Any = None
        self._wm_message = ""
        self._wm_seq = 0  # which submission the message is about
        self._wm_shown_seq = -1  # the one the latest prompt carried
        self._executor: Any = None  # PlanExecutor of the planner's current plan (V3)
        self._exec_levels = 0
        self._pending_planned = False  # the pending move came from the planner
        self._plan_key: tuple[int, int] | None = (
            None  # (revision, levels) already planned
        )
        self._goal_predicate: Callable[[tuple[tuple[int, ...], ...]], bool] | None = (
            None
        )
        self.last_mismatch: dict[str, Any] = {}
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
        self._moves: dict[
            str, list[int]
        ] = {}  # exact move label -> [tries, tries that changed the grid]
        self._dead: Counter[tuple[str, int]] = (
            Counter()
        )  # (move label, board hash) -> no-ops
        self._clicks: list[tuple[int, int]] = []
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
        self._mover: Region | None = None  # last object seen moving
        self._move_status = ""
        self._ignored_calls = 0  # extra parallel tool calls dropped from the last reply

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
        self._executor = None
        self._previous = observation
        self._note_pending(decision)

    def set_goal_predicate(
        self, goal: Callable[[tuple[tuple[int, ...], ...]], bool] | None
    ) -> None:
        """A target test on frames (host side) for planning when the model's source has no `is_goal`."""

        self._goal_predicate = goal

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
        strip_only = bool(keys) and all(k[0] in ("row", "col") for k in keys)
        if pairs:
            self._mover = pairs[0][1]
            dx, dy = pairs[0][1].x0 - pairs[0][0].x0, pairs[0][1].y0 - pairs[0][0].y0
            self._move_status = f"{label} MOVED the object by ({dx:+d},{dy:+d})"
        elif raw == 0 or (raw > 0 and (not changed or strip_only)):
            why = "nothing changed" if raw == 0 else "only a counter or timer changed"
            self._move_status = f"{label} was BLOCKED ({why}); the object did not move"
        else:
            self._move_status = f"{label} changed the grid but no object moved"
        exact = self._moves.setdefault(label, [0, 0])
        exact[0] += 1
        exact[1] += 1 if changed else 0
        if not changed:
            self._dead[(label, hash(before))] += 1
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
        if self._world_model:
            self._record_world_model(observation)
        if self._executor is not None and self._pending_planned:
            self._check_plan(observation, gained)
        if self._events and self._queue and gained <= 0:
            self._check_surprise(key)
        if self._use_ledger or self._events:
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

    def _status_header(self, observation: Observation) -> str:
        if not self._move_status:
            return self._untried_line(observation)
        m = self._mover
        where = (
            f"Controlled object (colour {m.color}, size {m.size}) is at x{m.x0}-{m.x1} y{m.y0}-{m.y1}, "
            f"centre {m.center}. "
            if m
            else "No object has been seen moving yet. "
        )
        return f"STATUS: {where}Last move: {self._move_status}.\n{self._untried_line(observation)}"

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
        history = self._history[-6:] or ["(no actions yet)"]
        reminder = ""
        self._call_nudged = self._nudge
        if self._nudge:
            self._nudge = False
            reminder = self._nudge_text(names)
        self._wm_shown_seq = self._wm_seq if self._wm_message else -1
        # Mandatory parts (never trimmed): header, last result, warnings, what each action did, the grid.
        ignored = (
            f"NOTE: your last reply had {self._ignored_calls} extra tool call(s); only the first "
            "action-bearing call was used, the rest were ignored. Send one call per turn.\n"
            if self._ignored_calls
            else ""
        )
        head = (
            reminder
            + self._status_header(observation)
            + ignored
            + f"Game state: {observation.state}; levels completed {observation.levels_completed} of "
            f"{observation.win_levels}. Available actions: {names}.\n"
            f"Result of your last action: {self._last_change}.{warning}\n"
            + (f"{self._surprise}\n" if self._surprise else "")
            + (f"{self._wm_message}\n" if self._wm_message else "")
        )
        effects = f"What each action has done so far: {self._effects_text()}\n"
        counters = self._counter_line()
        grid = (
            "Grid (rows top to bottom, x is the column, hex colours):\n"
            f"{ascii_grid(observation.grid)}"
        )

        def build(plan: str, hist: list[str] | None, ledger: str, objects: str) -> str:
            recent = "Recent steps:\n" + "\n".join(hist) + "\n\n" if hist else ""
            return f"{head}{plan}{effects}{ledger}{counters}{recent}{objects}\n\n{grid}"

        plan_text = self._plan_results()
        ledger_text = self._ledger_block()
        objects_text = describe_objects(observation.grid)
        text = build(plan_text, history, ledger_text, objects_text)
        budget = self._prompt_chars
        if budget > 0 and len(text) > budget:
            text = self._compact(
                budget, observation, build, history, objects_text, bool(ledger_text)
            )
        self.last_observation_text = text
        return text

    def _compact(
        self,
        budget: int,
        observation: Observation,
        build: Callable[[str, list[str] | None, str, str], str],
        history: list[str],
        objects_text: str,
        has_ledger: bool,
    ) -> str:
        """Trim to `budget` characters in a fixed order: plan results, oldest history lines, the
        hypotheses block (shrunk to the room left, then dropped), the object list from its smallest
        objects up. Header, last result, warnings, per-action effects and the grid are never cut; if
        they alone exceed the budget the text is returned without the trimmable parts and
        `compaction_overflows` goes up."""

        ledger_text = self._ledger_block()
        text = build("", history, ledger_text, objects_text)
        if len(text) <= budget:
            return text
        for dropped in range(1, len(history) + 1):
            text = build("", history[dropped:] or None, ledger_text, objects_text)
            if len(text) <= budget:
                return text
        if has_ledger:
            room = budget - len(build("", None, "", objects_text))
            block = self._ledger_block(room)
            if block:
                text = build("", None, block, objects_text)
                if len(text) <= budget:
                    return text
        shown = min(len(segment(observation.grid)), MAX_OBJECTS)
        low, high = 0, shown
        best = describe_objects(observation.grid, limit=0)
        while (
            low <= high
        ):  # the longest object list that fits (length grows with the limit)
            mid = (low + high) // 2
            candidate = describe_objects(observation.grid, limit=mid)
            if len(build("", None, "", candidate)) <= budget:
                best, low = candidate, mid + 1
            else:
                high = mid - 1
        text = build("", None, "", best)
        if len(text) > budget:
            self.compaction_overflows += 1
        return text

    def _ledger_block(self, budget: int | None = None) -> str:
        """What the moves so far show, as claims with evidence (computed from this game's own steps,
        plus the claims the model recorded when the tools switch is on). `budget` caps the block."""

        use_computed = self._use_ledger and bool(self._ledger_records)
        if not use_computed and not (self._tools and self._model_ledger.hypotheses):
            return ""
        ledger = Ledger()
        ledger.merge(self._model_ledger)  # first, so the model's ids stay h1, h2, ...
        if use_computed:
            ledger.merge(Ledger.from_trace(self._ledger_records))
        header = (
            "Hypotheses so far (your own claims keep their ids; cite them in evidence()):\n"
            if self._tools
            else "Hypotheses so far, computed from your moves:\n"
        )
        limit = self._ledger_chars
        if budget is not None:
            limit = min(limit, budget - len(header) - 1)
        if limit <= 0:
            return ""
        text = ledger.render(limit)
        return f"{header}{text}\n" if text else ""

    def _check_surprise(self, action: str) -> None:
        """Drop the rest of the plan when the move just settled contradicts a supported claim about it."""

        if "x" in self._pending_move:
            return
        before = Ledger.from_trace(
            self._ledger_records
        )  # without the move just settled
        expected = before.expected_motion(action)
        motion = cast(list[dict[str, Any]], self.last_settled["motion"])
        for color, size, dx, dy in expected:
            got = [
                (m["to"][0] - m["from"][0], m["to"][1] - m["from"][1])
                for m in motion
                if (m["color"], m["size"]) == (color, size)
            ]
            if (dx, dy) in got:
                continue
            result = f"it moved by {got}" if got else "nothing moved"
            self._surprise = (
                f"SURPRISE: {action} was expected to move the object (colour {color}, size {size}) "
                f"by ({dx}, {dy}) but {result}. The rest of your plan was dropped."
            )
            self._queue.clear()
            return

    def _event_for(self, observation: Observation) -> str:
        if self._nudge and self.last_event:  # a retry of the same call keeps its reason
            return self.last_event
        if (
            self._last_call_levels is None
            or observation.levels_completed != self._last_call_levels
        ):
            return "level_start"
        if self._surprise:
            return "surprise"
        if self._noop_streak >= 2:
            return "no_progress"
        return "plan_exhausted"

    @staticmethod
    def _calls_to_use(
        response: InferenceResponse, available: Sequence[int]
    ) -> list[Any]:
        """Tool calls up to and including the first one that carries a legal action. Later parallel calls
        are dropped: their results would come from a board state the first call already changed."""

        calls = list(response.tool_calls)
        for i, call in enumerate(calls):
            if call.function.name != "python":
                continue
            try:
                args = json.loads(call.function.arguments)
                code = args["code"]
                if not isinstance(code, str):
                    continue
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if any(d.action_id in available for d in _actions_from_python(code)):
                return calls[: i + 1]
        return calls

    def _apply_tool_calls(
        self, response: InferenceResponse, calls: list[Any] | None = None
    ) -> None:
        """Record the model's hypothesis()/evidence() calls (at most MAX_TOOL_CALLS per reply; bad ones are ignored)."""

        budget = MAX_TOOL_CALLS
        step = max(self._steps - 1, 0)
        for call in response.tool_calls if calls is None else calls:
            if call.function.name != "python":
                continue
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                continue
            if not isinstance(arguments, dict) or not isinstance(
                arguments.get("code"), str
            ):
                continue
            for name, args in _tool_calls_from_python(arguments["code"]):
                if name == "submit_world_model":
                    if self._world_model:
                        self._submit_world_model(args)
                    continue
                if budget <= 0:
                    continue
                budget -= 1
                if name == "hypothesis":
                    self._tool_hypothesis(args, step)
                else:
                    self._tool_evidence(args, step)

    def _tool_hypothesis(self, args: list[object], step: int) -> None:
        if len(args) not in (2, 3) or any(not isinstance(a, str) for a in args):
            return
        statement, kind = cast(str, args[0]).strip(), args[1]
        prediction = cast(str, args[2]).strip() if len(args) == 3 else None
        if not statement or kind not in _KINDS:
            return
        self._model_ledger.propose(
            statement[:MAX_STATEMENT],
            cast(Kind, kind),
            step,
            prediction[:MAX_STATEMENT] if prediction else None,
        )

    def _tool_evidence(self, args: list[object], step: int) -> None:
        if len(args) not in (2, 3) or any(not isinstance(a, str) for a in args):
            return
        hypothesis_id, polarity = args[0], args[1]
        note = cast(str, args[2]) if len(args) == 3 else ""
        if polarity not in ("support", "counter") or not any(
            h.id == hypothesis_id for h in self._model_ledger.hypotheses
        ):
            return
        self._model_ledger.observe(
            cast(str, hypothesis_id),
            step,
            cast(Any, polarity),
            note,
        )

    def _record_world_model(self, after: Observation) -> None:
        """Keep trace-style records (keyframes only) of this game for the world-model check."""

        before = self._previous
        if before is None or not before.grid or not after.grid:
            return
        move = dict(self._pending_move)
        if (
            move.get("name") == "RESET"
        ):  # a reset starts a new chain, nothing to predict
            self._wm_records.append(
                {
                    "kind": "start",
                    "levels": after.levels_completed,
                    "frame": {"base": encode_rows(grid_rows(after.grid))},
                }
            )
            return
        if not self._wm_records:
            self._wm_records.append(
                {
                    "kind": "start",
                    "levels": before.levels_completed,
                    "frame": {"base": encode_rows(grid_rows(before.grid))},
                }
            )
        self._wm_records.append(
            {
                "kind": "step",
                "step": self._steps - 1,
                "moves": [move],
                "state": after.state,
                "levels": after.levels_completed,
                "frame": {"key": encode_rows(grid_rows(after.grid))},
            }
        )

    def _submit_world_model(self, args: list[object]) -> None:
        if len(args) != 1 or not isinstance(args[0], str):
            return
        source = args[0]
        self._wm_seq += 1
        if len(source) > MAX_WORLD_MODEL_SOURCE:
            self._wm_message = f"World model rejected: source longer than {MAX_WORLD_MODEL_SOURCE} characters."
            return
        try:
            from .model_tool import submit_world_model
            from .world_model import RevisionLog

            if self._wm_log is None:
                self._wm_log = RevisionLog(self._game_id)
            result = submit_world_model(self._wm_log, source, list(self._wm_records))
        except Exception as exc:  # noqa: BLE001 - a broken checker must not cost the move
            self._wm_message = f"World model check failed: {type(exc).__name__}"
            return
        if result.accepted:
            old, self._wm_result = self._wm_result, result
            if old is not None:
                old.close()
        else:
            result.close()
        self._wm_message = result.message()[:MAX_WORLD_MODEL_MESSAGE]

    def _check_plan(self, observation: Observation, gained: int) -> None:
        """Compare the real frame with the one the planner predicted for the move just settled."""

        ex = self._executor
        if gained > 0:  # the level changed: the frame legitimately differs
            self._executor = None
            return
        status = ex.observe(observation.grid)
        if status.state == "continue":
            return
        self._executor = None
        revision = self._wm_result.spec.revision if self._wm_result else 0
        claim = f"World model r{revision} predicts this game"
        step = max(self._steps - 1, 0)
        if status.state == "done":
            hid = self._model_ledger.propose(claim, "mechanic", step)
            self._model_ledger.observe(hid, step, "support", "a whole plan matched")
            return
        report = status.report
        assert report is not None and self._previous is not None
        before = self._previous.grid
        predicted = describe_change(before, ex.frames[report.step])
        actual = describe_change(before, observation.grid)
        text = (
            f"PLAN ABORTED: your world model r{revision} predicted {predicted[:150]} but "
            f"{actual[:150]} happened at step {step} ({self._label_of(self._pending_move)}; plan "
            f"step {report.step}). {report.text()}. Fix the model so it still replays every move, "
            "including this one, and resubmit it."
        )
        self.last_mismatch = {
            "step": step,
            "plan_step": report.step,
            "revision": revision,
            "cells": report.cells_differing,
            "bbox": list(report.bbox) if report.bbox else None,
        }
        hid = self._model_ledger.propose(claim, "mechanic", step)
        self._model_ledger.observe(hid, step, "counter", text[:200])
        self._wm_message = f"{self._wm_message} {text}".strip()

    @staticmethod
    def _label_of(move: dict[str, object]) -> str:
        coords = f" at x={move['x']},y={move['y']}" if "x" in move else ""
        return f"{move['name']}{coords}"

    def _make_plan(self, observation: Observation) -> Any:
        """Search the accepted model from the current frame; None when it has no usable goal test."""

        from .planner import plan
        from .world_model import Move, transitions_from_trace

        model = self._wm_result.model
        goal = self._goal_predicate
        if goal is None and not getattr(model, "has_is_goal", False):
            return None
        moves = [
            Move(f"ACTION{a}") for a in observation.available_actions if a not in (0, 6)
        ]
        history = [
            (t.before, t.move)
            for t in transitions_from_trace(self._wm_records)
            if t.levels_before == observation.levels_completed
        ]
        return plan(model, observation.grid, moves, goal, history=history)

    def _planned_decision(self, observation: Observation) -> ActionDecision | None:
        """Next move of the planner's plan, planning first when a new verified model is available."""

        self._settle_pending(observation)
        safe = (
            observation.state == "NOT_FINISHED"
            and observation.levels_completed == self._exec_levels
        )
        ex = self._executor
        if ex is not None and (not safe or ex.finished):
            ex = self._executor = None
        if ex is None:
            if self._wm_result is None or observation.state != "NOT_FINISHED":
                return None
            key = (self._wm_result.spec.revision, observation.levels_completed)
            if (
                key == self._plan_key
            ):  # planned already (or aborted): wait for a revision
                return None
            self._plan_key = key
            try:
                result = self._make_plan(observation)
            except Exception:  # noqa: BLE001 - a broken planner must not cost the move
                return None
            if result is None:
                return None
            if not result.found or not result.moves:
                self._wm_message = (
                    f"{self._wm_message} Planner found no plan with your model "
                    f"({result.reason}).".strip()
                )
                return None
            from .planner import PlanExecutor

            ex = self._executor = PlanExecutor(result)
            self._exec_levels = observation.levels_completed
            self._plan_full = [self._decision_of(m) for m in result.moves]
            self._plan_pos = 0
        move = ex.next_move()
        assert move is not None
        decision = self._decision_of(move)
        if decision.action_id not in observation.available_actions:
            self._executor = None
            return None
        self._previous = observation
        self._note_pending(decision)
        self._pending_planned = True
        self._plan_pos += 1
        self.last_turn = {
            "src": "planned",
            "move": self._move(decision),
            "plan": [self._move(d) for d in self._plan_full],
            "plan_idx": self._plan_pos,
            "nudge": False,
        }
        return decision

    @staticmethod
    def _decision_of(move: Any) -> ActionDecision:
        data = {"x": move.x, "y": move.y} if move.x is not None else {}
        return ActionDecision(name=move.name, data=data, source="planned")

    def close(self) -> None:
        """Release the sandboxed world model (game end); safe to call more than once."""

        result, self._wm_result = self._wm_result, None
        if result is not None:
            result.close()

    def _plan_results(self) -> str:
        """Move-by-move effects of the last multi-move plan (a single move is already in the line above)."""

        if len(self._since_call) < 2:
            return ""
        lines = "\n".join(
            f"  {i}) {text}" for i, text in enumerate(self._since_call, 1)
        )
        return f"Results of your last plan, move by move:\n{lines}\n"

    @staticmethod
    def _label(decision: ActionDecision) -> str:
        coords = (
            f" at x={decision.data['x']},y={decision.data['y']}"
            if decision.data
            else ""
        )
        return f"{decision.name}{coords}"

    def _is_dead(
        self, decision: ActionDecision, grid: tuple[tuple[int, ...], ...]
    ) -> bool:
        """True when this exact move is known to do nothing here (see DEAD_ON_SAME_BOARD)."""

        label = self._label(decision)
        if self._dead[(label, hash(grid))] >= DEAD_ON_SAME_BOARD:
            return True
        tries, changed = self._moves.get(label, (0, 0))
        return bool(decision.data) and tries >= DEAD_CLICK_TRIES and changed == 0

    def _dead_moves(self, observation: Observation) -> list[str]:
        h = hash(observation.grid)
        labels = {
            k for (k, g), n in self._dead.items() if g == h and n >= DEAD_ON_SAME_BOARD
        }
        labels |= {
            k
            for k, (tries, changed) in self._moves.items()
            if " at x=" in k and tries >= DEAD_CLICK_TRIES and changed == 0
        }
        return sorted(labels)[:8]

    def _untried_line(self, observation: Observation) -> str:
        if not self._effects:
            return ""
        untried = [
            f"ACTION{a}"
            for a in observation.available_actions
            if a != 0 and f"ACTION{a}" not in self._effects
        ]
        text = f"Not tried yet in this game: {', '.join(untried)}.\n" if untried else ""
        if 6 in observation.available_actions and self._clicks:
            fresh = untried_clicks(observation.grid, self._clicks, limit=8)
            if fresh:
                listed = "; ".join(f"colour {r.color} at {r.center}" for r in fresh)
                text += f"Objects not clicked yet (x,y centres): {listed}.\n"
        dead = self._dead_moves(observation)
        if dead:
            text += f"These moves did nothing here, do not repeat them: {'; '.join(dead)}.\n"
        return text

    def _alternative(
        self, observation: Observation, avoid: ActionDecision
    ) -> ActionDecision | None:
        """A move to play instead of a dead one: an untried object to click, else an untried or live action."""

        available = observation.available_actions
        if avoid.data and 6 in available:
            for r in untried_clicks(observation.grid, self._clicks, limit=8):
                x, y = r.center
                cand = ActionDecision(
                    name="ACTION6", data={"x": x, "y": y}, source="loop-breaker"
                )
                if not self._is_dead(cand, observation.grid):
                    return cand
        plain = [a for a in available if a not in (0, 6)]
        for a in sorted(plain, key=lambda a: f"ACTION{a}" in self._effects):
            cand = ActionDecision(name=f"ACTION{a}", source="loop-breaker")
            if not self._is_dead(cand, observation.grid):
                return cand
        return None

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

        if self._world_model and not self._queue:
            planned = self._planned_decision(observation)
            if planned is not None:
                return planned
        if not self._queue:
            return None
        unsafe = (
            observation.state != "NOT_FINISHED"
            or observation.levels_completed != self._plan_levels
            or self._noop_streak >= 2
        )
        self._settle_pending(observation)
        if unsafe or self._noop_streak >= 2 or not self._queue:
            self._queue.clear()
            return None
        decision = self._queue.pop(0)
        if decision.action_id not in observation.available_actions or self._is_dead(
            decision, observation.grid
        ):
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
        self._pending_planned = False
        self._steps += 1
        self._pending_move = self._move(decision)
        if decision.data:
            self._clicks.append((decision.data["x"], decision.data["y"]))
        self._pending = (self._label(decision), decision.name)

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
        self._executor = None
        self._last_available = observation.available_actions
        self._game_id = observation.game_id
        if self._events:
            self.last_event = self._event_for(observation)
            self._last_call_levels = observation.levels_completed
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
            messages=(
                ChatMessage(role="system", content=self._system_prompt),
                user_message,
            ),
            tools=(self._tool,),
        )
        response = self._client.complete(request)
        self.last_response = response
        self._previous = observation
        calls = self._calls_to_use(response, observation.available_actions)
        self._ignored_calls = len(response.tool_calls) - len(calls)
        if self._tools:
            self._apply_tool_calls(response, calls)
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
        override: str | None = None
        if self._is_dead(decisions[0], observation.grid):
            swap = self._alternative(observation, decisions[0])
            if swap is not None:
                override = f"{self._label(decisions[0])} -> {self._label(swap)}"
                decisions = [swap]
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
        if override:
            self.last_turn["override"] = override
        if self._events and self.last_event:
            self.last_turn["event"] = self.last_event
        self._surprise = ""
        if (
            self._wm_shown_seq == self._wm_seq
        ):  # shown once, in the prompt after the submission
            self._wm_message = ""
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
