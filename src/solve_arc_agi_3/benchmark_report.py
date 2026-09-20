"""Parse arc-agi-3-benchmarking's logs.log and a tokens/sec sample into the
phase-0 baseline summary (slice V1, docs/SLICES.md).

`arc-agi-3-benchmarking`'s `main.py` logs the literal line
"--- EXISTING SCORECARD REPORT ---" followed by `json.dumps(scorecard.model_dump())`
on SIGINT/completion (see benchmarking's own `main.py:cleanup`). The exact
Scorecard schema comes from the external `arc-agi` SDK and hasn't been
independently verified here, so field extraction is defensive: known field
names are tried under a few plausible aliases, and anything missing surfaces
as `None` in the report rather than raising. Treat the parsed fields as
best-effort until checked against one real scorecard.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SCORECARD_MARKER = "--- EXISTING SCORECARD REPORT ---"

# A scorecard's per-game entries have been seen (informally, in ARC Prize's
# own docs) under both "rhae" and "score" naming; try both rather than
# assume one.
_RHAE_KEYS = ("rhae", "score")
_ACTIONS_KEYS = ("actions", "action_count", "total_actions")
_GAME_ID_KEYS = ("game_id", "game", "id")


@dataclass(frozen=True)
class GameResult:
    game_id: str | None
    rhae: float | None
    actions: int | None


@dataclass(frozen=True)
class PhaseZeroReport:
    games: list[GameResult]
    tokens_per_second: float | None

    @property
    def mean_rhae(self) -> float | None:
        scored = [g.rhae for g in self.games if g.rhae is not None]
        if not scored:
            return None
        return sum(scored) / len(scored)


def _first_present(entry: dict[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return None


def _extract_game_entries(scorecard: dict[str, object]) -> list[dict[str, object]]:
    """Scorecards have been seen with per-game results under "cards", "games",
    or as the scorecard's own top level for a single-game run. Try each."""
    for key in ("cards", "games", "results"):
        value = scorecard.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            return [v for v in value.values() if isinstance(v, dict)]
    # Single-game scorecard: the top-level dict itself is the one entry.
    if any(k in scorecard for k in _RHAE_KEYS + _GAME_ID_KEYS):
        return [scorecard]
    return []


def parse_scorecards_from_log(log_text: str) -> list[dict[str, object]]:
    """Extract every JSON scorecard block logged after SCORECARD_MARKER."""
    scorecards: list[dict[str, object]] = []
    marker_positions = [
        m.start() for m in re.finditer(re.escape(SCORECARD_MARKER), log_text)
    ]
    for pos in marker_positions:
        after_marker = log_text[pos + len(SCORECARD_MARKER) :]
        # The JSON block is pretty-printed (indent=2) starting at the next "{".
        brace_start = after_marker.find("{")
        if brace_start == -1:
            continue
        decoder = json.JSONDecoder()
        try:
            obj, _end = decoder.raw_decode(after_marker[brace_start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            scorecards.append(obj)
    return scorecards


def summarize(
    *,
    log_text: str,
    tokens_per_second: float | None = None,
) -> PhaseZeroReport:
    games: list[GameResult] = []
    for scorecard in parse_scorecards_from_log(log_text):
        for entry in _extract_game_entries(scorecard):
            rhae = _first_present(entry, _RHAE_KEYS)
            actions = _first_present(entry, _ACTIONS_KEYS)
            game_id = _first_present(entry, _GAME_ID_KEYS)
            games.append(
                GameResult(
                    game_id=str(game_id) if game_id is not None else None,
                    rhae=float(rhae) if isinstance(rhae, (int, float)) else None,
                    actions=int(actions) if isinstance(actions, (int, float)) else None,
                )
            )
    return PhaseZeroReport(games=games, tokens_per_second=tokens_per_second)
