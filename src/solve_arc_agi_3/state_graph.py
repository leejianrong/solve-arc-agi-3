"""Directed graph of boards seen and moves tried, for loop detection and exploration hints.

Nodes are any hashable board identity (the agent uses ``hash(observation.grid)``); edges are labelled by
the move that was taken. Original code; no I/O.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Sequence


class StateGraph:
    def __init__(self) -> None:
        self._edges: dict[Hashable, dict[str, Hashable]] = {}
        self._visits: dict[Hashable, int] = {}

    def add(self, state_hash: Hashable, move_label: str, next_hash: Hashable) -> None:
        """Record that `move_label` taken on `state_hash` led to `next_hash`.

        Each call counts one arrival at `next_hash`. Re-taking a move overwrites its target.
        """

        self._edges.setdefault(state_hash, {})[move_label] = next_hash
        self._visits.setdefault(state_hash, 0)
        self._visits[next_hash] = self._visits.get(next_hash, 0) + 1

    def visit(self, state_hash: Hashable) -> None:
        """Count a sighting without an edge (the first board of a level)."""

        self._visits[state_hash] = self._visits.get(state_hash, 0) + 1

    def visits(self, h: Hashable) -> int:
        return self._visits.get(h, 0)

    def tried(self, h: Hashable) -> dict[str, Hashable]:
        return dict(self._edges.get(h, {}))

    def unexplored(self, h: Hashable, available_labels: Iterable[str]) -> list[str]:
        done = self._edges.get(h, {})
        return [label for label in available_labels if label not in done]

    @staticmethod
    def is_loop(window: Sequence[Hashable]) -> bool:
        """True when the recent board hashes revisit one (a repeat anywhere in the window)."""

        return len(set(window)) < len(window)

    def path(self, from_hash: Hashable, to_hash: Hashable) -> list[str] | None:
        """Shortest known list of move labels from one board to another, [] if equal, None if unknown."""

        if from_hash == to_hash:
            return []
        prev: dict[Hashable, tuple[Hashable, str]] = {}
        seen = {from_hash}
        queue = deque([from_hash])
        while queue:
            cur = queue.popleft()
            for label, nxt in self._edges.get(cur, {}).items():
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = (cur, label)
                if nxt == to_hash:
                    out: list[str] = []
                    node = nxt
                    while node != from_hash:
                        node, lab = prev[node]
                        out.append(lab)
                    return out[::-1]
                queue.append(nxt)
        return None

    def summary(
        self, h: Hashable, available_labels: Sequence[str], max_chars: int = 200
    ) -> str:
        """One line: how often this board was seen, untried moves, and where tried moves went."""

        n = self.visits(h)
        parts = [f"This board seen {n} time{'s' if n != 1 else ''}."]
        left = self.unexplored(h, available_labels)
        parts.append(f"Untried from here: {', '.join(left) if left else 'none'}.")
        exits = []
        for label, nxt in self._edges.get(h, {}).items():
            if nxt == h:
                kind = "no change"
            elif self.visits(nxt) > 1:
                kind = "seen before"
            else:
                kind = "new"
            exits.append(f"{label}->{kind}")
        if exits:
            parts.append(f"Known exits: {', '.join(exits)}.")
        text = " ".join(parts)
        if len(text) > max_chars:
            text = text[: max(max_chars - 3, 0)].rstrip() + "..."
        return text
