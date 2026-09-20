"""Run-scoped append-only trace schema for baseline episodes."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / "trace.jsonl"
        self._sequence = 0
        self._now = now or (lambda: datetime.now(UTC))

    def append(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "sequence": self._sequence,
            "timestamp": self._now().isoformat(),
            "event": event,
            "data": data,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        self._sequence += 1
