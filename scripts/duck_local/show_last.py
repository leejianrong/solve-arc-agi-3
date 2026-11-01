#!/usr/bin/env python3
"""Print only what a Haiku player needs from a parked Duck request: the newest tool result and user message.

docker exec semula-arc3-semula-1 semula show <ID> --json | python3 scripts/duck_local/show_last.py [max_chars]
"""

import json
import sys


def text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))  # type: ignore[union-attr]


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    messages = json.load(sys.stdin)["request"]["messages"]
    tail = [m for m in messages if m["role"] != "system"][-3:]
    for m in tail:
        body = text_of(m.get("content") or "")
        if m.get("tool_calls"):
            body += " TOOL_CALL " + json.dumps(m["tool_calls"])[:600]
        print(f"--- {m['role']}\n{body[:limit]}")


if __name__ == "__main__":
    main()
