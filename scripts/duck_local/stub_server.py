#!/usr/bin/env python3
"""Scripted OpenAI-compatible stub: answers every chat completion with a `python` tool call that plays an action.

python3 stub_server.py PORT [LOGFILE]   # LOGFILE gets one JSON line per request (the request body)
"""

import itertools
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTIONS = itertools.cycle(
    ['action("UP")', 'action("RIGHT")', 'action("DOWN")', 'action("LEFT")']
)
LOG = sys.argv[2] if len(sys.argv) > 2 else None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # /v1/models and health probes
        self._send({"object": "list", "data": [{"id": "flashnext", "object": "model"}]})

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
        )
        if LOG:
            with open(LOG, "a") as fh:
                fh.write(json.dumps(body) + "\n")
        call = {
            "id": "call_stub",
            "type": "function",
            "function": {
                "name": "python",
                "arguments": json.dumps({"code": next(ACTIONS)}),
            },
        }
        msg = {"role": "assistant", "content": "", "tool_calls": [call]}
        self._send(
            {
                "id": "stub",
                "object": "chat.completion",
                "model": body.get("model", "flashnext"),
                "choices": [
                    {"index": 0, "message": msg, "finish_reason": "tool_calls"}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
