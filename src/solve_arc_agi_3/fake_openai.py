"""Deterministic in-process OpenAI-compatible server used by local tests."""

from __future__ import annotations

from typing import Any

from .inference import JsonObject


class DeterministicFakeServer:
    """Serve fixed OpenAI JSON responses without opening a network socket."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.requests: list[JsonObject] = []
        self.network_requests = 0

    def request_json(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> JsonObject:
        del headers, timeout_seconds
        if method == "GET" and url.endswith("/models"):
            return {"object": "list", "data": [{"id": self.model}]}
        if method != "POST" or not url.endswith("/chat/completions"):
            raise AssertionError(f"unexpected fake-server request: {method} {url}")
        if payload is None:
            raise AssertionError("chat completion request has no JSON body")
        self.requests.append(payload)
        return {
            "id": f"fake-response-{len(self.requests)}",
            "object": "chat.completion",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "The first legal move reaches the goal.",
                        "tool_calls": [
                            {
                                "id": "fake-tool-1",
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": '{"code":"action([\\"ACTION1\\"])"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": 8,
                "total_tokens": 40,
            },
        }


def assert_openai_request_shape(request: dict[str, Any]) -> None:
    """Give fake-server failures a focused message in smoke tests."""

    required = {"model", "messages", "temperature", "top_p", "max_tokens"}
    missing = required.difference(request)
    if missing:
        raise AssertionError(f"OpenAI request is missing fields: {sorted(missing)}")
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        raise AssertionError("OpenAI request did not expose the Python tool")
