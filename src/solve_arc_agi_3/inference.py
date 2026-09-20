"""Provider-neutral OpenAI-compatible chat-completions boundary."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from .baseline_config import InferenceConfig

JsonObject = dict[str, Any]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextContentPart(WireModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(WireModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContentPart(WireModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[TextContentPart | ImageContentPart, Field(discriminator="type")]


class ChatMessage(WireModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | tuple[ContentPart, ...]


class ToolDefinition(WireModel):
    name: str
    description: str
    parameters: JsonObject


class InferenceRequest(WireModel):
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    tools: tuple[ToolDefinition, ...] = ()


class FunctionCall(WireModel):
    name: str
    arguments: str


class ToolCall(WireModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class TokenUsage(WireModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class InferenceResponse(WireModel):
    response_id: str | None = None
    model: str | None = None
    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_seconds: float = Field(ge=0)


class InferenceClient(Protocol):
    def complete(self, request: InferenceRequest) -> InferenceResponse: ...


class JsonTransport(Protocol):
    def request_json(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> JsonObject: ...


class TransportError(RuntimeError):
    """The OpenAI-compatible endpoint could not return a valid JSON response."""


class UrllibJsonTransport:
    """Small standard-library HTTP transport used by all real providers."""

    def request_json(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> JsonObject:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (OSError, urllib.error.URLError) as exc:
            raise TransportError(f"request to {url} failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TransportError(f"endpoint {url} returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise TransportError(f"endpoint {url} returned a non-object JSON value")
        return cast(JsonObject, parsed)


class OpenAICompatibleClient:
    """One client for local vLLM, RunPod, OpenRouter, and an injected fake."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        transport: JsonTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._transport = transport or UrllibJsonTransport()
        self._monotonic = monotonic

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env is not None:
            api_key = os.environ.get(self.config.api_key_env, "").strip()
            if not api_key:
                raise ValueError(
                    f"required API key environment variable "
                    f"{self.config.api_key_env} is not set"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request_body(self, request: InferenceRequest) -> JsonObject:
        sampling = self.config.sampling
        body: JsonObject = {
            "model": self.config.model,
            "messages": [
                message.model_dump(mode="json") for message in request.messages
            ],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "seed": sampling.seed,
            "max_tokens": self.config.max_output_tokens,
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": tool.model_dump(),
                }
                for tool in request.tools
            ]
            body["tool_choice"] = "auto"
        if self.config.provider.value == "openrouter" and sampling.thinking:
            body["reasoning"] = {"enabled": True}
        return body

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        started = self._monotonic()
        raw = self._transport.request_json(
            method="POST",
            url=f"{self.config.base_url}/chat/completions",
            headers=self._headers(),
            payload=self._request_body(request),
            timeout_seconds=self.config.request_timeout_seconds,
        )
        elapsed = max(0.0, self._monotonic() - started)
        return _parse_chat_completion(raw, elapsed_seconds=elapsed)


def _parse_chat_completion(
    raw: JsonObject, *, elapsed_seconds: float
) -> InferenceResponse:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise TransportError("chat completion response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TransportError("chat completion choice has no message")

    raw_tool_calls = message.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        raise TransportError("message.tool_calls is not a list")
    try:
        tool_calls = tuple(ToolCall.model_validate(item) for item in raw_tool_calls)
    except (TypeError, ValueError) as exc:
        raise TransportError("message.tool_calls is invalid") from exc

    raw_usage = raw.get("usage") or {}
    if not isinstance(raw_usage, dict):
        raise TransportError("usage is not an object")
    usage = TokenUsage(
        prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
        completion_tokens=int(raw_usage.get("completion_tokens") or 0),
        total_tokens=int(raw_usage.get("total_tokens") or 0),
    )
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise TransportError("message.content is not text")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return InferenceResponse(
        response_id=str(raw["id"]) if raw.get("id") is not None else None,
        model=str(raw["model"]) if raw.get("model") is not None else None,
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        usage=usage,
        elapsed_seconds=elapsed_seconds,
    )


class ModelsProbe:
    """Readiness probe for an OpenAI-compatible ``/v1/models`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        transport: JsonTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibJsonTransport()

    def is_ready(self, expected_model: str) -> bool:
        try:
            raw = self._transport.request_json(
                method="GET",
                url=f"{self._base_url}/models",
                headers={"Accept": "application/json"},
                payload=None,
                timeout_seconds=self._timeout_seconds,
            )
        except TransportError:
            return False
        data = raw.get("data")
        if not isinstance(data, list):
            return False
        return any(
            isinstance(item, dict) and item.get("id") == expected_model for item in data
        )
