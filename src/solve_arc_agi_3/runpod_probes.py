"""Shared ops-glue probes for RunPod benchmark orchestrators.

Extracted from `scripts/runpod_benchmark.py` so `scripts/runpod_concurrency_sweep.py`
can reuse the exact same streaming-throughput and VRAM-polling code instead of
a second, divergent implementation. Not unit tested -- every function here
needs a real GPU/process (`nvidia-smi`, an HTTP round trip to a live server);
the testable seam is `runpod_report.py` / `concurrency_report.py`, which take
already-collected strings and numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Self


def run(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def install_wheelhouse(wheelhouse_dir: Path) -> None:
    result = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            str(wheelhouse_dir),
            "-r",
            str(wheelhouse_dir / "requirements.lock"),
        ]
    )
    print(result.stdout[-4000:], flush=True)
    print(result.stderr[-4000:], file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"offline wheelhouse install failed with exit {result.returncode}"
        )


def nvidia_smi_query(fields: str) -> str:
    result = run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    return result.stdout.strip().splitlines()[0]


class VramPoller:
    """Polls nvidia-smi in a background thread and tracks peak memory used."""

    def __init__(self, *, interval_seconds: float = 2.0) -> None:
        self._interval_seconds = interval_seconds
        self._peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                used_mb = int(nvidia_smi_query("memory.used").split(",")[0].strip())
                self._peak_mb = max(self._peak_mb, used_mb)
            except (subprocess.SubprocessError, ValueError, IndexError, OSError):
                pass
            self._stop.wait(self._interval_seconds)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def peak_bytes(self) -> int:
        return self._peak_mb * 1024 * 1024


class ChatBenchmarkResult:
    def __init__(
        self,
        *,
        time_to_first_token_seconds: float,
        total_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.time_to_first_token_seconds = time_to_first_token_seconds
        self.total_seconds = total_seconds
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def post_chat(
    *, base_url: str, model: str, prompt: str, max_tokens: int, stream: bool
) -> ChatBenchmarkResult:
    """POST one raw chat-completion for benchmarking (bypasses the agent path;
    only used for timing/context-ceiling probes, never as real-smoke evidence)."""

    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        if not stream:
            raw = json.loads(response.read().decode("utf-8"))
            ended = time.monotonic()
            usage = raw.get("usage", {})
            return ChatBenchmarkResult(
                time_to_first_token_seconds=ended - started,
                total_seconds=ended - started,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            )
        ttft: float | None = None
        prompt_tokens = 0
        completion_tokens = 0
        chunk_count = 0
        content_chunk_count = 0
        first_chunk_at: float | None = None
        first_nonempty_delta: dict[str, object] | None = None
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]":
                break
            chunk_count += 1
            if first_chunk_at is None:
                first_chunk_at = time.monotonic() - started
            chunk = json.loads(payload)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta", {})
                if delta and first_nonempty_delta is None:
                    first_nonempty_delta = delta
                # With thinking enabled, reasoning tokens stream before the
                # final answer's content tokens -- the first *token* of
                # either kind is the real TTFT. Field name for the reasoning
                # channel is unconfirmed for this vLLM version (the
                # non-streaming parser in inference.py tries both
                # reasoning_content and reasoning), so check all three.
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                ):
                    content_chunk_count += 1
                    if ttft is None:
                        ttft = time.monotonic() - started
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                completion_tokens = int(
                    usage.get("completion_tokens", completion_tokens)
                )
    ended = time.monotonic()
    print(
        "PHASE=stream_debug "
        f"sse_chunks={chunk_count} content_chunks={content_chunk_count} "
        f"first_chunk_at={first_chunk_at} ttft={ttft} total={ended - started} "
        f"first_nonempty_delta={json.dumps(first_nonempty_delta)}",
        flush=True,
    )
    return ChatBenchmarkResult(
        time_to_first_token_seconds=ttft if ttft is not None else ended - started,
        total_seconds=ended - started,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def measure_stable_context_tokens(*, base_url: str, model: str) -> int:
    for target_words in (25_000, 12_000, 4_000):
        try:
            result = post_chat(
                base_url=base_url,
                model=model,
                prompt="context " * target_words,
                max_tokens=1,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 -- try a smaller probe next
            print(
                f"PHASE=context_probe_failed words={target_words} error={exc}",
                flush=True,
            )
            continue
        if result.prompt_tokens > 0:
            return result.prompt_tokens
    raise RuntimeError(
        "no context-length probe produced a measurable prompt token count"
    )
