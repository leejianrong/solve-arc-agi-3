"""Pure helpers that turn N concurrent real agent episodes into one
ConcurrencyBenchmarkReport for a single point in the ARC-40 sweep.

Firing the episodes needs a real vLLM server and isn't unit-testable; this
module is the seam, same split as `runpod_report.py`: everything here is a
pure function over already-collected trace text and numbers, so report
assembly is tested on fixtures while `scripts/runpod_concurrency_sweep.py`
stays a thin, untested orchestrator that reuses the real agent path
(`BaselineAgentCore` / `OpenAICompatibleClient` via `smoke.run_provider_smoke`)
for every measured episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from .gpu_report import (
    ConcurrencyBenchmarkReport,
    ConcurrencyMetrics,
    ControlProvenance,
    DeploymentProvenance,
)


@dataclass(frozen=True)
class EpisodeResult:
    """One concurrent agent episode's outcome, read back from its own trace."""

    success: bool
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int


def parse_episode_result_from_trace_text(trace_jsonl_text: str) -> EpisodeResult:
    """Extract one smoke episode's outcome from its `trace.jsonl` contents.

    Reads the same `model_response` and `terminal` events every smoke run
    already writes (`smoke.py`'s `_run_smoke_episode`) -- no trace schema
    change needed to measure concurrency.
    """

    model_response: dict[str, object] | None = None
    terminal: dict[str, object] | None = None
    for line in trace_jsonl_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "model_response":
            model_response = record["data"]
        elif record.get("event") == "terminal":
            terminal = record["data"]
    if model_response is None:
        raise ValueError("trace has no model_response event")
    if terminal is None:
        raise ValueError("trace has no terminal event")

    usage = model_response.get("usage")
    if not isinstance(usage, dict):
        raise TypeError("model_response event has no usage")
    elapsed_seconds = model_response.get("elapsed_seconds")
    if not isinstance(elapsed_seconds, int | float):
        raise TypeError("model_response event has no numeric elapsed_seconds")
    return EpisodeResult(
        success=terminal.get("state") == "WIN",
        elapsed_seconds=float(elapsed_seconds),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
    )


def aggregate_concurrency_metrics(
    results: tuple[EpisodeResult, ...],
    *,
    concurrency: int,
    wall_clock_seconds: float,
    peak_vram_bytes: int,
    stable_context_tokens: int,
) -> ConcurrencyMetrics:
    """Reduce N concurrent episode outcomes to one concurrency level's metrics."""

    if not results:
        raise ValueError("aggregate_concurrency_metrics requires at least one result")

    total_prompt_tokens = sum(result.prompt_tokens for result in results)
    total_completion_tokens = sum(result.completion_tokens for result in results)
    return ConcurrencyMetrics(
        concurrency=concurrency,
        total_requests=len(results),
        successful_requests=sum(1 for result in results if result.success),
        aggregate_prompt_tokens_per_second=total_prompt_tokens / wall_clock_seconds,
        aggregate_completion_tokens_per_second=(
            total_completion_tokens / wall_clock_seconds
        ),
        request_latency_seconds=tuple(result.elapsed_seconds for result in results),
        peak_vram_bytes=peak_vram_bytes,
        stable_context_tokens=stable_context_tokens,
        wall_clock_seconds=wall_clock_seconds,
    )


def build_concurrency_benchmark_report(
    *,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    deployment: DeploymentProvenance,
    control: ControlProvenance,
    metrics: ConcurrencyMetrics,
) -> ConcurrencyBenchmarkReport:
    """Assemble and strictly validate one concurrency level's sweep report."""

    return ConcurrencyBenchmarkReport(
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        deployment=deployment,
        control=control,
        metrics=metrics,
    )
