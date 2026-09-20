"""Pure helpers that turn measured RunPod evidence into a GpuBenchmarkReport.

Gathering the raw numbers (nvidia-smi polling, HTTP timing, log files) needs a
real GPU and isn't unit-testable. This module is the seam: everything here is
a pure function over already-collected strings/numbers, so the report-assembly
logic is tested on fixtures while the ops glue (`scripts/runpod_benchmark.py`)
stays a thin, untested orchestrator -- the same split `baseline_cli.py` uses.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from .gpu_report import (
    AcceptanceEvidence,
    ControlProvenance,
    DeploymentProvenance,
    GpuBenchmarkReport,
    PerformanceMetrics,
)

_DOWNLOAD_INDICATORS = re.compile(
    r"Downloading|huggingface\.co|Fetching \d+ files", re.IGNORECASE
)


def count_offline_download_attempts(log_text: str) -> int:
    """Count log lines indicating a network fetch attempt during startup."""

    return sum(1 for line in log_text.splitlines() if _DOWNLOAD_INDICATORS.search(line))


def _find_last_event(lifecycle_jsonl_text: str, event_name: str) -> dict[str, Any]:
    for line in reversed(lifecycle_jsonl_text.splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == event_name:
            return dict(event)
    raise ValueError(f"lifecycle log contains no {event_name!r} event")


def parse_ready_elapsed_seconds(lifecycle_jsonl_text: str) -> float:
    """Extract the vLLM lifecycle's own recorded startup-to-ready duration."""

    return float(_find_last_event(lifecycle_jsonl_text, "ready")["elapsed_seconds"])


def parse_stopped_exit_code(lifecycle_jsonl_text: str) -> int:
    """Extract the vLLM process's own recorded exit code after shutdown."""

    return int(_find_last_event(lifecycle_jsonl_text, "stopped")["exit_code"])


def compute_prefill_decode_throughput(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    time_to_first_token_seconds: float,
    total_seconds: float,
) -> tuple[float, float]:
    """Approximate prefill/decode throughput from one streamed request.

    Prefill throughput is prompt tokens over time-to-first-token; decode
    throughput is completion tokens over the remaining generation time. Both
    denominators are floored so a pathologically fast response cannot divide
    by (near) zero.
    """

    decode_seconds = max(total_seconds - time_to_first_token_seconds, 1e-6)
    ttft = max(time_to_first_token_seconds, 1e-6)
    return prompt_tokens / ttft, completion_tokens / decode_seconds


def build_gpu_benchmark_report(
    *,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    deployment: DeploymentProvenance,
    control: ControlProvenance,
    metrics: PerformanceMetrics,
    acceptance: AcceptanceEvidence,
) -> GpuBenchmarkReport:
    """Assemble and strictly validate the acceptance report from evidence."""

    return GpuBenchmarkReport(
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        deployment=deployment,
        control=control,
        metrics=metrics,
        acceptance=acceptance,
    )
