import json
from datetime import UTC, datetime, timedelta

import pytest

from solve_arc_agi_3.concurrency_report import (
    EpisodeResult,
    aggregate_concurrency_metrics,
    build_concurrency_benchmark_report,
    parse_episode_result_from_trace_text,
)
from solve_arc_agi_3.gpu_report import ControlProvenance, DeploymentProvenance


def _trace_line(event: str, data: dict[str, object]) -> str:
    return json.dumps({"event": event, "data": data})


def _winning_trace(
    *, elapsed_seconds: float, prompt_tokens: int, completion_tokens: int
) -> str:
    return "\n".join(
        [
            _trace_line("configuration", {}),
            _trace_line("observation", {}),
            _trace_line(
                "model_response",
                {
                    "elapsed_seconds": elapsed_seconds,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                },
            ),
            _trace_line("action_decision", {}),
            _trace_line("environment_transition", {}),
            _trace_line("terminal", {"state": "WIN", "levels_completed": 1}),
        ]
    )


def test_parse_episode_result_from_trace_text_reads_success_and_usage() -> None:
    trace_text = _winning_trace(
        elapsed_seconds=0.42, prompt_tokens=1200, completion_tokens=80
    )

    result = parse_episode_result_from_trace_text(trace_text)

    assert result == EpisodeResult(
        success=True,
        elapsed_seconds=0.42,
        prompt_tokens=1200,
        completion_tokens=80,
    )


def test_parse_episode_result_from_trace_text_reads_non_win_as_failure() -> None:
    trace_text = "\n".join(
        [
            _trace_line(
                "model_response",
                {
                    "elapsed_seconds": 1.0,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                },
            ),
            _trace_line("terminal", {"state": "NOT_FINISHED", "levels_completed": 0}),
        ]
    )

    result = parse_episode_result_from_trace_text(trace_text)

    assert result.success is False


def test_parse_episode_result_from_trace_text_requires_model_response() -> None:
    trace_text = _trace_line("terminal", {"state": "WIN", "levels_completed": 1})

    with pytest.raises(ValueError, match="model_response"):
        parse_episode_result_from_trace_text(trace_text)


def test_parse_episode_result_from_trace_text_requires_terminal() -> None:
    trace_text = _trace_line(
        "model_response",
        {"elapsed_seconds": 1.0, "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )

    with pytest.raises(ValueError, match="terminal"):
        parse_episode_result_from_trace_text(trace_text)


def _results(*, count: int, successes: int) -> tuple[EpisodeResult, ...]:
    return tuple(
        EpisodeResult(
            success=index < successes,
            elapsed_seconds=1.0 + index * 0.1,
            prompt_tokens=1000,
            completion_tokens=50,
        )
        for index in range(count)
    )


def test_aggregate_concurrency_metrics_computes_aggregate_throughput() -> None:
    results = _results(count=4, successes=4)

    metrics = aggregate_concurrency_metrics(
        results,
        concurrency=4,
        wall_clock_seconds=2.0,
        peak_vram_bytes=50_000_000_000,
        stable_context_tokens=8000,
    )

    assert metrics.concurrency == 4
    assert metrics.total_requests == 4
    assert metrics.successful_requests == 4
    assert metrics.aggregate_prompt_tokens_per_second == pytest.approx(2000.0)
    assert metrics.aggregate_completion_tokens_per_second == pytest.approx(100.0)
    assert len(metrics.request_latency_seconds) == 4


def test_aggregate_concurrency_metrics_reports_partial_failures() -> None:
    results = _results(count=4, successes=1)

    metrics = aggregate_concurrency_metrics(
        results,
        concurrency=4,
        wall_clock_seconds=2.0,
        peak_vram_bytes=50_000_000_000,
        stable_context_tokens=8000,
    )

    assert metrics.successful_requests == 1
    assert metrics.total_requests == 4


def test_aggregate_concurrency_metrics_requires_at_least_one_result() -> None:
    with pytest.raises(ValueError, match="at least one result"):
        aggregate_concurrency_metrics(
            (),
            concurrency=1,
            wall_clock_seconds=1.0,
            peak_vram_bytes=1,
            stable_context_tokens=1,
        )


def _deployment() -> DeploymentProvenance:
    return DeploymentProvenance(
        container_image="ubuntu@sha256:" + "a" * 64,
        gpu_model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpu_count=1,
        gpu_memory_bytes=96_000_000_000,
        driver_version="580.82.07",
        cuda_version="12.8",
        python_version="3.12.14",
        vllm_version="0.19.0",
        torch_version="2.10.0",
    )


def _control() -> ControlProvenance:
    return ControlProvenance(
        control_id="duck-clean-room-v1",
        manifest_sha256="b" * 64,
        model_repository="vrfai/Qwen3.6-27B-FP8",
        model_revision="076636763143ca2b7cf0d66e16a09c2a0a689dfa",
        wheelhouse_dataset="driessmit1/arc3-vllm-h100-wheelhouse-v3@1",
        launch_command=("python", "-m", "vllm.entrypoints.openai.api_server"),
    )


def test_build_concurrency_benchmark_report_allows_concurrency_above_one() -> None:
    started = datetime(2026, 9, 21, tzinfo=UTC)
    metrics = aggregate_concurrency_metrics(
        _results(count=4, successes=4),
        concurrency=4,
        wall_clock_seconds=2.0,
        peak_vram_bytes=50_000_000_000,
        stable_context_tokens=8000,
    )

    report = build_concurrency_benchmark_report(
        run_id="sweep-concurrency-4",
        started_at=started,
        ended_at=started + timedelta(minutes=5),
        deployment=_deployment(),
        control=_control(),
        metrics=metrics,
    )

    assert report.metrics.concurrency == 4


def test_build_concurrency_benchmark_report_rejects_ended_before_started() -> None:
    started = datetime(2026, 9, 21, tzinfo=UTC)
    metrics = aggregate_concurrency_metrics(
        _results(count=1, successes=1),
        concurrency=1,
        wall_clock_seconds=1.0,
        peak_vram_bytes=1,
        stable_context_tokens=1,
    )

    with pytest.raises(Exception, match="ended_at"):
        build_concurrency_benchmark_report(
            run_id="sweep-concurrency-1",
            started_at=started,
            ended_at=started - timedelta(minutes=1),
            deployment=_deployment(),
            control=_control(),
            metrics=metrics,
        )
