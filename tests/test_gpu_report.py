from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from solve_arc_agi_3.gpu_report import (
    AcceptanceEvidence,
    ConcurrencyBenchmarkReport,
    ConcurrencyMetrics,
    ControlProvenance,
    DeploymentProvenance,
    GpuBenchmarkReport,
    PerformanceMetrics,
    write_concurrency_benchmark_report,
    write_gpu_benchmark_report,
)


def _report(*, concurrency: int = 1) -> GpuBenchmarkReport:
    started = datetime(2026, 9, 20, tzinfo=UTC)
    return GpuBenchmarkReport(
        run_id="runpod-acceptance-example",
        started_at=started,
        ended_at=started + timedelta(minutes=10),
        deployment=DeploymentProvenance(
            container_image="example/vllm@sha256:" + "a" * 64,
            gpu_model="NVIDIA RTX PRO 6000 Blackwell",
            gpu_count=1,
            gpu_memory_bytes=96_000_000_000,
            driver_version="example",
            cuda_version="example",
            python_version="3.12.12",
            vllm_version="0.19.0",
            torch_version="2.10.0",
        ),
        control=ControlProvenance(
            control_id="duck-clean-room-v1",
            manifest_sha256="b" * 64,
            model_repository="vrfai/Qwen3.6-27B-FP8",
            model_revision="076636763143ca2b7cf0d66e16a09c2a0a689dfa",
            wheelhouse_dataset="driessmit1/arc3-vllm-h100-wheelhouse-v3@1",
            launch_command=("python", "-m", "vllm.entrypoints.openai.api_server"),
        ),
        metrics=PerformanceMetrics(
            model_load_seconds=120,
            peak_vram_bytes=40_000_000_000,
            time_to_first_token_seconds=0.5,
            prefill_tokens_per_second=1000,
            decode_tokens_per_second=40,
            stable_context_tokens=32768,
            request_latency_seconds=(1.0,),
            concurrency=concurrency,
        ),
        acceptance=AcceptanceEvidence(
            full_asset_preflight_passed=True,
            real_smoke_request_passed=True,
            offline_download_attempts=0,
            server_exit_code=0,
        ),
    )


def test_gpu_report_writes_all_required_provenance(tmp_path: Path) -> None:
    path = tmp_path / "report.json"

    write_gpu_benchmark_report(_report(), path)

    text = path.read_text()
    assert '"model_load_seconds": 120.0' in text
    assert '"model_revision": "076636763143ca2b7cf0d66e16a09c2a0a689dfa"' in text
    assert not path.with_suffix(".json.tmp").exists()


def test_first_acceptance_report_rejects_concurrency_sweep() -> None:
    with pytest.raises(ValidationError, match="concurrency 1"):
        _report(concurrency=4)


def _concurrency_metrics(*, concurrency: int = 4) -> ConcurrencyMetrics:
    return ConcurrencyMetrics(
        concurrency=concurrency,
        total_requests=concurrency,
        successful_requests=concurrency,
        aggregate_prompt_tokens_per_second=4000.0,
        aggregate_completion_tokens_per_second=120.0,
        request_latency_seconds=tuple(1.0 for _ in range(concurrency)),
        peak_vram_bytes=60_000_000_000,
        stable_context_tokens=8000,
        wall_clock_seconds=5.0,
    )


def _concurrency_report(*, concurrency: int = 4) -> ConcurrencyBenchmarkReport:
    started = datetime(2026, 9, 21, tzinfo=UTC)
    return ConcurrencyBenchmarkReport(
        run_id="sweep-concurrency-4",
        started_at=started,
        ended_at=started + timedelta(minutes=5),
        deployment=DeploymentProvenance(
            container_image="example/vllm@sha256:" + "a" * 64,
            gpu_model="NVIDIA RTX PRO 6000 Blackwell",
            gpu_count=1,
            gpu_memory_bytes=96_000_000_000,
            driver_version="example",
            cuda_version="example",
            python_version="3.12.12",
            vllm_version="0.19.0",
            torch_version="2.10.0",
        ),
        control=ControlProvenance(
            control_id="duck-clean-room-v1",
            manifest_sha256="b" * 64,
            model_repository="vrfai/Qwen3.6-27B-FP8",
            model_revision="076636763143ca2b7cf0d66e16a09c2a0a689dfa",
            wheelhouse_dataset="driessmit1/arc3-vllm-h100-wheelhouse-v3@1",
            launch_command=("python", "-m", "vllm.entrypoints.openai.api_server"),
        ),
        metrics=_concurrency_metrics(concurrency=concurrency),
    )


def test_concurrency_benchmark_report_allows_concurrency_above_one() -> None:
    report = _concurrency_report(concurrency=16)

    assert report.metrics.concurrency == 16


def test_concurrency_metrics_rejects_more_successes_than_requests() -> None:
    with pytest.raises(ValidationError, match="successful_requests"):
        ConcurrencyMetrics(
            concurrency=4,
            total_requests=4,
            successful_requests=5,
            aggregate_prompt_tokens_per_second=1000.0,
            aggregate_completion_tokens_per_second=50.0,
            request_latency_seconds=(1.0, 1.0, 1.0, 1.0),
            peak_vram_bytes=1,
            stable_context_tokens=1,
            wall_clock_seconds=1.0,
        )


def test_concurrency_metrics_rejects_latency_count_mismatch() -> None:
    with pytest.raises(ValidationError, match="request_latency_seconds"):
        ConcurrencyMetrics(
            concurrency=4,
            total_requests=4,
            successful_requests=4,
            aggregate_prompt_tokens_per_second=1000.0,
            aggregate_completion_tokens_per_second=50.0,
            request_latency_seconds=(1.0, 1.0),
            peak_vram_bytes=1,
            stable_context_tokens=1,
            wall_clock_seconds=1.0,
        )


def test_write_concurrency_benchmark_report_persists_metrics(tmp_path: Path) -> None:
    path = tmp_path / "concurrency-4.json"

    write_concurrency_benchmark_report(_concurrency_report(), path)

    text = path.read_text()
    assert '"concurrency": 4' in text
    assert not path.with_suffix(".json.tmp").exists()
