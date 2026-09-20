from datetime import UTC, datetime, timedelta

import pytest

from solve_arc_agi_3.gpu_report import (
    AcceptanceEvidence,
    ControlProvenance,
    DeploymentProvenance,
    PerformanceMetrics,
)
from solve_arc_agi_3.runpod_report import (
    build_gpu_benchmark_report,
    compute_prefill_decode_throughput,
    count_offline_download_attempts,
    parse_ready_elapsed_seconds,
    parse_stopped_exit_code,
)


def test_count_offline_download_attempts_finds_known_indicators() -> None:
    log = (
        "INFO server started\n"
        "Downloading shards: 0%\n"
        "connecting to huggingface.co\n"
        "Fetching 12 files\n"
        "INFO server ready"
    )

    assert count_offline_download_attempts(log) == 3


def test_count_offline_download_attempts_is_zero_on_clean_log() -> None:
    log = "INFO server started\nINFO model loaded\nINFO server ready"

    assert count_offline_download_attempts(log) == 0


def test_parse_ready_elapsed_seconds_reads_last_ready_event() -> None:
    jsonl = (
        '{"event": "process_started", "monotonic_seconds": 1}\n'
        '{"event": "ready", "monotonic_seconds": 2, "elapsed_seconds": 87.5}'
    )

    assert parse_ready_elapsed_seconds(jsonl) == 87.5


def test_parse_ready_elapsed_seconds_requires_a_ready_event() -> None:
    with pytest.raises(ValueError, match="ready"):
        parse_ready_elapsed_seconds('{"event": "process_started"}')


def test_parse_stopped_exit_code_reads_last_stopped_event() -> None:
    jsonl = '{"event": "terminate_sent"}\n{"event": "stopped", "exit_code": 0}'

    assert parse_stopped_exit_code(jsonl) == 0


def test_compute_prefill_decode_throughput_splits_prefill_and_decode_time() -> None:
    prefill_tps, decode_tps = compute_prefill_decode_throughput(
        prompt_tokens=1000,
        completion_tokens=40,
        time_to_first_token_seconds=0.5,
        total_seconds=1.5,
    )

    assert prefill_tps == pytest.approx(2000.0)
    assert decode_tps == pytest.approx(40.0)


def test_compute_prefill_decode_throughput_floors_zero_durations() -> None:
    prefill_tps, decode_tps = compute_prefill_decode_throughput(
        prompt_tokens=10,
        completion_tokens=1,
        time_to_first_token_seconds=0.0,
        total_seconds=0.0,
    )

    assert prefill_tps > 0
    assert decode_tps > 0


def _deployment() -> DeploymentProvenance:
    return DeploymentProvenance(
        container_image="ubuntu@sha256:" + "a" * 64,
        gpu_model="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpu_count=1,
        gpu_memory_bytes=96_000_000_000,
        driver_version="570.0",
        cuda_version="12.8",
        python_version="3.12.5",
        vllm_version="0.19.0",
        torch_version="2.10.0",
    )


def _control() -> ControlProvenance:
    return ControlProvenance(
        control_id="duck-clean-room-v1",
        manifest_sha256="b" * 64,
        model_repository="vrfai/Qwen3.6-27B-FP8",
        model_revision="076636763143ca2b7cf0d66e16a09c2a0a689dfa",
        wheelhouse_dataset="driessmit1/arc3-vllm-h100-wheelhouse-v3",
        launch_command=("python", "-m", "vllm.entrypoints.openai.api_server"),
    )


def _metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        model_load_seconds=90.0,
        peak_vram_bytes=40_000_000_000,
        time_to_first_token_seconds=0.4,
        prefill_tokens_per_second=1500.0,
        decode_tokens_per_second=35.0,
        stable_context_tokens=32768,
        request_latency_seconds=(0.6, 0.7, 0.65),
        concurrency=1,
    )


def _acceptance() -> AcceptanceEvidence:
    return AcceptanceEvidence(
        full_asset_preflight_passed=True,
        real_smoke_request_passed=True,
        offline_download_attempts=0,
        server_exit_code=0,
    )


def test_build_gpu_benchmark_report_assembles_a_valid_report() -> None:
    started = datetime(2026, 9, 21, tzinfo=UTC)

    report = build_gpu_benchmark_report(
        run_id="runpod-acceptance-1",
        started_at=started,
        ended_at=started + timedelta(minutes=20),
        deployment=_deployment(),
        control=_control(),
        metrics=_metrics(),
        acceptance=_acceptance(),
    )

    assert report.schema_version == 1
    assert report.metrics.concurrency == 1
    assert report.acceptance.real_smoke_request_passed is True


def test_build_gpu_benchmark_report_rejects_ended_before_started() -> None:
    started = datetime(2026, 9, 21, tzinfo=UTC)

    with pytest.raises(Exception, match="ended_at"):
        build_gpu_benchmark_report(
            run_id="runpod-acceptance-1",
            started_at=started,
            ended_at=started - timedelta(minutes=1),
            deployment=_deployment(),
            control=_control(),
            metrics=_metrics(),
            acceptance=_acceptance(),
        )
