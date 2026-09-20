from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from solve_arc_agi_3.gpu_report import (
    AcceptanceEvidence,
    ControlProvenance,
    DeploymentProvenance,
    GpuBenchmarkReport,
    PerformanceMetrics,
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
