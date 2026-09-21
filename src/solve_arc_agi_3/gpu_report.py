"""Machine-readable evidence schema for the exact RunPod FP8 deployment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeploymentProvenance(ReportModel):
    provider: Literal["runpod"] = "runpod"
    pod_id_redacted: bool = True
    container_image: str = Field(pattern=r"^.+@sha256:[0-9a-f]{64}$")
    gpu_model: str
    gpu_count: int = Field(gt=0)
    gpu_memory_bytes: int = Field(gt=0)
    driver_version: str
    cuda_version: str
    python_version: str
    vllm_version: str
    torch_version: str


class ControlProvenance(ReportModel):
    control_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_repository: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    quantization: Literal["fp8"] = "fp8"
    wheelhouse_dataset: str
    launch_command: tuple[str, ...]


class PerformanceMetrics(ReportModel):
    model_load_seconds: float = Field(gt=0)
    peak_vram_bytes: int = Field(gt=0)
    time_to_first_token_seconds: float = Field(gt=0)
    prefill_tokens_per_second: float = Field(gt=0)
    decode_tokens_per_second: float = Field(gt=0)
    stable_context_tokens: int = Field(gt=0)
    request_latency_seconds: tuple[float, ...] = Field(min_length=1)
    concurrency: int = Field(gt=0)


class AcceptanceEvidence(ReportModel):
    full_asset_preflight_passed: bool
    real_smoke_request_passed: bool
    offline_download_attempts: int = Field(ge=0)
    server_exit_code: int


class GpuBenchmarkReport(ReportModel):
    schema_version: Literal[1] = 1
    run_id: str
    started_at: datetime
    ended_at: datetime
    deployment: DeploymentProvenance
    control: ControlProvenance
    metrics: PerformanceMetrics
    acceptance: AcceptanceEvidence

    @model_validator(mode="after")
    def validate_acceptance_run(self) -> GpuBenchmarkReport:
        if self.ended_at <= self.started_at:
            raise ValueError("ended_at must be after started_at")
        if self.metrics.concurrency != 1:
            raise ValueError("the first acceptance report must use concurrency 1")
        return self


def write_gpu_benchmark_report(report: GpuBenchmarkReport, path: Path) -> None:
    """Atomically persist exact deployment provenance and measured metrics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ConcurrencyMetrics(ReportModel):
    """Aggregate measurements from firing several concurrent agent episodes."""

    concurrency: int = Field(gt=0)
    total_requests: int = Field(gt=0)
    successful_requests: int = Field(ge=0)
    aggregate_prompt_tokens_per_second: float = Field(ge=0)
    aggregate_completion_tokens_per_second: float = Field(ge=0)
    request_latency_seconds: tuple[float, ...] = Field(min_length=1)
    peak_vram_bytes: int = Field(gt=0)
    stable_context_tokens: int = Field(gt=0)
    wall_clock_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_success_count(self) -> ConcurrencyMetrics:
        if self.successful_requests > self.total_requests:
            raise ValueError("successful_requests cannot exceed total_requests")
        if len(self.request_latency_seconds) != self.total_requests:
            raise ValueError("request_latency_seconds must have one entry per request")
        return self


class ConcurrencyBenchmarkReport(ReportModel):
    """One concurrency level's point in the ARC-40 GPU resource-envelope sweep.

    Deliberately not a `GpuBenchmarkReport`: that schema's validator pins
    concurrency to 1 for the first acceptance run, on purpose, and this report
    exists precisely to cover concurrency > 1.
    """

    schema_version: Literal[1] = 1
    run_id: str
    started_at: datetime
    ended_at: datetime
    deployment: DeploymentProvenance
    control: ControlProvenance
    metrics: ConcurrencyMetrics

    @model_validator(mode="after")
    def validate_sweep_run(self) -> ConcurrencyBenchmarkReport:
        if self.ended_at <= self.started_at:
            raise ValueError("ended_at must be after started_at")
        return self


def write_concurrency_benchmark_report(
    report: ConcurrencyBenchmarkReport, path: Path
) -> None:
    """Atomically persist one concurrency level's sweep report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_gpu_benchmark_schema(path: Path) -> None:
    """Persist a JSON Schema that a remote benchmark collector can validate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(GpuBenchmarkReport.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
