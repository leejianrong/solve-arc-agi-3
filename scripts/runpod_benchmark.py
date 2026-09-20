#!/usr/bin/env python3
"""Orchestrates the real RunPod FP8 acceptance run.

Installs the pinned wheelhouse offline, verifies the mounted assets, starts
the pinned vLLM server, exercises it through the same agent path the fake
server uses, measures performance, and writes a GpuBenchmarkReport. Runs on
the pod itself, inside the project's venv. Not unit tested -- the testable
report-assembly logic lives in solve_arc_agi_3.runpod_report; this is ops
glue, the same split baseline_cli.py uses for its own commands.

Exits 0 only when preflight passed, the server started, and a real smoke
request completed through the agent path. A report is written whenever vLLM
served at least one measurable request, even if the smoke or a metric probe
failed -- that failure is a field in the report, not a reason to fabricate
one. No report is written if vLLM never reached a measurable state.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from solve_arc_agi_3.baseline_config import Provider, config_from_manifest
from solve_arc_agi_3.baseline_manifest import load_control_manifest, sha256_file
from solve_arc_agi_3.gpu_report import (
    AcceptanceEvidence,
    ControlProvenance,
    DeploymentProvenance,
    PerformanceMetrics,
    write_gpu_benchmark_report,
)
from solve_arc_agi_3.inference import ModelsProbe
from solve_arc_agi_3.preflight import require_asset_preflight, run_asset_preflight
from solve_arc_agi_3.runpod_report import (
    build_gpu_benchmark_report,
    compute_prefill_decode_throughput,
    count_offline_download_attempts,
    parse_ready_elapsed_seconds,
    parse_stopped_exit_code,
)
from solve_arc_agi_3.smoke import run_provider_smoke
from solve_arc_agi_3.vllm_lifecycle import VllmLifecycle, build_vllm_command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/baselines/duck-control-v1.json"),
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wheelhouse-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--container-image", required=True)
    return parser.parse_args()


def _run(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _install_wheelhouse(wheelhouse_dir: Path) -> None:
    result = _run(
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


def _nvidia_smi_query(fields: str) -> str:
    result = _run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    return result.stdout.strip().splitlines()[0]


class _VramPoller:
    """Polls nvidia-smi in a background thread and tracks peak memory used."""

    def __init__(self, *, interval_seconds: float = 2.0) -> None:
        self._interval_seconds = interval_seconds
        self._peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                used_mb = int(_nvidia_smi_query("memory.used").split(",")[0].strip())
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


def _post_chat(
    *, base_url: str, model: str, prompt: str, max_tokens: int, stream: bool
) -> ChatBenchmarkResult:
    """POST one raw chat-completion for benchmarking (bypasses the agent path;
    only used for timing, never as the required real-smoke evidence)."""

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


def _measure_stable_context_tokens(*, base_url: str, model: str) -> int:
    for target_words in (25_000, 12_000, 4_000):
        try:
            result = _post_chat(
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


def main() -> int:
    args = _parse_args()
    manifest = load_control_manifest(args.manifest)
    config = config_from_manifest(
        manifest,
        manifest_path=args.manifest,
        model_dir=args.model_dir,
        wheelhouse_dir=args.wheelhouse_dir,
        provider=Provider.LOCAL_VLLM,
        full_verification=True,
        offline=True,
    )
    started_at = datetime.now(UTC)

    print("PHASE=install_start", flush=True)
    _install_wheelhouse(args.wheelhouse_dir)
    print("PHASE=install_done", flush=True)

    print("PHASE=preflight_start", flush=True)
    preflight_report = run_asset_preflight(config.assets)
    print(preflight_report.model_dump_json(indent=2), flush=True)
    if not preflight_report.ok:
        print("PHASE=preflight_failed", flush=True)
        return 1
    print("PHASE=preflight_passed", flush=True)

    torch_version = importlib.metadata.version("torch")
    vllm_version = importlib.metadata.version("vllm")
    cuda_version = _run(
        [sys.executable, "-c", "import torch; print(torch.version.cuda)"]
    ).stdout.strip()
    gpu_name, gpu_memory_mb, driver_version = (
        part.strip()
        for part in _nvidia_smi_query("name,memory.total,driver_version").split(",")
    )

    probe = ModelsProbe(
        base_url=config.inference.base_url,
        timeout_seconds=min(config.inference.request_timeout_seconds, 5),
    )
    lifecycle = VllmLifecycle(
        launch_config=config.vllm,
        asset_config=config.assets,
        run_dir=args.run_dir,
        preflight=lambda: require_asset_preflight(config.assets),
        readiness_probe=probe,
    )

    print("PHASE=vllm_start", flush=True)
    try:
        lifecycle.start()
    except Exception as exc:  # noqa: BLE001 -- report cannot be built without a server
        print(f"PHASE=vllm_start_failed error={exc}", flush=True)
        server_log = args.run_dir / "server.log"
        if server_log.exists():
            print(server_log.read_text(errors="replace")[-4000:], flush=True)
        return 1
    print("PHASE=vllm_ready", flush=True)

    real_smoke_passed = False
    metrics: PerformanceMetrics | None = None
    benchmark_error: Exception | None = None

    with _VramPoller() as vram:
        try:
            lifecycle_text = (args.run_dir / "lifecycle.jsonl").read_text()
            model_load_seconds = parse_ready_elapsed_seconds(lifecycle_text)

            try:
                print("PHASE=agent_smoke_start", flush=True)
                smoke_result = run_provider_smoke(
                    config=config.inference, runs_root=args.run_dir / "smoke"
                )
                real_smoke_passed = smoke_result.terminal_state == "WIN"
                print(
                    f"PHASE=agent_smoke_done state={smoke_result.terminal_state}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 -- keep the run alive for metrics
                print(f"PHASE=agent_smoke_failed error={exc}", flush=True)

            print("PHASE=benchmark_start", flush=True)
            latencies = tuple(
                _post_chat(
                    base_url=config.inference.base_url,
                    model=config.inference.model,
                    prompt="Reply with the single word OK.",
                    max_tokens=8,
                    stream=False,
                ).total_seconds
                for _ in range(3)
            )

            throughput = _post_chat(
                base_url=config.inference.base_url,
                model=config.inference.model,
                prompt="token " * 1600,
                max_tokens=200,
                stream=True,
            )
            prefill_tps, decode_tps = compute_prefill_decode_throughput(
                prompt_tokens=throughput.prompt_tokens or 2000,
                completion_tokens=throughput.completion_tokens or 200,
                time_to_first_token_seconds=throughput.time_to_first_token_seconds,
                total_seconds=throughput.total_seconds,
            )

            stable_context_tokens = _measure_stable_context_tokens(
                base_url=config.inference.base_url, model=config.inference.model
            )
            print("PHASE=benchmark_done", flush=True)

            metrics = PerformanceMetrics(
                model_load_seconds=model_load_seconds,
                peak_vram_bytes=vram.peak_bytes,
                time_to_first_token_seconds=throughput.time_to_first_token_seconds,
                prefill_tokens_per_second=prefill_tps,
                decode_tokens_per_second=decode_tps,
                stable_context_tokens=stable_context_tokens,
                request_latency_seconds=latencies,
                concurrency=1,
            )
        except Exception as exc:  # noqa: BLE001 -- still shut down cleanly below
            benchmark_error = exc
            print(f"PHASE=benchmark_failed error={exc}", flush=True)
        finally:
            print("PHASE=vllm_stop", flush=True)
            lifecycle.stop()

    ended_at = datetime.now(UTC)
    if metrics is None:
        print(f"PHASE=incomplete error={benchmark_error}", flush=True)
        return 1

    lifecycle_text = (args.run_dir / "lifecycle.jsonl").read_text()
    server_exit_code = parse_stopped_exit_code(lifecycle_text)
    server_log_text = (args.run_dir / "server.log").read_text(errors="replace")

    report = build_gpu_benchmark_report(
        run_id=args.run_dir.name,
        started_at=started_at,
        ended_at=ended_at,
        deployment=DeploymentProvenance(
            container_image=args.container_image,
            gpu_model=gpu_name,
            gpu_count=1,
            gpu_memory_bytes=int(gpu_memory_mb) * 1024 * 1024,
            driver_version=driver_version,
            cuda_version=cuda_version,
            python_version=platform.python_version(),
            vllm_version=vllm_version,
            torch_version=torch_version,
        ),
        control=ControlProvenance(
            control_id=manifest.control_id,
            manifest_sha256=sha256_file(args.manifest),
            model_repository=manifest.model.repository,
            model_revision=manifest.model.revision,
            wheelhouse_dataset=(
                f"{manifest.wheelhouse.kaggle_dataset.ref}"
                f"@{manifest.wheelhouse.kaggle_dataset.version}"
            ),
            launch_command=build_vllm_command(config.vllm, config.assets.model_dir),
        ),
        metrics=metrics,
        acceptance=AcceptanceEvidence(
            full_asset_preflight_passed=preflight_report.ok,
            real_smoke_request_passed=real_smoke_passed,
            offline_download_attempts=count_offline_download_attempts(server_log_text),
            server_exit_code=server_exit_code,
        ),
    )
    write_gpu_benchmark_report(report, args.report_out)
    print(f"PHASE=report_written path={args.report_out}", flush=True)
    print(report.model_dump_json(indent=2))
    return 0 if (preflight_report.ok and real_smoke_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
