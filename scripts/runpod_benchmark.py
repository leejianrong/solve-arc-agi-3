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
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

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
from solve_arc_agi_3.runpod_probes import (
    VramPoller,
    install_wheelhouse,
    measure_stable_context_tokens,
    nvidia_smi_query,
    post_chat,
    run,
)
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
    install_wheelhouse(args.wheelhouse_dir)
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
    cuda_version = run(
        [sys.executable, "-c", "import torch; print(torch.version.cuda)"]
    ).stdout.strip()
    gpu_name, gpu_memory_mb, driver_version = (
        part.strip()
        for part in nvidia_smi_query("name,memory.total,driver_version").split(",")
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

    with VramPoller() as vram:
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
                post_chat(
                    base_url=config.inference.base_url,
                    model=config.inference.model,
                    prompt="Reply with the single word OK.",
                    max_tokens=8,
                    stream=False,
                ).total_seconds
                for _ in range(3)
            )

            throughput = post_chat(
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

            stable_context_tokens = measure_stable_context_tokens(
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
