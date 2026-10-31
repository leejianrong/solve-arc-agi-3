#!/usr/bin/env python3
"""Orchestrates the ARC-40 GPU resource-envelope concurrency sweep.

Starts the pinned vLLM server once, then for each concurrency level in
`--concurrency-levels` (default 1,4,8,16) fires that many real agent episodes
at once -- the same `smoke.run_provider_smoke` path used by ARC-42's
acceptance run (`BaselineAgentCore` / `OpenAICompatibleClient`), not a
parallel raw client -- and writes one `ConcurrencyBenchmarkReport` per level
under `--evidence-dir`. Correctness/stability comes from each episode's own
real terminal outcome (WIN/other), read back from the trace each smoke run
already writes; throughput comes from the same episodes' own token usage and
elapsed time. The stable-context ceiling at each level is measured with the
existing raw context probe (`runpod_probes.measure_stable_context_tokens`,
already used and disclosed as timing/ceiling-only in ARC-42), fired
concurrently and reduced to its minimum across the N concurrent streams --
the guaranteed ceiling under that level's shared KV cache pressure.

Runs on the pod itself, inside the project's venv. Not unit tested -- the
testable report-assembly logic lives in solve_arc_agi_3.concurrency_report;
this is ops glue, the same split runpod_benchmark.py uses.

Exits 0 only when preflight passed and every concurrency level produced a
report with at least one successful episode. A level with zero successes
still gets a written report (evidence, not a fabricated pass) but flips the
exit code to 1.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from solve_arc_agi_3.baseline_config import (
    InferenceConfig,
    Provider,
    config_from_manifest,
)
from solve_arc_agi_3.baseline_manifest import load_control_manifest, sha256_file
from solve_arc_agi_3.concurrency_report import (
    EpisodeResult,
    aggregate_concurrency_metrics,
    build_concurrency_benchmark_report,
    parse_episode_result_from_trace_text,
)
from solve_arc_agi_3.gpu_report import (
    ConcurrencyMetrics,
    ControlProvenance,
    DeploymentProvenance,
    write_concurrency_benchmark_report,
)
from solve_arc_agi_3.inference import ModelsProbe
from solve_arc_agi_3.preflight import require_asset_preflight, run_asset_preflight
from solve_arc_agi_3.runpod_probes import (
    VramPoller,
    install_wheelhouse,
    measure_stable_context_tokens,
    nvidia_smi_query,
    run,
)
from solve_arc_agi_3.runpod_report import (
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
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--concurrency-levels", default="1,4,8,16")
    return parser.parse_args()


def _run_one_episode(*, config: InferenceConfig, smoke_root: Path) -> EpisodeResult:
    """Run one real agent episode; parse its outcome from its own trace file."""

    result = run_provider_smoke(config=config, runs_root=smoke_root)
    trace_text = result.trace_path.read_text(encoding="utf-8")
    return parse_episode_result_from_trace_text(trace_text)


def _run_concurrent_episodes(
    *, concurrency: int, config: InferenceConfig, run_dir: Path
) -> tuple[EpisodeResult, ...]:
    """Fire `concurrency` real agent episodes at once through the agent path."""

    smoke_root = run_dir / "smoke" / f"concurrency-{concurrency}"
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_run_one_episode, config=config, smoke_root=smoke_root)
            for _ in range(concurrency)
        ]
        return tuple(future.result() for future in futures)


def _measure_stable_context_at_concurrency(
    *, concurrency: int, base_url: str, model: str
) -> int:
    """Fire `concurrency` concurrent context-ceiling probes; take the minimum.

    The minimum is the guaranteed ceiling: any of the N concurrent streams
    could be the one starved by the others' KV cache usage.
    """

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(measure_stable_context_tokens, base_url=base_url, model=model)
            for _ in range(concurrency)
        ]
        return min(future.result() for future in futures)


def _run_one_level(
    *,
    concurrency: int,
    config: InferenceConfig,
    run_dir: Path,
    deployment: DeploymentProvenance,
    control: ControlProvenance,
) -> tuple[ConcurrencyMetrics, datetime, datetime]:
    started = datetime.now(UTC)
    with VramPoller() as vram:
        wall_start = time.monotonic()
        results = _run_concurrent_episodes(
            concurrency=concurrency, config=config, run_dir=run_dir
        )
        wall_clock_seconds = max(time.monotonic() - wall_start, 1e-6)
        stable_context_tokens = _measure_stable_context_at_concurrency(
            concurrency=concurrency, base_url=config.base_url, model=config.model
        )
        metrics = aggregate_concurrency_metrics(
            results,
            concurrency=concurrency,
            wall_clock_seconds=wall_clock_seconds,
            peak_vram_bytes=vram.peak_bytes,
            stable_context_tokens=stable_context_tokens,
        )
    ended = datetime.now(UTC)
    return metrics, started, ended


def main() -> int:
    args = _parse_args()
    concurrency_levels = [int(level) for level in args.concurrency_levels.split(",")]
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
    deployment = DeploymentProvenance(
        container_image=args.container_image,
        gpu_model=gpu_name,
        gpu_count=1,
        gpu_memory_bytes=int(gpu_memory_mb) * 1024 * 1024,
        driver_version=driver_version,
        cuda_version=cuda_version,
        python_version=platform.python_version(),
        vllm_version=vllm_version,
        torch_version=torch_version,
    )
    control = ControlProvenance(
        control_id=manifest.control_id,
        manifest_sha256=sha256_file(args.manifest),
        model_repository=manifest.model.repository,
        model_revision=manifest.model.revision,
        wheelhouse_dataset=(
            f"{manifest.wheelhouse.kaggle_dataset.ref}"
            f"@{manifest.wheelhouse.kaggle_dataset.version}"
        ),
        launch_command=build_vllm_command(config.vllm, config.assets.model_dir),
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
    except Exception as exc:  # noqa: BLE001 -- no sweep can run without a server
        print(f"PHASE=vllm_start_failed error={exc}", flush=True)
        server_log = args.run_dir / "server.log"
        if server_log.exists():
            print(server_log.read_text(errors="replace")[-4000:], flush=True)
        return 1
    print("PHASE=vllm_ready", flush=True)

    lifecycle_text = (args.run_dir / "lifecycle.jsonl").read_text()
    model_load_seconds = parse_ready_elapsed_seconds(lifecycle_text)
    print(f"PHASE=model_load_seconds value={model_load_seconds}", flush=True)

    all_levels_passed = True
    try:
        for concurrency in concurrency_levels:
            print(f"PHASE=level_start concurrency={concurrency}", flush=True)
            try:
                metrics, level_started, level_ended = _run_one_level(
                    concurrency=concurrency,
                    config=config.inference,
                    run_dir=args.run_dir,
                    deployment=deployment,
                    control=control,
                )
            except Exception as exc:  # noqa: BLE001 -- record the failure, keep sweeping
                print(
                    f"PHASE=level_failed concurrency={concurrency} error={exc}",
                    flush=True,
                )
                all_levels_passed = False
                continue

            report = build_concurrency_benchmark_report(
                run_id=f"{args.run_dir.name}-concurrency-{concurrency}",
                started_at=level_started,
                ended_at=level_ended,
                deployment=deployment,
                control=control,
                metrics=metrics,
            )
            write_concurrency_benchmark_report(
                report, args.evidence_dir / f"concurrency-{concurrency}.json"
            )
            print(
                "PHASE=level_done "
                f"concurrency={concurrency} "
                f"successful={metrics.successful_requests}/{metrics.total_requests} "
                f"completion_tps={metrics.aggregate_completion_tokens_per_second:.2f} "
                f"stable_context_tokens={metrics.stable_context_tokens}",
                flush=True,
            )
            if metrics.successful_requests == 0:
                all_levels_passed = False
    finally:
        print("PHASE=vllm_stop", flush=True)
        lifecycle.stop()

    lifecycle_text = (args.run_dir / "lifecycle.jsonl").read_text()
    server_exit_code = parse_stopped_exit_code(lifecycle_text)
    print(f"PHASE=server_exit_code value={server_exit_code}", flush=True)
    print("PHASE=sweep_done", flush=True)
    return 0 if (all_levels_passed and server_exit_code == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
