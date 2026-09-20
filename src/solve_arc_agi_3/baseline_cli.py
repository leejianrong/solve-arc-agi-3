"""Command-line entry point for local smoke and offline artifact operations."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from collections.abc import Sequence
from pathlib import Path

from .baseline_config import (
    BaselineConfig,
    InferenceConfig,
    Provider,
    config_from_manifest,
)
from .baseline_manifest import load_control_manifest
from .inference import ModelsProbe
from .offline_install import build_offline_install_command
from .preflight import require_asset_preflight, run_asset_preflight
from .smoke import run_deterministic_smoke, run_provider_smoke
from .vllm_lifecycle import VllmLifecycle, build_vllm_command

DEFAULT_MANIFEST = Path("configs/baselines/duck-control-v1.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solve-arc-agi-3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run the no-network fake-server smoke")
    smoke.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    smoke.add_argument("--runs-root", type=Path, default=Path("runs/smoke"))

    provider_smoke = subparsers.add_parser(
        "provider-smoke", help="run the smoke against a configured real endpoint"
    )
    provider_smoke.add_argument("--config", type=Path, required=True)
    provider_smoke.add_argument(
        "--runs-root", type=Path, default=Path("runs/provider-smoke")
    )

    for name, help_text in (
        ("preflight", "verify mounted offline assets"),
        ("vllm-command", "print the pinned vLLM launch command"),
        ("install-command", "print the offline-only wheel install command"),
        ("serve", "preflight and run the owned local vLLM server"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--wheelhouse-dir", type=Path, required=True)
        command.add_argument("--full", action="store_true")
        if name == "serve":
            command.add_argument("--run-dir", type=Path, required=True)
    return parser


def _config(args: argparse.Namespace, *, provider: Provider) -> BaselineConfig:
    manifest = load_control_manifest(args.manifest)
    return config_from_manifest(
        manifest,
        manifest_path=args.manifest,
        model_dir=args.model_dir,
        wheelhouse_dir=args.wheelhouse_dir,
        provider=provider,
        full_verification=args.full,
        offline=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        manifest = load_control_manifest(args.manifest)
        config = config_from_manifest(
            manifest,
            manifest_path=args.manifest,
            model_dir=Path("unused-in-fake-smoke/model"),
            wheelhouse_dir=Path("unused-in-fake-smoke/wheels"),
            provider=Provider.FAKE,
            base_url="http://fake.invalid/v1",
        )
        result = run_deterministic_smoke(
            config=config.inference, runs_root=args.runs_root
        )
        print(result.model_dump_json(indent=2))
        return 0
    if args.command == "provider-smoke":
        inference_config = InferenceConfig.model_validate_json(
            args.config.read_text(encoding="utf-8")
        )
        if inference_config.provider is Provider.FAKE:
            raise ValueError("provider-smoke requires a real provider configuration")
        result = run_provider_smoke(config=inference_config, runs_root=args.runs_root)
        print(result.model_dump_json(indent=2))
        return 0

    config = _config(args, provider=Provider.LOCAL_VLLM)
    if args.command == "preflight":
        report = run_asset_preflight(config.assets)
        print(report.model_dump_json(indent=2))
        return 0 if report.ok else 1
    if args.command == "vllm-command":
        print(
            json.dumps(
                list(build_vllm_command(config.vllm, config.assets.model_dir)),
                indent=2,
            )
        )
        return 0
    if args.command == "install-command":
        print(
            json.dumps(
                list(build_offline_install_command(config.assets.wheelhouse_dir)),
                indent=2,
            )
        )
        return 0
    if args.command == "serve":
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
        stop_requested = threading.Event()
        previous_sigint = signal.signal(
            signal.SIGINT, lambda _signum, _frame: stop_requested.set()
        )
        previous_sigterm = signal.signal(
            signal.SIGTERM, lambda _signum, _frame: stop_requested.set()
        )
        try:
            with lifecycle:
                stop_requested.wait()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
