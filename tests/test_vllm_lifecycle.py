import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from solve_arc_agi_3.baseline_config import Provider, config_from_manifest
from solve_arc_agi_3.baseline_manifest import load_control_manifest
from solve_arc_agi_3.preflight import PreflightReport
from solve_arc_agi_3.vllm_lifecycle import (
    VllmLifecycle,
    VllmLifecycleError,
    build_vllm_command,
)

PROJECT_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs/baselines/duck-control-v1.json"


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeProcess:
    pid = 4321

    def __init__(self, *, wait_times_out: bool = False) -> None:
        self.exit_code = None
        self.terminated = False
        self.killed = False
        self.wait_times_out = wait_times_out

    def poll(self):
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout_seconds: float) -> int:
        del timeout_seconds
        if self.wait_times_out and not self.killed:
            raise TimeoutError
        self.exit_code = 0 if not self.killed else -9
        return self.exit_code


class FakeRunner:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.starts = 0
        self.command: tuple[str, ...] = ()
        self.env: Mapping[str, str] = {}

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> FakeProcess:
        del cwd, log_path
        self.starts += 1
        self.command = tuple(command)
        self.env = env
        return self.process


class FakeProbe:
    def __init__(self, ready_on_call: int) -> None:
        self.ready_on_call = ready_on_call
        self.calls = 0

    def is_ready(self, expected_model: str) -> bool:
        assert expected_model == "vrfai/Qwen3.6-27B-FP8"
        self.calls += 1
        return self.calls >= self.ready_on_call


def _config(tmp_path: Path):
    manifest = load_control_manifest(MANIFEST_PATH)
    return config_from_manifest(
        manifest,
        manifest_path=MANIFEST_PATH,
        model_dir=tmp_path / "model",
        wheelhouse_dir=tmp_path / "wheels",
        provider=Provider.LOCAL_VLLM,
    )


def _report(config) -> PreflightReport:
    return PreflightReport(
        control_id="duck-clean-room-v1",
        manifest_path=config.assets.manifest_path,
        model_dir=config.assets.model_dir,
        wheelhouse_dir=config.assets.wheelhouse_dir,
        full_verification=False,
        verified_files=(),
        errors=(),
    )


def test_build_command_contains_every_pinned_serving_flag(tmp_path: Path) -> None:
    config = _config(tmp_path)

    command = build_vllm_command(config.vllm, config.assets.model_dir)

    assert command[:3] == (
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    )
    assert command[command.index("--served-model-name") + 1] == config.inference.model
    assert command[command.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert "--enable-prefix-caching" in command
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true}'
    )


def test_lifecycle_preflights_starts_polls_records_and_stops_owned_process(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    clock = FakeClock()
    process = FakeProcess()
    runner = FakeRunner(process)
    probe = FakeProbe(ready_on_call=2)
    lifecycle = VllmLifecycle(
        launch_config=config.vllm,
        asset_config=config.assets,
        run_dir=tmp_path / "run",
        preflight=lambda: _report(config),
        readiness_probe=probe,
        process_runner=runner,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    lifecycle.start()
    lifecycle.stop()

    assert runner.starts == 1
    assert runner.env["HF_HUB_OFFLINE"] == "1"
    assert runner.env["TRANSFORMERS_OFFLINE"] == "1"
    assert process.terminated is True
    assert process.killed is False
    assert json.loads((tmp_path / "run/pid.json").read_text())["pid"] == 4321
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "run/lifecycle.jsonl").read_text().splitlines()
    ]
    assert events == ["process_started", "ready", "terminate_sent", "stopped"]


def test_lifecycle_does_not_spawn_when_preflight_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    process = FakeProcess()
    runner = FakeRunner(process)
    bad_report = _report(config).model_copy(update={"errors": ("missing model",)})
    lifecycle = VllmLifecycle(
        launch_config=config.vllm,
        asset_config=config.assets,
        run_dir=tmp_path / "run",
        preflight=lambda: bad_report,
        readiness_probe=FakeProbe(ready_on_call=1),
        process_runner=runner,
    )

    with pytest.raises(VllmLifecycleError, match="preflight"):
        lifecycle.start()

    assert runner.starts == 0


def test_shutdown_kills_only_owned_process_after_timeout(tmp_path: Path) -> None:
    config = _config(tmp_path)
    process = FakeProcess(wait_times_out=True)
    runner = FakeRunner(process)
    lifecycle = VllmLifecycle(
        launch_config=config.vllm,
        asset_config=config.assets,
        run_dir=tmp_path / "run",
        preflight=lambda: _report(config),
        readiness_probe=FakeProbe(ready_on_call=1),
        process_runner=runner,
    )
    lifecycle.start()

    lifecycle.stop()

    assert process.terminated is True
    assert process.killed is True


def test_startup_timeout_is_bounded_and_cleans_up_owned_process(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    launch = config.vllm.model_copy(
        update={"startup_timeout_seconds": 2, "readiness_poll_seconds": 1}
    )
    clock = FakeClock()
    process = FakeProcess()
    lifecycle = VllmLifecycle(
        launch_config=launch,
        asset_config=config.assets,
        run_dir=tmp_path / "run",
        preflight=lambda: _report(config),
        readiness_probe=FakeProbe(ready_on_call=100),
        process_runner=FakeRunner(process),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(VllmLifecycleError, match="readiness timeout"):
        lifecycle.start()

    assert clock.value == 2
    assert process.terminated is True
