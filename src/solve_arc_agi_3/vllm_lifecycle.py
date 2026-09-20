"""Owned-process lifecycle for the pinned offline vLLM server."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Protocol, Self

from .baseline_config import AssetConfig, VllmLaunchConfig
from .preflight import PreflightReport


class OwnedProcess(Protocol):
    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout_seconds: float) -> int: ...


class ProcessRunner(Protocol):
    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> OwnedProcess: ...


class ReadinessProbe(Protocol):
    def is_ready(self, expected_model: str) -> bool: ...


class _PopenProcess:
    def __init__(self, process: subprocess.Popen[bytes], log_file: IO[bytes]) -> None:
        self._process = process
        self._log_file = log_file

    @property
    def pid(self) -> int:
        return self._process.pid

    def poll(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        try:
            os.killpg(self._process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait(self, timeout_seconds: float) -> int:
        try:
            return self._process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("vLLM process did not exit before timeout") from exc
        finally:
            if self._process.poll() is not None:
                self._log_file.close()


class SubprocessRunner:
    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> OwnedProcess:
        log_file = log_path.open("ab")
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise
        return _PopenProcess(process, log_file)


def build_vllm_command(config: VllmLaunchConfig, model_dir: Path) -> tuple[str, ...]:
    """Build the complete pinned vLLM launch command deterministically."""

    command = [
        *config.executable,
        "--model",
        str(model_dir.resolve()),
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-model-len",
        str(config.max_model_len),
        "--tool-call-parser",
        config.tool_call_parser,
        "--reasoning-parser",
        config.reasoning_parser,
        "--generation-config",
        config.generation_config,
        "--enable-auto-tool-choice",
    ]
    if config.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if config.preserve_thinking:
        command.extend(["--default-chat-template-kwargs", '{"enable_thinking":true}'])
    return tuple(command)


class VllmLifecycleError(RuntimeError):
    pass


class VllmLifecycle:
    """Start, observe, and stop only the vLLM process created by this instance."""

    def __init__(
        self,
        *,
        launch_config: VllmLaunchConfig,
        asset_config: AssetConfig,
        run_dir: Path,
        preflight: Callable[[], PreflightReport],
        readiness_probe: ReadinessProbe,
        process_runner: ProcessRunner | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._launch_config = launch_config
        self._asset_config = asset_config
        self.run_dir = run_dir
        self._preflight = preflight
        self._readiness_probe = readiness_probe
        self._process_runner = process_runner or SubprocessRunner()
        self._monotonic = monotonic
        self._sleep = sleep
        self._process: OwnedProcess | None = None

    def _record_event(self, event: str, **fields: object) -> None:
        entry = {"event": event, "monotonic_seconds": self._monotonic(), **fields}
        with (self.run_dir / "lifecycle.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, sort_keys=True) + "\n")

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._asset_config.offline:
            env.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
        return env

    def start(self) -> None:
        if self._process is not None:
            raise VllmLifecycleError("vLLM lifecycle has already started a process")

        self.run_dir.mkdir(parents=True, exist_ok=False)
        try:
            report = self._preflight()
            if not report.ok:
                raise VllmLifecycleError("asset preflight returned errors")
            command = build_vllm_command(
                self._launch_config, self._asset_config.model_dir
            )
            (self.run_dir / "launch_config.json").write_text(
                json.dumps(
                    {
                        "command": list(command),
                        "offline": self._asset_config.offline,
                        "preflight": report.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            self._process = self._process_runner.start(
                command,
                cwd=self.run_dir,
                env=self._environment(),
                log_path=self.run_dir / "server.log",
            )
            (self.run_dir / "pid.json").write_text(
                json.dumps({"pid": self._process.pid, "owned_by_current_run": True})
                + "\n",
                encoding="utf-8",
            )
            self._record_event("process_started", pid=self._process.pid)

            started = self._monotonic()
            deadline = started + self._launch_config.startup_timeout_seconds
            while self._monotonic() < deadline:
                exit_code = self._process.poll()
                if exit_code is not None:
                    raise VllmLifecycleError(
                        f"vLLM exited before readiness with code {exit_code}"
                    )
                if self._readiness_probe.is_ready(
                    self._launch_config.served_model_name
                ):
                    self._record_event(
                        "ready", elapsed_seconds=self._monotonic() - started
                    )
                    return
                self._sleep(self._launch_config.readiness_poll_seconds)
            raise VllmLifecycleError(
                "vLLM readiness timeout after "
                f"{self._launch_config.startup_timeout_seconds} seconds"
            )
        except Exception as exc:
            self._record_event("startup_error", error=str(exc))
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is not None:
            self._record_event("process_already_exited", pid=process.pid)
            return

        stopped = self._monotonic()
        process.terminate()
        self._record_event("terminate_sent", pid=process.pid)
        try:
            exit_code = process.wait(self._launch_config.shutdown_timeout_seconds)
        except TimeoutError:
            process.kill()
            self._record_event("kill_sent", pid=process.pid)
            exit_code = process.wait(self._launch_config.shutdown_timeout_seconds)
        self._record_event(
            "stopped",
            pid=process.pid,
            exit_code=exit_code,
            elapsed_seconds=self._monotonic() - stopped,
        )

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
