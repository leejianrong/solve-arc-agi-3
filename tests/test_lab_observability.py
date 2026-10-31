"""Rehearsal of the pod-side observability (scripts/lab_obs.sh): every way a pod job can fail must surface fast,
ship its evidence off the pod and terminate exactly once. Runs the real library with stubs on a seconds-long clock;
all scenarios run in parallel so the whole file takes about ten seconds."""

from __future__ import annotations

import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

HARNESS = Path(__file__).parent / "lab_obs_harness.sh"
SCENARIOS = (
    "phase_deadline",
    "fatal_line",
    "vllm_log_excerpt",
    "stall",
    "heartbeat_prevents_stall",
    "healthy",
    "finish",
    "single_action",
)


class Run:
    def __init__(self, scenario: str, root: Path) -> None:
        self.dir = root / scenario
        done = subprocess.run(
            ["bash", str(HARNESS), scenario, str(self.dir)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.out = done.stdout + done.stderr
        self.facts = dict(
            line[5:].split("=", 1)
            for line in done.stdout.splitlines()
            if line.startswith("FACT ")
        )

    def bundle(self) -> str:
        with tarfile.open(self.dir / "w" / "diag.tgz") as tar:
            return "".join(
                tar.extractfile(m).read().decode(errors="replace")  # type: ignore[union-attr]
                for m in tar.getmembers()
                if m.isfile()
            )


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Run]:
    root = tmp_path_factory.mktemp("lab")
    with ThreadPoolExecutor(len(SCENARIOS)) as pool:
        return dict(
            zip(
                SCENARIOS,
                pool.map(lambda name: Run(name, root), SCENARIOS),
                strict=True,
            )
        )


def test_a_phase_that_overruns_its_limit_fails_fast_and_ships_evidence(
    runs: dict[str, Run],
) -> None:
    r = runs["phase_deadline"]
    assert "FATAL: phase slow exceeded its 2s limit" in r.out
    assert (
        r.facts["terminated_count"] == "1"
        and "phase slow exceeded" in r.facts["terminated_reasons"]
    )
    assert r.facts["has_tgz"] == "yes" and "RP_ARTIFACT_CODE=arc3lab-failure-" in r.out
    assert (r.dir / "w" / "FAILED").read_text().strip()  # the marker records why


def test_a_known_fatal_line_fails_the_job_at_once(runs: dict[str, Run]) -> None:
    r = runs["fatal_line"]
    assert (
        "FATAL: fatal line in the job log: RuntimeError: Engine core initialization failed"
        in r.out
    )
    assert (
        r.facts["terminated_count"] == "1"
        and (r.dir / "w" / "FAILED").read_text().strip()
    )


def test_the_first_vllm_error_and_the_log_tail_are_printed_into_the_job_log_and_bundled(
    runs: dict[str, Run],
) -> None:
    r = runs["vllm_log_excerpt"]
    assert (
        "--- DIAG excerpt (failure) ---" in r.out
        and "ERROR engine core failed to start" in r.out
    )
    assert (
        "RuntimeError: boom" in r.out
    )  # visible in `pod logs` even if the relay fails
    assert (
        "ERROR engine core failed to start" in r.bundle()
        and r.facts["terminated_count"] == "1"
    )


def test_silence_fails_the_job_but_heartbeats_count_as_output(
    runs: dict[str, Run],
) -> None:
    assert (
        "FATAL: no output for 3s" in runs["stall"].out
        and runs["stall"].facts["terminated_count"] == "1"
    )
    ok = runs["heartbeat_prevents_stall"]
    assert "FATAL" not in ok.out and ok.facts["terminated_count"] == "0"
    assert (
        "HEARTBEAT t=" in ok.out and "phase=-" in ok.out and "gpu=1024MiB,5%" in ok.out
    )


def test_a_healthy_job_is_left_alone(runs: dict[str, Run]) -> None:
    r = runs["healthy"]
    assert (
        "PHASE=work" in r.out
        and "FATAL" not in r.out
        and r.facts["terminated_count"] == "0"
    )


def test_a_normal_finish_ships_the_results_and_terminates_once(
    runs: dict[str, Run],
) -> None:
    r = runs["finish"]
    assert (
        "DONE: job finished" in r.out and "RP_ARTIFACT_CODE=arc3lab-results-" in r.out
    )
    assert (
        r.facts["terminated_count"] == "1"
        and r.facts["terminated_reasons"] == "job finished|"
    )
    assert "ok" in r.bundle()


def test_two_failures_at_once_terminate_only_once(runs: dict[str, Run]) -> None:
    assert runs["single_action"].facts["terminated_count"] == "1"


def test_the_diagnostics_never_contain_the_environment(runs: dict[str, Run]) -> None:
    for name in ("phase_deadline", "vllm_log_excerpt"):
        assert "s3cr3t-sentinel-value" not in runs[name].out
        assert "s3cr3t-sentinel-value" not in runs[name].bundle()
