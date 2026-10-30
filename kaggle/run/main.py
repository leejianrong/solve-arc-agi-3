"""Offline Kaggle entry: serve the pinned Qwen3.6 27B FP8 with vLLM and play ARC-AGI-3 with our agent.

One script for both modes. In a competition rerun (KAGGLE_IS_COMPETITION_RERUN) games come from the
gateway; otherwise it plays the bundled public games offline for a few minutes (the dry run).

Order matters because Kaggle wants an action within 15 minutes of the run starting and the model takes
minutes to load: games start as soon as the gateway is up and take only their forced RESET; model calls
wait for a flag file that this script writes when vLLM answers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import VENV, build_venv, find_dataset, log, run, runtime_env  # noqa: E402

START = time.time()
RERUN = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
HARD_LIMIT_S = 9 * 3600
PLAY_UNTIL_S = int(os.getenv("ARC3_PLAY_SECONDS", str(8 * 3600 + 30 * 60 if RERUN else 25 * 60)))
GRACE_S = 15 * 60  # time for the agents to finish and the scorecard to close after the deadline

BUNDLE = find_dataset("solve-arc-agi-3-src")
MODEL = find_dataset("vrfai-qwen3-6-27b-fp8-hf-snapshot")
WHEELS = find_dataset("arc3-vllm-h100-wheelhouse-v3")
COMP = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
WORK = Path("/kaggle/working")
AGENTS = WORK / "ARC-AGI-3-Agents"
READY = Path("/tmp/arc3-model-ready")
RUN_DIR = Path(f"/tmp/arc3-run-{int(START)}")  # the vLLM lifecycle refuses an existing run dir
SUMMARY = WORK / "run_summary.json"
summary: dict[str, object] = {"mode": "rerun" if RERUN else "dry", "bundle_commit": ""}

log(f"mode={summary['mode']} play_seconds={PLAY_UNTIL_S}")
(WORK / "GIT_COMMIT").write_text((BUNDLE / "GIT_COMMIT").read_text() if (BUNDLE / "GIT_COMMIT").exists() else "")
summary["bundle_commit"] = (BUNDLE / "GIT_COMMIT").read_text().strip() if (BUNDLE / "GIT_COMMIT").exists() else ""


def write_summary() -> None:
    summary["elapsed_seconds"] = round(time.time() - START)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))


# 1+2. Venv, model server and readiness flag, all in the background: on a cold mount the venv alone took 22 minutes
#      and the model 8 more, while Kaggle wants an action inside 15. The games do not wait for any of it.
server: subprocess.Popen[bytes] | None = None
server_env = runtime_env(BUNDLE)
py = str(VENV / "bin/python")
FAILED = Path(str(READY) + ".failed")


def boot() -> None:
    global server
    try:
        build_venv(WHEELS)
        summary["venv_ready_s"] = round(time.time() - START)
        manifest = str(BUNDLE / "configs/baselines/duck-control-v1.json")
        # Quick preflight in the rerun (sizes and checksum manifest); the full 36 GB hash check is part of the
        # dry run only, to save about three minutes.
        cmd = [py, "-m", "solve_arc_agi_3", "serve", "--manifest", manifest, "--model-dir", str(MODEL),
               "--wheelhouse-dir", str(WHEELS), "--run-dir", str(RUN_DIR)]
        if os.getenv("ARC3_FULL_PREFLIGHT", "0" if RERUN else "1") == "1":
            cmd.append("--full")
        server = subprocess.Popen(cmd, env=server_env, start_new_session=True)
        t = time.time()
        while server.poll() is None:
            try:
                with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f"vLLM exited before it was ready (code {server.poll()})")
        body = json.dumps({"model": "vrfai/Qwen3.6-27B-FP8", "messages": [{"role": "user", "content": "ok"}], "max_tokens": 8}).encode()
        try:  # warm-up so the first real call does not pay for graph capture
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=300).read()
        except Exception as exc:
            log(f"warm-up request failed: {exc}")
        READY.write_text("ready")
        summary["model_ready_s"] = round(time.time() - START)
        log(f"model ready: {time.time() - t:.0f}s of serving, T+{time.time() - START:.0f}s")
    except Exception as exc:
        FAILED.write_text(str(exc))
        summary["boot_error"] = str(exc)
        log(f"BOOT FAILED, games will use fallback moves: {exc}")


threading.Thread(target=boot, daemon=True).start()

# The agents run on the system python (as in the official starter) so they never wait for the venv.
run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-index",
     "--find-links", str(COMP / "arc_agi_3_wheels"), "arc-agi", "python-dotenv"], check=False)
env = dict(os.environ)
env.update(
    PYTHONPATH=str(BUNDLE),
    ARC3_MANIFEST=str(BUNDLE / "configs/baselines/duck-control-v1.json"),
    ARC3_DEADLINE_EPOCH=str(START + PLAY_UNTIL_S),
    ARC3_READY_FLAG=str(READY),
    ARC3_MAX_CONCURRENT_CALLS=os.getenv("ARC3_MAX_CONCURRENT_CALLS", "32"),
    ARC3_MAX_ACTIONS=os.getenv("ARC3_MAX_ACTIONS", "1500"),
    # Thinking off and a short answer by default: with thinking on, an answer needs >4096 tokens, which at ~25
    # tokens/s per stream exceeds the request timeout (first dry run: 354 of 506 steps fell back).
    ARC3_THINKING=os.getenv("ARC3_THINKING", "0"),
    ARC3_MAX_OUTPUT_TOKENS=os.getenv("ARC3_MAX_OUTPUT_TOKENS", "1024"),
    ARC3_REQUEST_TIMEOUT=os.getenv("ARC3_REQUEST_TIMEOUT", "180"),
    MPLBACKEND="agg",
)

# 3. The official agents repo with our agent plugged in.
if AGENTS.exists():
    shutil.rmtree(AGENTS)
shutil.copytree(COMP / "ARC-AGI-3-Agents", AGENTS, ignore=shutil.ignore_patterns(".git"))
(AGENTS / "agents/templates/my_agent.py").write_text(
    "from agents.agent import Agent\n"
    "from solve_arc_agi_3.kaggle_agent import agent_class_from_env\n\n"
    "MyAgent = agent_class_from_env(Agent)\n"
)
(AGENTS / "agents/__init__.py").write_text(
    "from typing import Type\n"
    "from dotenv import load_dotenv\n"
    "from .agent import Agent, Playback\n"
    "from .swarm import Swarm\n"
    "from .templates.my_agent import MyAgent\n\n"
    "load_dotenv()\n"
    'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"myagent": MyAgent}\n'
)

agent_proc: subprocess.Popen[bytes] | None = None
try:
    if RERUN:
        (AGENTS / ".env").write_text(
            "SCHEME=http\nHOST=gateway\nPORT=8001\nARC_API_KEY=test-key-123\n"
            "ARC_BASE_URL=http://gateway:8001/\nOPERATION_MODE=online\nENVIRONMENTS_DIR=\n"
            f"RECORDINGS_DIR={WORK}/server_recording\n"
        )
        run(["curl", "--fail", "--retry", "999", "--retry-all-errors", "--retry-delay", "5",
             "--retry-max-time", "600", "http://gateway:8001/api/games"], check=False)
        cmd = [sys.executable, "main.py", "--agent", "myagent"]
    else:
        env.update(OPERATION_MODE="offline", ENVIRONMENTS_DIR=str(COMP / "environment_files"),
                   RECORDINGS_DIR=str(WORK / "recordings"), ARC3_DRY_GAMES=os.getenv("ARC3_DRY_GAMES", "12"))
        cmd = [sys.executable, str(BUNDLE / "run/dry_runner.py")]
    summary["agents_started_s"] = round(time.time() - START)
    log(f"starting agents (T+{time.time() - START:.0f}s): {' '.join(cmd)}")
    agent_proc = subprocess.Popen(cmd, cwd=AGENTS, env=env)
    limit = min(START + PLAY_UNTIL_S + GRACE_S, START + HARD_LIMIT_S - 300)
    while agent_proc.poll() is None and time.time() < limit:
        time.sleep(10)
    if agent_proc.poll() is None:
        log("agents did not finish inside the grace window; stopping them")
        agent_proc.terminate()
    summary["agents_exit"] = agent_proc.poll()
finally:
    if server is not None:
        server.terminate()
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill()
    for f in sorted(RUN_DIR.glob("*.log"))[-2:]:
        shutil.copy(f, WORK / f"vllm-{f.name}")
    if not RERUN:  # the starter writes this placeholder in non-rerun mode; keeps the output valid
        try:
            import pandas as pd

            pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
                WORK / "submission.parquet", index=False)
        except Exception as exc:
            log(f"placeholder submission not written: {exc}")
    write_summary()
    log(f"done at T+{time.time() - START:.0f}s")
