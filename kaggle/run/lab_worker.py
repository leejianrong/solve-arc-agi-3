"""Pod-side agent lab (RunPod): boot the pinned vLLM once, then run queued agent experiments.

Why: a Kaggle dry run costs an hour (cold boot, one configuration). Here the model server stays up and each
experiment is a short offline run of the current source bundle (re-downloaded before every experiment), so a
code change turns into a measured result in about ten minutes. Results are printed as LAB_RESULT lines, which
`runpodctl pod logs` shows without any public IP.

Layout it expects (created by scripts/runpod_lab_entrypoint.sh): /kaggle/input/{datasets,competitions}/... as
on Kaggle, so the same env.py helpers and the same agent shim work unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import VENV, build_venv, find_dataset, log, runtime_env

BUNDLE_SLUG = "leejianrong/solve-arc-agi-3-src"
BUNDLE_DIR = Path("/kaggle/input/datasets/leejianrong/solve-arc-agi-3-src")
COMP = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
OUT = Path("/workspace/out")
WORK = Path("/workspace/lab")
AGENT_PY = os.environ.get("ARC3_AGENT_PYTHON", "/opt/agent-venv/bin/python")
LIFETIME_S = int(os.environ.get("LAB_LIFETIME_S", str(3 * 3600)))
START = time.time()


def refresh_bundle() -> None:
    """Download the newest source bundle over the previous copy."""
    tmp = Path("/tmp/bundle-dl")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            BUNDLE_SLUG,
            "-p",
            str(tmp),
            "--unzip",
            "-q",
        ],
        check=True,
        timeout=300,
    )
    shutil.rmtree(BUNDLE_DIR, ignore_errors=True)
    BUNDLE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp, BUNDLE_DIR)


def boot_server() -> subprocess.Popen[bytes]:
    wheels = find_dataset("arc3-vllm-h100-wheelhouse-v3")
    model = find_dataset("vrfai-qwen3-6-27b-fp8-hf-snapshot")
    build_venv(wheels)
    env = runtime_env(BUNDLE_DIR)
    manifest = str(BUNDLE_DIR / "configs/baselines/duck-control-v1.json")
    run_dir = Path(f"/tmp/arc3-lab-run-{int(time.time())}")
    cmd = [
        str(VENV / "bin/python"),
        "-m",
        "solve_arc_agi_3",
        "serve",
        "--manifest",
        manifest,
        "--model-dir",
        str(model),
        "--wheelhouse-dir",
        str(wheels),
        "--run-dir",
        str(run_dir),
    ]
    server = subprocess.Popen(cmd, env=env, start_new_session=True)
    t = time.time()
    while server.poll() is None:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:1234/v1/models", timeout=5
            ) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        # The pod (and its disk) disappears right after the job, so put the evidence in the job log now.
        for f in sorted(run_dir.glob("*.log"))[:2]:
            lines = f.read_text(errors="replace").splitlines()
            err = [
                i
                for i, x in enumerate(lines)
                if "Error" in x or "ERROR" in x or "FAILED" in x
            ]
            start = max(0, (err[0] if err else len(lines) - 40) - 5)
            print(
                f"LAB_VLLM_LOG {f.name} ({len(lines)} lines), first error near line {err[0] if err else None}:",
                flush=True,
            )
            print("\n".join(x[:300] for x in lines[start : start + 60]), flush=True)
        raise RuntimeError(f"vLLM exited early (code {server.poll()})")
    log(
        f"LAB_SERVER_READY after {time.time() - t:.0f}s of serving (T+{time.time() - START:.0f}s)"
    )
    return server


def prepare_agents_repo() -> Path:
    repo = WORK / "ARC-AGI-3-Agents"
    shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(
        COMP / "ARC-AGI-3-Agents", repo, ignore=shutil.ignore_patterns(".git")
    )
    (repo / "agents/templates/my_agent.py").write_text(
        "from agents.agent import Agent\nfrom solve_arc_agi_3.kaggle_agent import agent_class_from_env\n\nMyAgent = agent_class_from_env(Agent)\n"
    )
    (repo / "agents/__init__.py").write_text(
        "from typing import Type\nfrom dotenv import load_dotenv\nfrom .agent import Agent, Playback\nfrom .swarm import Swarm\n"
        'from .templates.my_agent import MyAgent\n\nload_dotenv()\nAVAILABLE_AGENTS: dict[str, Type[Agent]] = {"myagent": MyAgent}\n'
    )
    return repo


def summarize(name: str, log_text: str, scorecard: Path) -> dict[str, object]:
    levels: dict[str, int] = {}
    steps = 0
    for line in log_text.splitlines():
        m = re.search(r"\| (\S+) - \S+: count (\d+), levels completed (\d+)", line)
        if m:
            steps += 1
            levels[m.group(1)] = max(levels.get(m.group(1), 0), int(m.group(3)))
    stats = [line for line in log_text.splitlines() if "model stats:" in line]
    score = None
    if scorecard.exists():
        try:
            score = json.loads(scorecard.read_text()).get("score")
        except Exception:
            pass
    return {
        "id": name,
        "score": score,
        "steps": steps,
        "fallbacks": log_text.count("model step failed"),
        "games_with_levels": {g: n for g, n in levels.items() if n},
        "model_stats": stats[-1].split("| INFO |")[-1].strip() if stats else None,
    }


def run_experiment(exp: dict[str, object]) -> dict[str, object]:
    name = str(exp["id"])
    out = OUT / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    repo = prepare_agents_repo()
    seconds = int(exp.get("seconds", 600))  # type: ignore[call-overload]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(BUNDLE_DIR),
        ARC3_MANIFEST=str(BUNDLE_DIR / "configs/baselines/duck-control-v1.json"),
        ARC3_DEADLINE_EPOCH=str(time.time() + seconds),
        ARC3_MAX_ACTIONS="1500",
        OPERATION_MODE="offline",
        ENVIRONMENTS_DIR=str(COMP / "environment_files"),
        RECORDINGS_DIR=str(out / "recordings"),
        ARC3_SCORECARD_OUT=str(out / "scorecard.json"),
        MPLBACKEND="agg",
    )
    env.pop("ARC3_READY_FLAG", None)
    env.update({k: str(v) for k, v in dict(exp.get("env", {})).items()})  # type: ignore[call-overload]
    log(f"LAB_START {name} env={exp.get('env')} seconds={seconds}")
    t = time.time()
    proc = subprocess.Popen(
        [AGENT_PY, str(BUNDLE_DIR / "run/dry_runner.py")],
        cwd=repo,
        env=env,
        stdout=(out / "run.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        proc.wait(timeout=seconds + 900)
    except subprocess.TimeoutExpired:
        proc.kill()
    text = (out / "run.log").read_text(errors="replace")
    result = summarize(name, text, out / "scorecard.json")
    result["wall_s"] = round(time.time() - t)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("LAB_RESULT " + json.dumps(result), flush=True)
    for line in [x for x in text.splitlines() if "model sample #" in x][:3]:
        print("LAB_SAMPLE " + line[:400], flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    refresh_bundle()
    server = boot_server()
    done: set[str] = set()
    last_beat = 0.0
    try:
        while time.time() - START < LIFETIME_S:
            try:
                refresh_bundle()
                queue = json.loads((BUNDLE_DIR / "run/lab_queue.json").read_text())
            except Exception as exc:
                log(f"LAB_QUEUE_ERROR {exc}")
                time.sleep(30)
                continue
            todo = [e for e in queue if e["id"] not in done]
            if any(e["id"] == "STOP" for e in todo):
                log("LAB_STOP requested")
                break
            if not todo:
                if time.time() - last_beat > 300:
                    log(
                        f"LAB_IDLE waiting for new experiments (T+{time.time() - START:.0f}s)"
                    )
                    last_beat = time.time()
                time.sleep(30)
                continue
            exp = todo[0]
            done.add(str(exp["id"]))
            try:
                run_experiment(exp)
            except Exception as exc:
                log(f"LAB_EXPERIMENT_FAILED {exp['id']}: {exc}")
    finally:
        server.terminate()
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill()
        log("LAB_DONE")


if __name__ == "__main__":
    main()
