"""Shared helpers for the offline Kaggle run scripts (no third-party imports)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
VENV = Path("/tmp/arc3-venv")  # /kaggle/working is only 20 GB; /tmp has ~1 TB
CUDA = Path("/usr/local/cuda")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_dataset(slug: str) -> Path:
    """Locate a mounted dataset, whichever layout Kaggle used for it."""
    name = slug.split("/")[-1]
    for pattern in (f"datasets/*/{name}", f"*/{name}", name):
        hits = sorted(INPUT.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"dataset {slug} is not mounted under {INPUT}")


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if check and proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd[0]}")
    return proc.returncode


def runtime_env(bundle: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        PATH=f"{VENV}/bin:{CUDA}/bin:{env.get('PATH', '')}",
        CUDA_HOME=str(CUDA),
        PYTHONPATH=str(bundle),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
    )
    return env


def build_venv(wheelhouse: Path) -> None:
    """Python 3.12 venv (the wheelhouse is cp312; Kaggle's default is 3.13), installed offline."""
    if (VENV / "bin" / "vllm").exists():
        log("venv already built")
        return
    run(["uv", "venv", "--python", "/usr/bin/python3.12", str(VENV)])
    run(
        [
            "uv", "pip", "install", "--python", str(VENV / "bin" / "python"),
            "--no-index", "--find-links", str(wheelhouse),
            "-r", str(wheelhouse / "requirements.lock"),
        ]
    )
    log("venv ready: " + subprocess.check_output([str(VENV / "bin/python"), "-c", "import vllm,torch;print(vllm.__version__,torch.__version__)"], text=True).strip())


__all__ = ["INPUT", "VENV", "build_venv", "find_dataset", "log", "run", "runtime_env", "sys"]
