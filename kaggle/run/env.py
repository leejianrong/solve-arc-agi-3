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
PY312 = os.environ.get("ARC3_PY312", "/usr/bin/python3.12")  # RunPod images may differ from Kaggle


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


NVIDIA_LIB = Path("/usr/local/nvidia/lib64")  # the image's driver library dir (also on LD_LIBRARY_PATH)


def libcuda_link_dir() -> str:
    """Directory holding a linkable ``libcuda.so``, for FlashInfer's JIT link step.

    FlashInfer compiles Blackwell kernels at first use and links them with ``-lcuda``. Kaggle puts the driver
    in /usr/local/nvidia/lib64 (on LD_LIBRARY_PATH, so runtime works) but LIBRARY_PATH, which the linker
    reads, points at a CUDA ``stubs`` directory that does not exist, so the final link failed with "cannot
    find -lcuda" after every kernel compiled. Measured with kaggle/probe/probe.py.
    """

    if (NVIDIA_LIB / "libcuda.so").exists():
        return str(NVIDIA_LIB)
    link_dir = Path("/tmp/libcuda-link")
    link_dir.mkdir(exist_ok=True)
    found = subprocess.run(
        "ldconfig -p | grep -m1 'libcuda.so.1'", shell=True, capture_output=True, text=True
    ).stdout
    target = found.split("=>")[-1].strip() if "=>" in found else ""
    if target:
        link = link_dir / "libcuda.so"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    log(f"libcuda link dir: {link_dir} (driver {target or 'not found'})")
    return str(link_dir)


def runtime_env(bundle: Path) -> dict[str, str]:
    env = dict(os.environ)
    link_dir = libcuda_link_dir()
    env.update(
        PATH=f"{VENV}/bin:{CUDA}/bin:{env.get('PATH', '')}",
        CUDA_HOME=str(CUDA),
        LIBRARY_PATH=f"{link_dir}:{CUDA}/lib64:{env.get('LIBRARY_PATH', '')}",
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
    run(["uv", "venv", "--python", PY312, str(VENV)])
    run(
        [
            "uv", "pip", "install", "--python", str(VENV / "bin" / "python"),
            "--no-index", "--find-links", str(wheelhouse),
            "-r", str(wheelhouse / "requirements.lock"),
        ]
    )
    log("venv ready: " + subprocess.check_output([str(VENV / "bin/python"), "-c", "import vllm,torch;print(vllm.__version__,torch.__version__)"], text=True).strip())


__all__ = ["INPUT", "VENV", "build_venv", "find_dataset", "log", "run", "runtime_env", "sys"]
