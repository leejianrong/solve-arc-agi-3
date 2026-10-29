"""Environment probe for the ARC-AGI-3 Kaggle GPU image (ARC-110, step 0).

Prints what the offline notebook can rely on. Makes no network calls and loads no model.
"""
import glob
import os
import shutil
import subprocess
import sys


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    print((r.stdout + r.stderr).strip()[:3000], flush=True)


print("python:", sys.version, sys.executable)
print("KAGGLE_IS_COMPETITION_RERUN:", os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
for cmd in (
    "nvidia-smi",
    "nvcc --version",
    "which -a nvcc gcc g++ ninja uv pip python3 python3.12",
    "ls -d /usr/local/cuda* /usr/lib/cuda* 2>&1",
    "find / -name curand_kernel.h -not -path '*/proc/*' 2>/dev/null | head",
    "find / -name nvcc -type f 2>/dev/null | head",
    "nproc; free -g; df -h /kaggle/working /tmp",
    "ls /kaggle/input /kaggle/input/* | head -60",
    "pip --version; pip list 2>/dev/null | grep -i -E 'torch|vllm|transformers|flashinfer|triton|numpy|arc'",
    "python3 -c \"import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"",
):
    sh(cmd)
for p in glob.glob("/kaggle/input/*/*")[:40]:
    print(p, "dir" if os.path.isdir(p) else os.path.getsize(p))
print("disk free /tmp GB:", shutil.disk_usage("/tmp").free / 1e9)
