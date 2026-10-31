"""Find a linkable libcuda on the Kaggle GPU image and prove a `-lcuda` link works (ARC-110)."""

import os
import pathlib
import subprocess
import tempfile


def sh(cmd: str) -> str:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=300, check=False
    )
    out = (r.stdout + r.stderr).strip()
    print(out[:2500], flush=True)
    return out


print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("LIBRARY_PATH:", os.environ.get("LIBRARY_PATH"))
sh("ldconfig -p | grep -i -E 'libcuda|libnvidia-ml' | head")
found = sh(
    "find / \\( -name 'libcuda.so*' -o -name 'libcuda.a' \\) -not -path '/proc/*' 2>/dev/null | head -20"
).splitlines()
sh(
    "python3 - <<'PY'\nimport ctypes\nl=ctypes.CDLL('libcuda.so.1')\nprint('dlopen ok', l)\nPY"
)
sh(
    "cat /proc/self/maps | head -0; python3 -c \"import torch;torch.zeros(1).cuda();import os;print([l.split()[-1] for l in open('/proc/self/maps') if 'libcuda' in l][:2])\""
)

d = tempfile.mkdtemp()
pathlib.Path(f"{d}/t.c").write_text(
    "int cuInit(unsigned int);int main(){return cuInit(0)==-1;}\n"
)
for cand in [
    os.path.dirname(p) for p in found if p.endswith(("libcuda.so", "libcuda.so.1"))
]:
    link = f"{d}/lnk"
    os.makedirs(link, exist_ok=True)
    src = [
        p for p in found if os.path.dirname(p) == cand and p.endswith("libcuda.so.1")
    ]
    if src:
        try:
            os.symlink(src[0], f"{link}/libcuda.so")
        except FileExistsError:
            pass
        sh(f"gcc {d}/t.c -L{link} -lcuda -o {d}/t && echo LINK_OK_{cand}")
sh(
    "ls /usr/local/cuda/lib64 | head -20; ls /usr/local/cuda/targets/x86_64-linux/lib/stubs 2>&1 | head"
)
