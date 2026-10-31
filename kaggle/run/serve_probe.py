"""ARC-110 step 1: build the venv offline, serve the pinned model with vLLM, make one request.

Not the final run: it proves the install, the server and the model load on the real GPU, with timings.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import VENV, build_venv, find_dataset, log, run, runtime_env

BUNDLE = find_dataset("solve-arc-agi-3-src")
MODEL = find_dataset("vrfai-qwen3-6-27b-fp8-hf-snapshot")
WHEELS = find_dataset("arc3-vllm-h100-wheelhouse-v3")
RUN_DIR = Path(
    f"/tmp/arc3-run-{int(time.time())}"
)  # the lifecycle refuses an existing run dir
log(f"bundle={BUNDLE} model={MODEL} wheels={WHEELS}")

t0 = time.time()
build_venv(WHEELS)
log(f"venv seconds: {time.time() - t0:.0f}")

env = runtime_env(BUNDLE)
py = str(VENV / "bin" / "python")
base = [py, "-m", "solve_arc_agi_3"]
manifest = str(BUNDLE / "configs/baselines/duck-control-v1.json")
args = [
    "--manifest",
    manifest,
    "--model-dir",
    str(MODEL),
    "--wheelhouse-dir",
    str(WHEELS),
]

t1 = time.time()
run(
    base + ["preflight", *args], env=env, check=False
)  # the full hash check passed in the previous probe
log(f"preflight seconds: {time.time() - t1:.0f}")

server = subprocess.Popen(
    base + ["serve", *args, "--run-dir", str(RUN_DIR)], env=env, start_new_session=True
)
t2 = time.time()
ready = False
while time.time() - t2 < 1500 and server.poll() is None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5) as r:
            ready = r.status == 200
    except Exception:
        ready = False
    if ready:
        break
    time.sleep(10)
log(f"server ready={ready} after {time.time() - t2:.0f}s (exit={server.poll()})")

if ready:
    body = json.dumps(
        {
            "model": "vrfai/Qwen3.6-27B-FP8",
            "messages": [
                {"role": "user", "content": "Reply with the single word: ready"}
            ],
            "max_tokens": 64,
        }
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        body,
        {"Content-Type": "application/json"},
    )
    t3 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    log(f"one request seconds: {time.time() - t3:.1f}; usage={out.get('usage')}")
    log("reply: " + json.dumps(out["choices"][0]["message"])[:300])

if ready:  # aggregate throughput with concurrent streams (sizes the 110-game run)
    import concurrent.futures as cf

    def one(i: int) -> int:
        body = json.dumps(
            {
                "model": "vrfai/Qwen3.6-27B-FP8",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Count from {i} upward, one number per line, 300 numbers.",
                    }
                ],
                "max_tokens": 300,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:1234/v1/chat/completions",
            body,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["usage"]["completion_tokens"]

    for n in (1, 8, 32):
        t4 = time.time()
        with cf.ThreadPoolExecutor(n) as ex:
            total = sum(ex.map(one, range(n)))
        dt = time.time() - t4
        log(
            f"concurrency {n}: {total} tokens in {dt:.1f}s = {total / dt:.1f} tok/s aggregate, {total / dt / n:.1f} tok/s per stream"
        )

server.terminate()
try:
    server.wait(60)
except subprocess.TimeoutExpired:
    server.kill()
import shutil

for f in sorted(RUN_DIR.glob("*.log")):
    shutil.copy(
        f, Path("/kaggle/working") / f.name
    )  # full logs come back as kernel output files
    log(f"--- tail of {f.name}")
    print("\n".join(f.read_text(errors="replace").splitlines()[-60:]))
sys.exit(0 if ready else 1)
