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
from env import VENV, build_venv, find_dataset, log, run, runtime_env  # noqa: E402

BUNDLE = find_dataset("solve-arc-agi-3-src")
MODEL = find_dataset("vrfai-qwen3-6-27b-fp8-hf-snapshot")
WHEELS = find_dataset("arc3-vllm-h100-wheelhouse-v3")
RUN_DIR = Path(f"/tmp/arc3-run-{int(time.time())}")  # the lifecycle refuses an existing run dir
log(f"bundle={BUNDLE} model={MODEL} wheels={WHEELS}")

t0 = time.time()
build_venv(WHEELS)
log(f"venv seconds: {time.time() - t0:.0f}")

env = runtime_env(BUNDLE)
py = str(VENV / "bin" / "python")
base = [py, "-m", "solve_arc_agi_3"]
manifest = str(BUNDLE / "configs/baselines/duck-control-v1.json")
args = ["--manifest", manifest, "--model-dir", str(MODEL), "--wheelhouse-dir", str(WHEELS)]

t1 = time.time()
run(base + ["preflight", *args, "--full"], env=env, check=False)
log(f"preflight seconds: {time.time() - t1:.0f}")

server = subprocess.Popen(base + ["serve", *args, "--run-dir", str(RUN_DIR)], env=env, start_new_session=True)
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
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
            "max_tokens": 64,
        }
    ).encode()
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", body, {"Content-Type": "application/json"})
    t3 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    log(f"one request seconds: {time.time() - t3:.1f}; usage={out.get('usage')}")
    log("reply: " + json.dumps(out["choices"][0]["message"])[:300])

server.terminate()
try:
    server.wait(60)
except subprocess.TimeoutExpired:
    server.kill()
for f in sorted(RUN_DIR.glob("*.log"))[:3]:
    log(f"--- tail of {f.name}")
    print("\n".join(f.read_text(errors="replace").splitlines()[-60:]))
sys.exit(0 if ready else 1)
