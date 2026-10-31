#!/usr/bin/env python3
"""Driver-side watcher for the RunPod lab pod: the second layer behind the pod's own scripts/lab_obs.sh.

The pod handles what it can see from inside (phase deadlines, fatal log lines, silence). This handles what only an
outside view can: a container that never starts, a pod that died without a trace, a cost cap, and collecting the
evidence. Every poll appends the new log lines to a file under --out, so nothing is lost when the pod disappears.

    scripts/runpod_launch_lab.sh watch --max-usd 3.4

``decide`` is a pure function (tested in tests/test_runpod_watch.py); the rest is polling and side effects.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

FATAL = re.compile(
    r"^FATAL: |BOOT_CHECK_FAILED|No space left on device|CUDA out of memory"
)
CODE = re.compile(r"RP_ARTIFACT_CODE=(\S+)")


@dataclass
class State:
    elapsed: float  # seconds since the pod was created
    cost_per_hr: float
    max_usd: float
    pod_present: bool
    lines_seen: int  # container log lines received so far
    secs_since_new_line: float  # no new container line for this long
    new_lines: list[str] = field(default_factory=list)
    startup_grace: float = 900  # a container should print something within this long
    silence_limit: float = 600  # the pod prints a heartbeat every 30 s; this long without one is a dead pod


@dataclass(frozen=True)
class Decision:
    action: str  # continue | finish | abort
    reason: str = ""


def decide(s: State) -> Decision:
    if not s.pod_present:
        return Decision("finish", "pod is gone")
    spent = s.cost_per_hr * s.elapsed / 3600
    if spent > s.max_usd:
        return Decision(
            "abort", f"estimated spend ${spent:.2f} is over the ${s.max_usd:.2f} cap"
        )
    for line in s.new_lines:
        if line.startswith("DONE:"):
            return Decision("finish", "job reported DONE")
        if FATAL.search(line):
            return Decision("abort", f"fatal line: {line[:160]}")
    if s.lines_seen == 0 and s.elapsed > s.startup_grace:
        return Decision(
            "abort",
            f"no container output after {s.startup_grace:.0f}s (image pull or start stalled)",
        )
    if s.lines_seen > 0 and s.secs_since_new_line > s.silence_limit:
        return Decision(
            "abort",
            f"no log line for {s.secs_since_new_line:.0f}s (heartbeats stopped)",
        )
    return Decision("continue")


def _run(cmd: list[str], timeout: int = 60) -> str:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return r.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def find_pod(name: str) -> dict[str, object] | None:
    try:
        pods = json.loads(_run(["runpodctl", "pod", "list"]) or "[]")
    except json.JSONDecodeError:
        return None
    return next((p for p in pods if p.get("name") == name), None)


def fetch_logs(pod_id: str) -> list[str]:
    out = []
    for raw in _run(["runpodctl", "pod", "logs", pod_id]).splitlines():
        try:
            out.append(str(json.loads(raw).get("line", "")))
        except json.JSONDecodeError:
            out.append(raw)
    return out


def receive(code: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        ["runpodctl", "receive", code], timeout=180
    )  # writes into the cwd, so move it afterwards
    for f in Path.cwd().glob("diag.tgz"):
        f.replace(dest / f.name)
        return True
    return False


def delete_and_verify(name: str) -> bool:
    for _ in range(3):
        pod = find_pod(name)
        if pod is None:
            return True
        _run(["runpodctl", "pod", "delete", str(pod["id"])])
        time.sleep(5)
    return find_pod(name) is None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-name", default="arc3-agent-lab")
    ap.add_argument(
        "--max-usd", type=float, required=True, help="hard cap on estimated spend"
    )
    ap.add_argument("--poll", type=float, default=20)
    ap.add_argument("--out", type=Path, default=Path("build/runpod-watch"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log_file = a.out / f"{stamp}.log"
    t0 = time.time()
    seen = 0
    last_new = t0
    code: str | None = None
    verdict = Decision("continue")
    while True:
        pod = find_pod(a.pod_name)
        lines = fetch_logs(str(pod["id"])) if pod else []
        new = lines[seen:] if len(lines) >= seen else lines
        if new:
            last_new = time.time()
            seen = len(lines)
            with log_file.open("a") as f:
                f.write("\n".join(new) + "\n")
            for ln in new:
                print(ln[:240], flush=True)
                m = CODE.search(ln)
                if m and m.group(1) not in ("send-failed", "no-runpodctl"):
                    code = m.group(1)
        verdict = decide(
            State(
                elapsed=time.time() - t0,
                cost_per_hr=float(pod.get("costPerHr", 0) or 0) if pod else 0.0,
                max_usd=a.max_usd,
                pod_present=pod is not None,
                lines_seen=seen,
                secs_since_new_line=time.time() - last_new,
                new_lines=new,
            )
        )
        if verdict.action != "continue":
            break
        time.sleep(a.poll)
    print(f"WATCH: {verdict.action}: {verdict.reason}", flush=True)
    if code is None and verdict.action == "abort":
        time.sleep(
            90
        )  # the pod ships its evidence on its own failures; give it the chance
        for ln in fetch_logs(str(pod["id"])) if find_pod(a.pod_name) and pod else []:
            m = CODE.search(ln)
            if m:
                code = m.group(1)
    if code and receive(code, a.out / stamp):
        print(f"WATCH: evidence saved under {a.out / stamp}", flush=True)
    ok = delete_and_verify(a.pod_name)
    print(
        "WATCH: pod deleted and verified gone"
        if ok
        else "WATCH: WARNING pod still present, delete by hand",
        flush=True,
    )
    print(f"WATCH: full log at {log_file}", flush=True)
    return 0 if ok and verdict.action == "finish" else 1


if __name__ == "__main__":
    sys.exit(main())
