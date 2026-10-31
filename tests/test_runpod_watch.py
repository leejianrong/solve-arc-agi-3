import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "runpod_watch", Path(__file__).parent.parent / "scripts/runpod_watch.py"
)
assert spec and spec.loader
rw = importlib.util.module_from_spec(spec)
sys.modules["runpod_watch"] = rw
spec.loader.exec_module(rw)


def st(**kw):
    base = {
        "elapsed": 100,
        "cost_per_hr": 1.69,
        "max_usd": 3.4,
        "pod_present": True,
        "lines_seen": 5,
        "secs_since_new_line": 10,
    }
    return rw.State(**{**base, **kw})


def test_healthy_continues():
    assert rw.decide(st()).action == "continue"


def test_pod_gone_finishes():
    assert rw.decide(st(pod_present=False)).action == "finish"


def test_cost_cap_aborts():
    assert rw.decide(st(elapsed=3600 * 3)).action == "abort"


def test_fatal_and_done_lines():
    assert rw.decide(st(new_lines=["FATAL: phase vllm exceeded"])).action == "abort"
    assert (
        rw.decide(st(new_lines=["x", "BOOT_CHECK_FAILED: no GPU visible"])).action
        == "abort"
    )
    assert rw.decide(st(new_lines=["DONE: job finished"])).action == "finish"


def test_no_output_after_start_aborts():
    assert rw.decide(st(lines_seen=0, elapsed=1000)).action == "abort"
    assert rw.decide(st(lines_seen=0, elapsed=100)).action == "continue"


def test_silence_aborts():
    assert rw.decide(st(secs_since_new_line=700)).action == "abort"
