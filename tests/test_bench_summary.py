import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "bench_summary", ROOT / "scripts/bench_summary.py"
)
assert spec and spec.loader
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

A = ROOT / "tests/fixtures/trace_sample"
B = ROOT / "tests/fixtures/trace_sample_b"


def step(name: str, n: int, levels: int = 0, src: str = "model", **xy: int) -> dict:
    return {
        "kind": "step",
        "src": src,
        "moves": [{"name": name, **xy}],
        "levels": levels,
        "change": {"n": n},
    }


def test_game_stats_counts() -> None:
    recs = [
        {"kind": "start", "levels": 0, "avail": [1, 6]},
        step("ACTION1", 0),
        step("ACTION1", 0, src="fallback"),
        step("ACTION1", 3),
        step("ACTION6", 0, x=1, y=2),
        step("ACTION6", 5, levels=1, x=3, y=4, src="probe"),
    ]
    s = bs.game_stats(recs)
    assert s["steps"] == 5 and s["cleared"] == 1 and s["first_clear"] == 5
    assert s["idle"] == pytest.approx(60.0)
    assert s["clicks"] == 2 and s["repeat"] == 2
    assert s["fallback"] == pytest.approx(20.0) and s["probe"] == pytest.approx(20.0)
    assert s["override"] == 0 and s["cluster"] == "cleared"


def test_no_clear_gives_dash() -> None:
    s = bs.game_stats(
        [{"kind": "start", "levels": 0, "avail": [1]}, step("ACTION1", 1)]
    )
    assert s["first_clear"] is None
    assert bs._cell([None]) == "-"


def test_cell_mean_min_max() -> None:
    assert bs._cell([2, 4]) == "3 (2-4)"
    assert bs._cell([2, None]) == "2 (2-2) n=1/2"


def test_summarize_fixtures_with_repeats() -> None:
    out = bs.summarize({"x": bs.load_runs(A) + bs.load_runs(B), "y": bs.load_runs(B)})
    assert "== x: 2 run(s), 1 game(s)" in out and "== y: 1 run(s)" in out
    assert "lp85" in out and "ALL" in out and "cluster" in out


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    bs.main([f"a={A}", f"a={B}"])
    assert "== a: 2 run(s)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        bs.main(["nolabel"])
