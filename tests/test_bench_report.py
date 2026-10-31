import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "bench_report", ROOT / "scripts/bench_report.py"
)
assert spec and spec.loader
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)

A = ROOT / "tests/fixtures/trace_sample"
B = ROOT / "tests/fixtures/trace_sample_b"


def test_report_and_charts(tmp_path: Path) -> None:
    out = tmp_path / "study"
    br.main([f"a={A}", f"b={B}", "--out", str(out), "--title", "Fixture study"])
    md = (out / "report.md").read_text()
    for needle in ("Fixture study", "a", "b", "## Findings", "## Limits", "lp85"):
        assert needle in md
    assert "== a:" in md and "== b:" in md
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) >= 5
    for p in pngs:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 2000
        assert p.name in md
