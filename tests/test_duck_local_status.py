import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_dir = Path(__file__).resolve().parents[1] / "scripts" / "duck_local"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _dir / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sh = _load("status_header")
duck_local = _load("duck_local")


def board(r: int, c: int, bar: int = 6) -> list[str]:
    """8x8 board: background B, a 2x2 red object at (r, c), a status bar on the bottom row."""
    g = [["B"] * 8 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            g[r + i][c + j] = "R"
    for j in range(bar):
        g[7][j] = "Y"
    return ["".join(row) for row in g]


def test_moved_then_blocked_then_dead() -> None:
    t = sh.StatusTracker()
    t.update(board(2, 2), "RIGHT", board(2, 3), 0)
    text = t.render(["UP", "DOWN", "LEFT", "RIGHT", "RESET"])
    assert "RIGHT MOVED the object by (row +0, col +1)" in text
    assert "rows 2-3 cols 3-4" in text
    assert "Not tried yet in this game: UP, DOWN, LEFT." in text
    t.update(board(2, 3), "UP", board(2, 3), 0)
    assert "UP was BLOCKED (nothing changed)" in t.render()
    assert "do not repeat" not in t.render()
    t.update(board(2, 3), "UP", board(2, 3), 0)
    text = t.render()
    assert "do not repeat them: UP." in text
    assert "WARNING: your last 2 actions changed nothing" in text


def test_counter_only_change_is_blocked() -> None:
    t = sh.StatusTracker()
    t.update(board(2, 2, 6), "LEFT", board(2, 2, 5), 0)
    assert "BLOCKED (only a counter or timer changed)" in t.render()


def test_working_click_and_dead_click() -> None:
    t = sh.StatusTracker()
    a, b = board(2, 2), board(4, 4)
    t.update(a, "MOUSE(row=1, col=1)", b, 0)
    assert "Clicks that changed the board: MOUSE(1,1) x1" in t.render()
    for _ in range(3):
        t.update(b, "MOUSE(5,5)", b, 0)
    assert "MOUSE(5,5)" in t.render().split("do not repeat them:")[1]


def test_loop_warning_on_cycling_boards() -> None:
    t = sh.StatusTracker()
    a, b = board(2, 2), board(2, 3)
    for _ in range(3):
        t.update(a, "RIGHT", b, 0)
        t.update(b, "LEFT", a, 0)
    assert "keep returning to the same board" in t.render()


def test_level_change_resets_mover() -> None:
    t = sh.StatusTracker()
    t.update(board(2, 2), "RIGHT", board(2, 3), 0)
    t.update(board(2, 3), "DOWN", board(0, 0), 1)
    text = t.render()
    assert "cleared the level" in text and "No object has been seen moving" in text


def test_empty_before_first_action() -> None:
    assert sh.StatusTracker().render(["UP"]) == ""


def _entry(action: str, grid: list[str], step: int) -> SimpleNamespace:
    return SimpleNamespace(
        action=action, frame=SimpleNamespace(grid=grid, step=step, level=1)
    )


def test_status_lines_is_incremental_and_rebuilds() -> None:
    owner = SimpleNamespace()
    es = [_entry("", board(2, 2), 0), _entry("RIGHT", board(2, 3), 1)]
    assert "RIGHT MOVED" in sh.status_lines(owner, es, es[-1].frame, ["UP"])[0]
    es.append(_entry("UP", board(2, 3), 2))
    assert "UP was BLOCKED" in sh.status_lines(owner, es, es[-1].frame, ["UP"])[0]
    assert owner._status_header_state[0].effects["RIGHT"] == [1, 1]
    assert (
        sh.status_lines(owner, es[:1], es[0].frame, None) == []
    )  # history shrank: rebuilt


SRC = (
    "    def f(self):\n"
    "        lines = ['a']\n"
    "        lines.extend(self._prepare_auto_diff(current_frame, previous_step_summary))\n"
    "        return lines\n"
)


def test_hook_applies_once_and_flag_off_is_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patched = duck_local.apply_status_hook(SRC)
    assert patched.count("duck_local status hook") == 1
    with pytest.raises(ValueError):
        duck_local.apply_status_hook("nothing here")
    with pytest.raises(ValueError):
        duck_local.apply_status_hook(SRC + SRC)

    def run(src: str, flag: str | None) -> list[str]:
        if flag is None:
            monkeypatch.delenv("ARC3_STATUS_HEADER", raising=False)
        else:
            monkeypatch.setenv("ARC3_STATUS_HEADER", flag)
        owner = SimpleNamespace(_prepare_auto_diff=lambda *_: ["diff"])
        ns = {
            "os": __import__("os"),
            "current_frame": None,
            "previous_step_summary": None,
        }
        ns.update(history_entries=[], valid_actions=[], _normalize_valid_actions=list)
        mod = {"inference": SimpleNamespace(agent=SimpleNamespace(status_header=sh))}
        monkeypatch.setitem(__import__("sys").modules, "inference", mod["inference"])
        monkeypatch.setitem(
            __import__("sys").modules, "inference.agent", mod["inference"].agent
        )
        monkeypatch.setitem(
            __import__("sys").modules, "inference.agent.status_header", sh
        )
        exec("class C:\n" + src, ns)  # noqa: S102
        return ns["C"].f(owner)

    base = run(SRC, None)
    assert run(patched, None) == base == run(patched, "0")
    assert run(patched, "1") == base  # on, but no actions yet: still nothing added
