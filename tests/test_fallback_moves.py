from __future__ import annotations

from solve_arc_agi_3.agent import Observation
from solve_arc_agi_3.kaggle_agent import fallback_decision

GRID = tuple(
    tuple(1 if 4 <= x <= 6 and 4 <= y <= 6 else 0 for x in range(20)) for y in range(20)
)


def obs(avail: tuple[int, ...] = (1, 2, 6)) -> Observation:
    return Observation(
        game_id="g",
        grid=GRID,
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=3,
        available_actions=avail,
    )


class Core:
    def __init__(
        self, dead: list[str] | None = None, effects=None, clicks=None
    ) -> None:
        self._dead = dead or []
        self._effects = effects or {}
        self._clicks = clicks or []
        if dead is not None:
            self.dead_moves = lambda o: list(self._dead)


def label(d) -> str:
    return d.name + (f" at x={d.data['x']},y={d.data['y']}" if d.data else "")


def test_prefers_untried_action() -> None:
    core = Core(effects={"ACTION1": [2, 1]})
    d = fallback_decision((1, 2, 6), 0, core, obs())
    assert d.name == "ACTION2"


def test_untried_action_skipped_when_dead() -> None:
    core = Core(dead=["ACTION2"], effects={"ACTION1": [1, 1]})
    d = fallback_decision((1, 2, 6), 0, core, obs())
    assert d.name == "ACTION6"  # untried object centre instead


def test_untried_object_centre_then_never_dead() -> None:
    core = Core(dead=[], effects={"ACTION1": [1, 1], "ACTION2": [1, 1]})
    d = fallback_decision((1, 2, 6), 0, core, obs())
    assert (d.name, d.data) == ("ACTION6", {"x": 5, "y": 5})
    core2 = Core(dead=["ACTION6 at x=5,y=5"], effects=core._effects)
    d2 = fallback_decision((1, 2, 6), 0, core2, obs())
    assert label(d2) != "ACTION6 at x=5,y=5"


def test_cycle_never_returns_dead_move() -> None:
    core = Core(
        dead=["ACTION1"],
        effects={"ACTION1": [1, 0], "ACTION2": [1, 1]},
        clicks=[(5, 5)],
    )
    for step in range(30):
        assert fallback_decision((1, 2), step, core, obs((1, 2))).name == "ACTION2"


def test_all_dead_still_returns_something() -> None:
    core = Core(dead=["ACTION1"], effects={"ACTION1": [1, 0]})
    assert fallback_decision((1,), 3, core, obs((1,))).name == "ACTION1"


def test_core_without_dead_moves_accessor() -> None:
    core = Core(effects={"ACTION1": [1, 1], "ACTION2": [1, 1]}, clicks=[(5, 5)])
    d = fallback_decision((1, 2, 6), 4, core, obs())
    assert d.source == "fallback"
