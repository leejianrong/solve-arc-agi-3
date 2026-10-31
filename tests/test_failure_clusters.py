from typing import Any

from solve_arc_agi_3.failure_clusters import classify, report


def game(
    moves: list[tuple[str, int, bool]],
    avail: list[int],
    *,
    clear_at: int | None = None,
    call_ok: bool = True,
    fallback_every: int = 0,
    failed_every: int = 0,
) -> list[dict[str, Any]]:
    """moves: (name, cells changed, click?) per step."""
    records: list[dict[str, Any]] = [{"kind": "start", "levels": 0, "avail": avail}]
    for i, (name, changed, click) in enumerate(moves):
        move: dict[str, Any] = {"name": name}
        if click:
            move.update(x=i % 7, y=i % 5)
        step: dict[str, Any] = {
            "kind": "step",
            "step": i,
            "src": "model",
            "moves": [move],
            "levels": 1 if clear_at == i else 0,
            "change": {"n": changed},
            "call": {"ok": call_ok},
        }
        if failed_every and i % failed_every == 0:
            step["src"], step["call"] = "fallback", {"ok": False}
        elif fallback_every and i % fallback_every == 0:
            step["src"] = "fallback"
        records.append(step)
    return records


def cycle(n: int, names: list[str], changed: int) -> list[tuple[str, int, bool]]:
    return [
        (names[i % len(names)], changed if i % 2 == 0 else 0, False) for i in range(n)
    ]


FOUR = [f"ACTION{i}" for i in range(1, 5)]


def test_each_cluster_is_recognised_by_its_rule() -> None:
    cases = {
        "runtime": game(cycle(20, FOUR, 5), [1, 2, 3, 4], failed_every=3),
        "tool_syntax": game(cycle(20, FOUR, 5), [1, 2, 3, 4], fallback_every=3),
        "exploration": game([("ACTION1", 5, False)] * 40, [1, 2, 3, 4]),
        "memory": game(
            [(n, 5, False) for n in FOUR] + [("ACTION2", 0, False)] * 6, [1, 2, 3, 4]
        ),
        "goal": game(cycle(40, FOUR, 5), [1, 2, 3, 4]),
        "perception": game([(FOUR[i % 4], 0, False) for i in range(40)], [1, 2, 3, 4]),
    }
    for expected, records in cases.items():
        assert classify(expected, records).cluster == expected, expected


def test_a_click_game_that_keeps_clicking_the_same_few_cells_is_an_exploration_failure() -> (
    None
):
    moves = [("ACTION6", 5, True)] * 40
    records = game(moves, [6])
    for r in records[1:]:  # always the same cell
        r["moves"][0].update(x=1, y=1)
    assert classify("c", records).cluster == "exploration"


def test_a_cleared_game_counts_only_the_steps_up_to_the_first_clear() -> None:
    v = classify("g", game(cycle(30, FOUR, 5), [1, 2, 3, 4], clear_at=7))
    assert v.cleared and v.cluster == "cleared" and v.wasted == 8 and v.steps == 30


def test_the_report_ranks_clusters_by_actions_spent() -> None:
    from solve_arc_agi_3.failure_clusters import GameVerdict

    verdicts = [
        GameVerdict("aaaa-1", False, 10, "goal", 10, {}),
        GameVerdict("bbbb-1", False, 90, "exploration", 90, {}),
        GameVerdict("cccc-1", True, 40, "cleared", 12, {}),
    ]
    lines = report(verdicts).splitlines()
    assert lines[0].startswith("3 games, 1 with a cleared level, 140 actions")
    assert lines[4].startswith("| exploration | 1 | 90") and lines[5].startswith(
        "| goal | 1 | 10"
    )
