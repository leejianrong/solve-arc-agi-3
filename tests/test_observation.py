from solve_arc_agi_3.observation import (
    ascii_grid,
    background_color,
    changed_count,
    describe_change,
    describe_objects,
    segment,
)

GRID = (
    (0, 0, 0, 0, 0, 0),
    (0, 3, 3, 0, 0, 5),
    (0, 3, 3, 0, 0, 0),
    (0, 0, 0, 0, 7, 0),
)


def test_ascii_grid_is_one_hex_row_per_line() -> None:
    assert ascii_grid(((0, 10, 15),)) == "0af"


def test_segmentation_finds_regions_with_centres_in_reading_order() -> None:
    regions = segment(GRID)
    assert [(r.color, r.size) for r in regions] == [(3, 4), (5, 1), (7, 1)]
    assert regions[0].center == (1, 1) and regions[1].center == (5, 1)
    assert background_color(GRID) == 0


def test_same_colour_regions_that_do_not_touch_are_separate_objects() -> None:
    grid = ((1, 0, 1), (0, 0, 0))
    assert len(segment(grid)) == 2


def test_object_list_is_capped_and_says_how_many_were_dropped() -> None:
    grid = tuple(
        tuple(1 if (x + y) % 2 == 0 else 0 for x in range(20)) for y in range(10)
    )
    text = describe_objects(grid, limit=5)
    assert text.count("\n- color") == 5 and "smaller objects not listed" in text


def test_change_description_names_where_and_what() -> None:
    after = tuple(
        tuple(9 if (x, y) == (1, 1) else c for x, c in enumerate(row))
        for y, row in enumerate(GRID)
    )
    assert changed_count(GRID, after) == 1
    assert (
        describe_change(GRID, after) == "1 cells changed: 1 cells at x1-1 y1-1 (3->9)"
    )
    assert describe_change(GRID, GRID) == "nothing changed"
    assert describe_change(None, GRID) == "no earlier frame"
    assert describe_change(GRID, GRID[:2]) == "the grid size changed"


def test_encoding_is_deterministic() -> None:
    assert describe_objects(GRID) == describe_objects(GRID)


def _bar(
    filled: int, width: int = 20, thickness: int = 1
) -> tuple[tuple[int, ...], ...]:
    rows = [[0] * width for _ in range(width)]
    for y in range(width - thickness, width):
        for x in range(filled):
            rows[y][x] = 5
    rows[3][3] = 2  # an unrelated object
    return tuple(tuple(r) for r in rows)


def test_shrinking_edge_bar_is_a_counter_on_first_change() -> None:
    from solve_arc_agi_3.observation import counter_strips

    assert counter_strips(_bar(10), _bar(9)) == [(9, 19, 9, 19)]
    assert counter_strips(_bar(10, thickness=2), _bar(9, thickness=2))  # 2 thick
    assert counter_strips(_bar(5), _bar(6))  # filling


def test_non_counter_changes_are_not_flagged() -> None:
    from solve_arc_agi_3.observation import counter_strips

    thick = counter_strips(_bar(10, thickness=3), _bar(9, thickness=3))
    assert thick == []
    moved = list(map(list, _bar(10)))
    moved[3][3], moved[8][8] = 0, 2  # interior object moved, not on the edge
    assert counter_strips(_bar(10), tuple(map(tuple, moved))) == []


def test_object_list_labels_counter() -> None:
    from solve_arc_agi_3.observation import counter_strips, describe_objects

    after = _bar(9)
    text = describe_objects(after, counters=counter_strips(_bar(10), after))
    bar_line = next(ln for ln in text.splitlines() if "color 5" in ln)
    assert bar_line.endswith("(counter, not a goal)")
    other = next(ln for ln in text.splitlines() if "color 2" in ln)
    assert "counter" not in other
    assert "counter" not in describe_objects(after)
