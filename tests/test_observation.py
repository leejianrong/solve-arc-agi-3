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
