from solve_arc_agi_3.observation import (
    describe_key,
    describe_motion,
    object_motion,
    region_key,
)


def grid(
    width: int, height: int, blocks: dict[tuple[int, int], int]
) -> tuple[tuple[int, ...], ...]:
    rows = [[0] * width for _ in range(height)]
    for (x, y), c in blocks.items():
        rows[y][x] = c
    return tuple(map(tuple, rows))


def block(x: int, y: int, c: int = 5) -> dict[tuple[int, int], int]:
    return {(x + dx, y + dy): c for dx in range(2) for dy in range(2)}


def test_a_moved_block_is_reported_with_its_displacement() -> None:
    before, after = grid(20, 12, block(3, 4)), grid(20, 12, block(8, 4))
    pairs = object_motion(before, after)
    assert [(a.x0, b.x0) for a, b in pairs] == [(3, 8)]
    assert (
        describe_motion(pairs) == "color 5 size 4 moved (+5,+0) from (3, 4) to (8, 4)"
    )


def test_unmoved_objects_and_pure_recolours_are_not_motion() -> None:
    a = grid(20, 12, {**block(3, 4), **block(12, 2, 7)})
    assert object_motion(a, a) == []
    recoloured = grid(20, 12, {**block(3, 4, 9), **block(12, 2, 7)})
    assert object_motion(a, recoloured) == []


def test_identical_objects_are_matched_by_nearest_so_one_move_is_one_report() -> None:
    before = grid(40, 12, {**block(2, 2), **block(10, 2), **block(18, 2)})
    after = grid(40, 12, {**block(2, 2), **block(10, 2), **block(21, 2)})
    pairs = object_motion(before, after)
    assert [(a.x0, b.x0) for a, b in pairs] == [(18, 21)]


def test_a_growing_bar_on_the_bottom_row_keeps_one_identity() -> None:
    first = region_key((5, 63, 5, 63, 1), 64, 64)
    second = region_key((6, 63, 6, 63, 1), 64, 64)
    assert first == second == ("row", 63, 63)
    assert describe_key(first) == "rows y63-63"


def test_a_mid_board_change_is_keyed_by_its_exact_box() -> None:
    assert region_key((10, 20, 12, 22, 5), 64, 64) == ("box", 10, 20, 12, 22)
    assert describe_key(("box", 10, 20, 12, 22)) == "x10-12 y20-22"
    # a big region against the border is not a status strip
    assert region_key((0, 60, 63, 63, 200), 64, 64)[0] == "box"
