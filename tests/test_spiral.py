from dragon_oracle.spiral import spiral_order


def test_spiral_3x3_full():
    grid = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
    path = spiral_order(grid)
    expected = [
        (0, 0), (0, 1), (0, 2),
        (1, 2), (2, 2), (2, 1),
        (2, 0), (1, 0), (1, 1),
    ]
    assert path == expected


def test_spiral_3x3_with_holes():
    grid = [[1, None, 3],
            [None, 5, 6],
            [7, 8, None]]
    path = spiral_order(grid)
    # Same traversal order but skip None cells
    expected = [
        (0, 0), (0, 2),
        (1, 2), (2, 1),
        (2, 0), (1, 1),
    ]
    assert path == expected


def test_spiral_4x4_full():
    grid = [[i * 4 + j + 1 for j in range(4)] for i in range(4)]
    path = spiral_order(grid)
    expected = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (2, 3), (3, 3),
        (3, 2), (3, 1), (3, 0),
        (2, 0), (1, 0),
        (1, 1), (1, 2),
        (2, 2), (2, 1),
    ]
    assert path == expected


def test_spiral_empty():
    assert spiral_order([]) == []
    assert spiral_order([[]]) == []


def test_spiral_single():
    assert spiral_order([[1]]) == [(0, 0)]
    assert spiral_order([[None]]) == []


def test_spiral_counterclockwise():
    grid = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
    path_cw = spiral_order(grid, direction="clockwise")
    path_ccw = spiral_order(grid, direction="counterclockwise")
    # Both should visit all 9 cells but in different order
    assert len(path_cw) == 9
    assert len(path_ccw) == 9
    assert set(path_cw) == set(path_ccw)
    assert path_cw != path_ccw


def test_spiral_start_corners():
    grid = [[1, 2], [3, 4]]
    tl = spiral_order(grid, start_corner="top-left")
    tr = spiral_order(grid, start_corner="top-right")
    bl = spiral_order(grid, start_corner="bottom-left")
    br = spiral_order(grid, start_corner="bottom-right")

    # Each starts at its respective corner
    assert tl[0] == (0, 0)
    assert tr[0] == (0, 1)
    assert bl[0] == (1, 0)
    assert br[0] == (1, 1)
