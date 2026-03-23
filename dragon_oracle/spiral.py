from typing import List, Tuple, Optional


def spiral_order(
    grid: List[List[Optional[int]]],
    direction: str = "clockwise",
    start_corner: str = "top-left",
) -> List[Tuple[int, int]]:
    """Generate spiral traversal of a grid, skipping None cells.

    Args:
        grid: 2D list where grid[row][col] is a tile index or None.
        direction: "clockwise" or "counterclockwise".
        start_corner: "top-left", "top-right", "bottom-left", "bottom-right".

    Returns:
        List of (row, col) tuples in spiral order, only for non-None cells.
    """
    if not grid or not grid[0]:
        return []

    rows = len(grid)
    cols = len(grid[0])

    # Transform grid based on start corner so we always spiral from top-left
    transformed = _transform_grid(grid, rows, cols, start_corner, direction)
    t_rows = len(transformed)
    t_cols = len(transformed[0]) if transformed else 0

    # Perform clockwise spiral from top-left on the transformed grid
    raw_path = _clockwise_spiral(transformed, t_rows, t_cols)

    # Map coordinates back to original grid
    result = []
    for r, c in raw_path:
        orig_r, orig_c = _inverse_transform(
            r, c, rows, cols, start_corner, direction
        )
        if grid[orig_r][orig_c] is not None:
            result.append((orig_r, orig_c))

    return result


def _clockwise_spiral(
    grid: List[List[Optional[int]]], rows: int, cols: int
) -> List[Tuple[int, int]]:
    """Standard clockwise spiral from top-left."""
    result = []
    top, bottom, left, right = 0, rows - 1, 0, cols - 1

    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            result.append((top, col))
        top += 1

        for row in range(top, bottom + 1):
            result.append((row, right))
        right -= 1

        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append((bottom, col))
            bottom -= 1

        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append((row, left))
            left += 1

    return result


def _transform_grid(
    grid: List[List[Optional[int]]],
    rows: int,
    cols: int,
    start_corner: str,
    direction: str,
) -> List[List[Optional[int]]]:
    """Transform grid so that the desired spiral becomes a clockwise top-left spiral."""
    g = [row[:] for row in grid]  # copy

    # Handle start corner by flipping
    if start_corner == "top-right":
        g = [row[::-1] for row in g]
    elif start_corner == "bottom-left":
        g = g[::-1]
    elif start_corner == "bottom-right":
        g = [row[::-1] for row in g[::-1]]

    # Handle counterclockwise by transposing
    if direction == "counterclockwise":
        t_rows = len(g)
        t_cols = len(g[0]) if g else 0
        g = [[g[r][c] for r in range(t_rows)] for c in range(t_cols)]

    return g


def _inverse_transform(
    r: int,
    c: int,
    orig_rows: int,
    orig_cols: int,
    start_corner: str,
    direction: str,
) -> Tuple[int, int]:
    """Map transformed coordinates back to original grid coordinates."""
    # Undo counterclockwise transpose
    if direction == "counterclockwise":
        r, c = c, r

    # Undo start corner flips
    if start_corner == "top-right":
        c = orig_cols - 1 - c
    elif start_corner == "bottom-left":
        r = orig_rows - 1 - r
    elif start_corner == "bottom-right":
        r = orig_rows - 1 - r
        c = orig_cols - 1 - c

    return r, c
