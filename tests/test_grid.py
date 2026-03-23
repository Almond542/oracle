import numpy as np
from dragon_oracle.grid import GridClusterer
from dragon_oracle.config import OracleConfig


def make_clusterer():
    config = OracleConfig()
    return GridClusterer(config)


def test_cluster_regular_grid():
    """Perfect 4x3 grid of centroids."""
    centers = []
    for row in range(3):
        for col in range(4):
            centers.append((100 + col * 80, 100 + row * 80))

    clusterer = make_clusterer()
    result = clusterer.cluster(centers)

    assert result.num_rows == 3
    assert result.num_cols == 4
    assert len(result.cells) == 12


def test_cluster_with_missing_cells():
    """Grid with some positions empty."""
    centers = [
        (100, 100), (180, 100), (260, 100),  # row 0: full
        (100, 180),             (260, 180),   # row 1: missing middle
        (100, 260), (180, 260), (260, 260),   # row 2: full
    ]
    clusterer = make_clusterer()
    result = clusterer.cluster(centers)

    assert result.num_rows == 3
    assert result.num_cols == 3
    assert len(result.cells) == 8
    # Middle cell of row 1 should be None
    assert result.grid_matrix[1][1] is None


def test_cluster_with_jitter():
    """Grid with slight position noise."""
    rng = np.random.default_rng(42)
    centers = []
    for row in range(4):
        for col in range(5):
            jx = rng.normal(0, 3)
            jy = rng.normal(0, 3)
            centers.append((100 + col * 80 + jx, 100 + row * 80 + jy))

    clusterer = make_clusterer()
    result = clusterer.cluster(centers)

    assert result.num_rows == 4
    assert result.num_cols == 5


def test_cluster_single_tile():
    centers = [(200, 300)]
    clusterer = make_clusterer()
    result = clusterer.cluster(centers)

    assert result.num_rows == 1
    assert result.num_cols == 1
    assert result.grid_matrix[0][0] == 0


def test_cluster_empty():
    clusterer = make_clusterer()
    result = clusterer.cluster([])
    assert result.num_rows == 0
    assert result.num_cols == 0
