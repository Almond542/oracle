import pytest
from dragon_oracle.tiles import TileDetector
from dragon_oracle.grid import GridClusterer
from dragon_oracle.spiral import spiral_order
from dragon_oracle.overlay import OverlayRenderer
from dragon_oracle.config import OracleConfig


@pytest.fixture
def config():
    return OracleConfig()


def test_end_to_end_no_aruco(config, sample_image):
    """Full pipeline on a static image without ArUco markers."""
    detector = TileDetector(config)
    clusterer = GridClusterer(config)
    renderer = OverlayRenderer(config)

    # Detect tiles
    tiles = detector.detect(sample_image)
    assert len(tiles) > 0

    # Cluster into grid
    centers = [t.center for t in tiles]
    grid_result = clusterer.cluster(centers)
    assert grid_result.num_rows > 0
    assert grid_result.num_cols > 0

    # Spiral traversal
    path = spiral_order(
        grid_result.grid_matrix,
        config.spiral_direction,
        config.spiral_start_corner,
    )
    assert len(path) > 0

    # Render overlay
    output = renderer.render(sample_image, path, grid_result)
    assert output.shape == sample_image.shape


def test_pipeline_all_images(config, all_test_images):
    """Pipeline should not crash on any test image."""
    detector = TileDetector(config)
    clusterer = GridClusterer(config)
    renderer = OverlayRenderer(config)

    for name, img in all_test_images:
        tiles = detector.detect(img)
        if tiles:
            centers = [t.center for t in tiles]
            grid_result = clusterer.cluster(centers)
            path = spiral_order(grid_result.grid_matrix)
            output = renderer.render(img, path, grid_result)
            assert output.shape == img.shape, f"Shape mismatch on {name}"
