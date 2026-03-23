import pytest
from dragon_oracle.tiles import TileDetector
from dragon_oracle.config import OracleConfig


@pytest.fixture
def detector():
    return TileDetector(OracleConfig())


def test_detect_tiles_finds_some(detector, sample_image):
    """Should detect at least some tiles in a game board image."""
    tiles = detector.detect(sample_image)
    assert len(tiles) >= 5, f"Expected >= 5 tiles, got {len(tiles)}"


def test_detect_tiles_reasonable_count(detector, sample_image):
    """Should not detect an unreasonable number of tiles."""
    tiles = detector.detect(sample_image)
    assert len(tiles) <= 60, f"Expected <= 60 tiles, got {len(tiles)}"


def test_all_images_detect_tiles(detector, all_test_images):
    """Every test image should produce at least some tile detections."""
    for name, img in all_test_images:
        tiles = detector.detect(img)
        assert len(tiles) >= 3, f"{name}: Expected >= 3 tiles, got {len(tiles)}"


def test_tile_properties(detector, sample_image):
    """Detected tiles should have valid properties."""
    tiles = detector.detect(sample_image)
    for tile in tiles:
        assert tile.area > 0
        assert tile.aspect_ratio > 0
        cx, cy = tile.center
        assert 0 <= cx <= sample_image.shape[1]
        assert 0 <= cy <= sample_image.shape[0]
