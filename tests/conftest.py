import pytest
import cv2
import numpy as np
from pathlib import Path

TEST_IMAGE_DIR = Path(__file__).parent.parent / "test_images"


@pytest.fixture
def sample_image():
    """Load a single representative test image, resized to 720p."""
    img_path = sorted(TEST_IMAGE_DIR.glob("IMG_*.jpg"))[0]
    img = cv2.imread(str(img_path))
    return cv2.resize(img, (1280, 720))


@pytest.fixture
def all_test_images():
    """Load all test images, resized to 720p."""
    images = []
    for p in sorted(TEST_IMAGE_DIR.glob("IMG_*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            images.append((p.name, cv2.resize(img, (1280, 720))))
    return images
