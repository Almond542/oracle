import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from dragon_oracle.config import OracleConfig


@dataclass
class DetectedTile:
    contour: np.ndarray
    bounding_rect: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]  # centroid (cx, cy)
    area: float
    aspect_ratio: float


class TileDetector:
    def __init__(self, config: OracleConfig):
        self._min_area = config.tile_min_area
        self._max_area = config.tile_max_area
        self._ar_range = config.tile_aspect_ratio_range
        self._blur_k = config.tile_blur_kernel
        self._min_rectangularity = 0.55

    def detect(self, image: np.ndarray) -> List[DetectedTile]:
        """Detect card tiles using combined Canny edge + HSV border detection.

        Two methods are merged to handle both face-up cards (strong color
        contrast, caught by Canny) and face-down cards (white/light borders
        on colored backgrounds, caught by HSV thresholding).
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Method 1: Canny edges on grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)
        edges = cv2.Canny(blurred, 20, 80)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours1, _ = cv2.findContours(
            closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Method 2: White/light card border detection (low sat + high val)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        border_mask = ((s_ch < 100) & (v_ch > 160)).astype(np.uint8) * 255
        dilated = cv2.dilate(border_mask, kernel, iterations=2)
        filled = dilated.copy()
        fill_contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(filled, fill_contours, -1, 255, cv2.FILLED)
        separated = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel, iterations=1)
        contours2, _ = cv2.findContours(
            separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Evaluate all contours from both methods
        tiles = []
        for contour in list(contours1) + list(contours2):
            tile = self._evaluate_contour(contour)
            if tile is not None:
                tiles.append(tile)

        tiles = self._remove_duplicates(tiles)
        return tiles

    def _evaluate_contour(self, contour: np.ndarray):
        """Check if a contour matches tile criteria."""
        area = cv2.contourArea(contour)
        if area < self._min_area or area > self._max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w < 15 or h < 15 or w > 80 or h > 80:
            return None

        aspect_ratio = w / h
        if not (self._ar_range[0] <= aspect_ratio <= self._ar_range[1]):
            return None

        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < self._min_rectangularity:
            return None

        center = self._compute_centroid(contour, x, y, w, h)

        return DetectedTile(
            contour=contour,
            bounding_rect=(x, y, w, h),
            center=center,
            area=area,
            aspect_ratio=aspect_ratio,
        )

    def _compute_centroid(
        self, contour: np.ndarray, x: int, y: int, w: int, h: int
    ) -> Tuple[float, float]:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (M["m10"] / M["m00"], M["m01"] / M["m00"])
        return (x + w / 2.0, y + h / 2.0)

    def _remove_duplicates(
        self, tiles: List[DetectedTile], distance_threshold: float = 25.0
    ) -> List[DetectedTile]:
        """Remove tiles whose centers are too close (keeps larger one)."""
        if not tiles:
            return tiles

        tiles.sort(key=lambda t: t.area, reverse=True)
        kept = []

        for tile in tiles:
            is_dup = False
            for existing in kept:
                dx = tile.center[0] - existing.center[0]
                dy = tile.center[1] - existing.center[1]
                if (dx * dx + dy * dy) < distance_threshold ** 2:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(tile)

        return kept
