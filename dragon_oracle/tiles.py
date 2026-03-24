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
        self._min_rectangularity = config.tile_min_rectangularity
        self._min_dim = config.tile_min_dimension
        self._max_dim = config.tile_max_dimension
        self._dup_distance = config.tile_duplicate_distance
        self._canny_low = config.tile_canny_low
        self._canny_high = config.tile_canny_high
        self._hsv_sat_max = config.tile_hsv_sat_max
        self._hsv_val_min = config.tile_hsv_val_min
        self._morph_close_iters = config.tile_morph_close_iterations

    def detect(self, image: np.ndarray) -> List[DetectedTile]:
        """Detect card tiles using combined Canny edge + HSV border detection.

        Two methods are merged to handle both face-up cards (strong color
        contrast, caught by Canny) and face-down cards (white/light borders
        on colored backgrounds, caught by HSV thresholding).
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Method 1: Canny edges on grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, d=self._blur_k, sigmaColor=50, sigmaSpace=50)
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_low = max(10, int(otsu_thresh * 0.5))
        canny_high = max(30, int(otsu_thresh * 1.0))
        edges = cv2.Canny(blurred, canny_low, canny_high)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=self._morph_close_iters)
        contours1, _ = cv2.findContours(
            closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Method 2: White/light card border detection (low sat + high val)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        val_threshold = max(self._hsv_val_min - 40, int(np.percentile(v_ch, 75)))
        sat_threshold = min(self._hsv_sat_max + 30, int(np.percentile(s_ch, 60)))
        border_mask = ((s_ch < sat_threshold) & (v_ch > val_threshold)).astype(np.uint8) * 255
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
        tiles = self._filter_by_median_size(tiles)
        return tiles

    def _evaluate_contour(self, contour: np.ndarray):
        """Check if a contour matches tile criteria."""
        area = cv2.contourArea(contour)
        if area < self._min_area or area > self._max_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w < self._min_dim or h < self._min_dim or w > self._max_dim or h > self._max_dim:
            return None

        aspect_ratio = w / h
        if not (self._ar_range[0] <= aspect_ratio <= self._ar_range[1]):
            return None

        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < self._min_rectangularity:
            return None

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if not (4 <= len(approx) <= 10):
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

    def _remove_duplicates(self, tiles: List[DetectedTile]) -> List[DetectedTile]:
        """Remove duplicate tiles using IoU overlap and center distance."""
        if not tiles:
            return tiles

        tiles.sort(key=lambda t: t.area, reverse=True)
        kept = []

        for tile in tiles:
            is_dup = False
            for existing in kept:
                # Check IoU overlap
                if self._compute_iou(tile.bounding_rect, existing.bounding_rect) > 0.3:
                    is_dup = True
                    break
                # Fallback: center distance check
                dx = tile.center[0] - existing.center[0]
                dy = tile.center[1] - existing.center[1]
                if (dx * dx + dy * dy) < self._dup_distance ** 2:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(tile)

        return kept

    @staticmethod
    def _compute_iou(rect1, rect2) -> float:
        """Compute Intersection over Union of two bounding rects (x, y, w, h)."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        xi = max(x1, x2)
        yi = max(y1, y2)
        xa = min(x1 + w1, x2 + w2)
        ya = min(y1 + h1, y2 + h2)
        inter = max(0, xa - xi) * max(0, ya - yi)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def _filter_by_median_size(self, tiles: List[DetectedTile]) -> List[DetectedTile]:
        """Remove outlier tiles whose area is far from the median."""
        if len(tiles) < 5:
            return tiles
        areas = [t.area for t in tiles]
        median_area = float(np.median(areas))
        return [t for t in tiles if 0.5 * median_area <= t.area <= 2.0 * median_area]
