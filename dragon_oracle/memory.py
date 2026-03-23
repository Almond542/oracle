import cv2
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from dragon_oracle.config import OracleConfig
from dragon_oracle.tiles import DetectedTile
from dragon_oracle.grid import GridClusterer


def extract_tile_image(
    frame: np.ndarray, tile: DetectedTile, target_size: Tuple[int, int]
) -> np.ndarray:
    """Crop a tile from the frame and resize to target_size (w, h)."""
    x, y, w, h = tile.bounding_rect
    margin = 2
    x1 = max(0, x + margin)
    y1 = max(0, y + margin)
    x2 = min(frame.shape[1], x + w - margin)
    y2 = min(frame.shape[0], y + h - margin)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)


class CardClassifier:
    """Classifies a cropped tile image as face-up or face-down."""

    def __init__(self, config: OracleConfig):
        self._hue_lo, self._hue_hi = config.facedown_hue_range
        self._sat_lo, self._sat_hi = config.facedown_sat_range
        self._val_lo, self._val_hi = config.facedown_val_range
        self._threshold = config.facedown_ratio_threshold

    def is_facedown(self, tile_image: np.ndarray) -> bool:
        """Returns True if tile is face-down (red/orange/pink sun pattern).

        Uses only the center 60% of the tile to avoid background bleed.
        """
        th, tw = tile_image.shape[:2]
        if th < 10 or tw < 10:
            return True

        my = th // 5
        mx = tw // 5
        center = tile_image[my:th - my, mx:tw - mx]

        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        lower = np.array([self._hue_lo, self._sat_lo, self._val_lo])
        upper = np.array([self._hue_hi, self._sat_hi, self._val_hi])
        mask = cv2.inRange(hsv, lower, upper)

        total = mask.shape[0] * mask.shape[1]
        matched = cv2.countNonZero(mask)
        ratio = matched / total

        return ratio >= self._threshold


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_game_cards(
    tiles: List[DetectedTile], frame_w: int, frame_h: int
) -> List[DetectedTile]:
    """Remove non-card detections: UI buttons, moves counter, background.

    Stages:
    1. Size filter — keep tiles within 60-140% of median card dimensions
    2. Aspect ratio consistency — reject tiles >25% off median AR
    3. Board region — reject outliers far from the main tile cluster
    """
    if len(tiles) < 3:
        return tiles

    # --- Stage 1: Size filter ---
    widths = [t.bounding_rect[2] for t in tiles]
    heights = [t.bounding_rect[3] for t in tiles]
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))

    size_ok = []
    for t in tiles:
        _, _, w, h = t.bounding_rect
        if 0.6 * med_w <= w <= 1.4 * med_w and 0.6 * med_h <= h <= 1.4 * med_h:
            size_ok.append(t)

    if len(size_ok) < 3:
        return size_ok

    # --- Stage 2: Aspect ratio consistency ---
    ars = [t.bounding_rect[2] / t.bounding_rect[3] for t in size_ok]
    med_ar = float(np.median(ars))
    ar_ok = []
    for t in size_ok:
        tile_ar = t.bounding_rect[2] / t.bounding_rect[3]
        if med_ar > 0 and abs(tile_ar - med_ar) / med_ar < 0.25:
            ar_ok.append(t)

    if len(ar_ok) < 3:
        ar_ok = size_ok  # fallback if AR filter is too aggressive

    # --- Stage 3: Board region filter ---
    all_cx = [t.center[0] for t in ar_ok]
    all_cy = [t.center[1] for t in ar_ok]
    min_x, max_x = min(all_cx), max(all_cx)
    min_y, max_y = min(all_cy), max(all_cy)
    board_w = max_x - min_x
    board_h = max_y - min_y

    margin_x = board_w * 0.10 if board_w > 0 else med_w
    margin_y = board_h * 0.10 if board_h > 0 else med_h

    filtered = []
    for t in ar_ok:
        cx, cy = t.center
        if (min_x - margin_x <= cx <= max_x + margin_x
                and min_y - margin_y <= cy <= max_y + margin_y):
            filtered.append(t)

    return filtered if len(filtered) >= 3 else ar_ok


# ---------------------------------------------------------------------------
# MemoryBoard — the main class
# ---------------------------------------------------------------------------

class MemoryBoard:
    """Fixed 6x10 grid that accumulates revealed card images across captures.

    First capture: uses GridClusterer to establish anchor row/col positions.
    Subsequent captures: snaps each tile to the nearest anchor position.
    Cards are stored once and never overwritten.
    """

    def __init__(self, config: OracleConfig):
        self._card_w, self._card_h = config.card_store_size
        self._rows = config.board_rows
        self._cols = config.board_cols
        self._tol_factor = config.anchor_tolerance_factor
        self._config = config
        self._padding = 6
        self._bg_color = (40, 40, 50)
        self._empty_color = (70, 70, 80)

        # Persistent state
        self._cards: Dict[Tuple[int, int], np.ndarray] = {}
        self._known_positions: Set[Tuple[int, int]] = set()
        self._capture_count: int = 0

        # Anchor — set once on first capture, reused for all subsequent
        self._anchor_rows: Optional[List[float]] = None  # Y positions
        self._anchor_cols: Optional[List[float]] = None  # X positions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_capture(
        self,
        frame: np.ndarray,
        tiles: List[DetectedTile],
        classifier: CardClassifier,
    ) -> np.ndarray:
        """Process a new photo. Only adds newly-revealed face-up cards."""
        self._capture_count += 1
        fh, fw = frame.shape[:2]

        # Step 1: Filter UI junk
        cards_only = _filter_game_cards(tiles, fw, fh)

        if not cards_only:
            print(f"  Capture #{self._capture_count}: no game cards found after filtering")
            return self.build_grid_image()

        # Step 2: Map tiles to fixed grid
        if self._anchor_rows is None:
            grid_map = self._establish_anchor(cards_only)
        else:
            grid_map = self._snap_to_anchor(cards_only)
            # If less than 30% of filtered tiles mapped, the camera likely
            # moved significantly — re-establish anchor from this frame
            if len(grid_map) < len(cards_only) * 0.3 and len(cards_only) >= 5:
                print(f"  Re-anchoring ({len(grid_map)}/{len(cards_only)} mapped)")
                grid_map = self._establish_anchor(cards_only)

        # Step 3: Classify and store new face-up cards
        new_reveals = 0
        face_down = 0

        for (row, col), tile in grid_map.items():
            self._known_positions.add((row, col))

            # Never overwrite an existing stored card
            if (row, col) in self._cards:
                continue

            tile_img = extract_tile_image(
                frame, tile, (self._card_w, self._card_h)
            )

            if classifier.is_facedown(tile_img):
                face_down += 1
            else:
                self._cards[(row, col)] = tile_img.copy()
                new_reveals += 1

        print(
            f"  Capture #{self._capture_count}: "
            f"{len(tiles)} detected, "
            f"{len(cards_only)} after filter, "
            f"{len(grid_map)} mapped to grid, "
            f"+{new_reveals} new reveals, "
            f"{face_down} face-down, "
            f"{self.revealed_count}/{self._rows * self._cols} total"
        )

        return self.build_grid_image()

    def build_grid_image(self) -> np.ndarray:
        """Create a clean grid image showing all revealed cards."""
        p = self._padding
        cw, ch = self._card_w, self._card_h
        img_w = self._cols * (cw + p) + p
        img_h = self._rows * (ch + p) + p

        grid_img = np.full((img_h, img_w, 3), self._bg_color, dtype=np.uint8)

        for r in range(self._rows):
            for c in range(self._cols):
                x = p + c * (cw + p)
                y = p + r * (ch + p)

                if (r, c) in self._cards:
                    # Revealed card
                    grid_img[y:y + ch, x:x + cw] = self._cards[(r, c)]
                elif (r, c) in self._known_positions:
                    # Detected but still face-down — "?" placeholder
                    cv2.rectangle(
                        grid_img, (x, y), (x + cw, y + ch),
                        self._empty_color, -1,
                    )
                    cv2.rectangle(
                        grid_img, (x, y), (x + cw, y + ch),
                        (90, 90, 100), 1,
                    )
                    tx = x + cw // 2 - 10
                    ty = y + ch // 2 + 10
                    cv2.putText(
                        grid_img, "?", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 130), 2,
                    )
                else:
                    # Never detected — faint empty slot
                    cv2.rectangle(
                        grid_img, (x, y), (x + cw, y + ch),
                        (55, 55, 65), 1,
                    )

        return grid_img

    @property
    def revealed_count(self) -> int:
        return len(self._cards)

    @property
    def total_positions(self) -> int:
        return len(self._known_positions)

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def positions(self) -> Set[Tuple[int, int]]:
        return set(self._cards.keys())

    def clear(self) -> None:
        """Reset everything including the anchor."""
        self._cards.clear()
        self._known_positions.clear()
        self._capture_count = 0
        self._anchor_rows = None
        self._anchor_cols = None

    # ------------------------------------------------------------------
    # Anchor establishment (first capture only)
    # ------------------------------------------------------------------

    def _establish_anchor(
        self, tiles: List[DetectedTile]
    ) -> Dict[Tuple[int, int], DetectedTile]:
        """First capture: cluster tiles into rows/cols, save as anchor.

        Uses GridClusterer for gap-analysis-based clustering, then pads
        the detected rows/cols to the full board_rows x board_cols grid
        by extrapolating from the median spacing.
        """
        centers = [t.center for t in tiles]
        clusterer = GridClusterer(self._config)
        grid_result = clusterer.cluster(centers)

        row_positions = list(grid_result.row_positions)
        col_positions = list(grid_result.col_positions)

        # Pad rows to board_rows
        row_positions = self._pad_positions(
            row_positions, self._rows
        )
        # Pad cols to board_cols
        col_positions = self._pad_positions(
            col_positions, self._cols
        )

        self._anchor_rows = row_positions
        self._anchor_cols = col_positions

        print(
            f"  Anchor established: {len(row_positions)} rows, "
            f"{len(col_positions)} cols"
        )

        # Map detected tiles to the anchor grid
        return self._snap_to_anchor(tiles)

    @staticmethod
    def _pad_positions(positions: List[float], target_count: int) -> List[float]:
        """Extend a list of sorted positions to target_count by extrapolating spacing.

        If we detected 4 rows but need 6, compute the average spacing and
        add rows above/below to reach 6 total.
        """
        if len(positions) >= target_count:
            return positions[:target_count]

        if len(positions) < 2:
            # Can't compute spacing from a single point — use a default
            return positions + [positions[0] + 60 * i for i in range(1, target_count)]

        # Compute median spacing between consecutive positions
        diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        spacing = float(np.median(diffs))

        if spacing <= 0:
            spacing = 60.0

        # Add positions at the end until we reach target_count
        while len(positions) < target_count:
            positions.append(positions[-1] + spacing)

        return positions

    # ------------------------------------------------------------------
    # Snap-to-anchor (all captures after the first)
    # ------------------------------------------------------------------

    def _snap_to_anchor(
        self, tiles: List[DetectedTile]
    ) -> Dict[Tuple[int, int], DetectedTile]:
        """Map each tile to the nearest anchor (row, col).

        Skips tiles that are too far from any anchor position (junk that
        survived the filter). Resolves conflicts by keeping the tile
        closest to the anchor intersection point.
        """
        anchor_rows = self._anchor_rows
        anchor_cols = self._anchor_cols

        # Compute snap tolerance from spacing
        row_diffs = [anchor_rows[i + 1] - anchor_rows[i]
                     for i in range(len(anchor_rows) - 1)]
        col_diffs = [anchor_cols[i + 1] - anchor_cols[i]
                     for i in range(len(anchor_cols) - 1)]
        row_tol = float(np.median(row_diffs)) * self._tol_factor if row_diffs else 40.0
        col_tol = float(np.median(col_diffs)) * self._tol_factor if col_diffs else 40.0

        # assigned[(row, col)] = (tile, distance_to_anchor_center)
        assigned: Dict[Tuple[int, int], Tuple[DetectedTile, float]] = {}

        for tile in tiles:
            cx, cy = tile.center

            # Find nearest row
            row_idx, row_dist = self._nearest_index(anchor_rows, cy)
            if row_dist > row_tol:
                continue  # too far from any row — skip

            # Find nearest col
            col_idx, col_dist = self._nearest_index(anchor_cols, cx)
            if col_dist > col_tol:
                continue  # too far from any col — skip

            # Clamp to valid grid range
            row_idx = max(0, min(row_idx, self._rows - 1))
            col_idx = max(0, min(col_idx, self._cols - 1))

            # Total distance to the anchor intersection point
            dist = row_dist ** 2 + col_dist ** 2

            key = (row_idx, col_idx)
            if key not in assigned or dist < assigned[key][1]:
                assigned[key] = (tile, dist)

        return {pos: tile for pos, (tile, _) in assigned.items()}

    @staticmethod
    def _nearest_index(positions: List[float], value: float) -> Tuple[int, float]:
        """Find the index of the nearest position and the distance to it."""
        best_idx = 0
        best_dist = abs(positions[0] - value)
        for i in range(1, len(positions)):
            d = abs(positions[i] - value)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist
