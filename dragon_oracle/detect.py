"""
Cell-difference detection for the Dragon Oracle grid.

Compares each cell in a captured screenshot against the reference image
(all cards face-down) to find newly flipped face-up cards.
"""
from typing import List, Set, Tuple

import cv2
import numpy as np

from dragon_oracle.capture import get_cell_inset, split_into_cells
from dragon_oracle.config import OracleConfig


def detect_flipped_cells(
    board_img: np.ndarray,
    reference_cells: List[List[np.ndarray]],
    cfg: OracleConfig,
    known_cells: Set[Tuple[int, int]],
) -> List[dict]:
    """Find cells that differ from the reference and aren't already recorded.

    Args:
        board_img: current screenshot of the board region
        reference_cells: grid of cell images from the all-face-down reference
        cfg: configuration with thresholds
        known_cells: set of (row, col) already recorded in the board

    Returns:
        List of {"row": int, "col": int, "jpeg": bytes, "diff_score": float}
        sorted by diff_score descending.
    """
    current_cells = split_into_cells(board_img, cfg.board_rows, cfg.board_cols)
    results = []

    for r in range(cfg.board_rows):
        for c in range(cfg.board_cols):
            if (r, c) in known_cells:
                continue

            ref_inset = get_cell_inset(reference_cells[r][c], cfg.cell_inset_fraction)
            cur_inset = get_cell_inset(current_cells[r][c], cfg.cell_inset_fraction)

            # Ensure same dimensions
            if ref_inset.shape != cur_inset.shape:
                continue

            diff = cv2.absdiff(cur_inset, ref_inset)
            mean_diff = float(diff.mean())

            # Count pixels where any channel changed significantly
            max_channel_diff = diff.max(axis=2)
            changed_pixels = int((max_channel_diff > cfg.diff_threshold).sum())
            total_pixels = max_channel_diff.size
            changed_frac = changed_pixels / total_pixels if total_pixels > 0 else 0

            if mean_diff > cfg.diff_threshold and changed_frac > cfg.changed_pixel_fraction:
                cell_full = current_cells[r][c]
                thumb = cv2.resize(cell_full,
                                   (cfg.card_store_w, cfg.card_store_h),
                                   interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", thumb,
                                       [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    continue
                results.append({
                    "row": r,
                    "col": c,
                    "jpeg": buf.tobytes(),
                    "diff_score": mean_diff,
                })

    results.sort(key=lambda x: x["diff_score"], reverse=True)
    return results
