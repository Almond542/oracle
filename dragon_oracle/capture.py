"""
Screenshot capture and cell splitting for the Dragon Oracle grid.
"""
from typing import List

import cv2
import mss
import numpy as np

from dragon_oracle.config import OracleConfig


def take_full_screenshot() -> np.ndarray:
    """Capture the entire primary monitor as a BGR numpy array."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        img = np.array(sct.grab(monitor))
    # mss returns BGRA on Windows; drop alpha channel
    return img[:, :, :3].copy()


def crop_region(screenshot: np.ndarray, cfg: OracleConfig) -> np.ndarray:
    """Crop a full screenshot to the calibrated board region."""
    x = cfg.region_left
    y = cfg.region_top
    w = cfg.region_width
    h = cfg.region_height
    return screenshot[y:y + h, x:x + w].copy()


def split_into_cells(board_img: np.ndarray,
                     rows: int, cols: int) -> List[List[np.ndarray]]:
    """Divide a board image into a rows x cols grid of cell images."""
    h, w = board_img.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y1 = r * cell_h
            x1 = c * cell_w
            row_cells.append(board_img[y1:y1 + cell_h, x1:x1 + cell_w].copy())
        cells.append(row_cells)
    return cells


def get_cell_inset(cell_img: np.ndarray, inset_frac: float) -> np.ndarray:
    """Return the inner portion of a cell, trimming inset_frac from each edge."""
    h, w = cell_img.shape[:2]
    dy = int(h * inset_frac)
    dx = int(w * inset_frac)
    return cell_img[dy:h - dy, dx:w - dx]
