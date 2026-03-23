import cv2
import numpy as np
from typing import List, Optional

from dragon_oracle.tiles import DetectedTile
from dragon_oracle.grid import GridResult
from dragon_oracle.aruco import ArucoResult


def draw_contours(image: np.ndarray, tiles: List[DetectedTile]) -> np.ndarray:
    """Draw detected tile contours with centroids marked."""
    vis = image.copy()
    for tile in tiles:
        cv2.drawContours(vis, [tile.contour], -1, (0, 255, 0), 2)
        cx, cy = int(tile.center[0]), int(tile.center[1])
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
    return vis


def draw_grid_lines(image: np.ndarray, grid_result: GridResult) -> np.ndarray:
    """Draw row/column cluster boundaries."""
    vis = image.copy()
    h, w = vis.shape[:2]

    for y in grid_result.row_positions:
        cv2.line(vis, (0, int(y)), (w, int(y)), (255, 0, 0), 1)

    for x in grid_result.col_positions:
        cv2.line(vis, (int(x), 0), (int(x), h), (255, 0, 0), 1)

    # Label each cell
    for cell in grid_result.cells:
        cx, cy = int(cell.center[0]), int(cell.center[1])
        label = f"{cell.row},{cell.col}"
        cv2.putText(vis, label, (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return vis


def draw_aruco_markers(image: np.ndarray,
                       aruco_result: ArucoResult) -> np.ndarray:
    """Draw detected ArUco markers with IDs."""
    vis = image.copy()
    for mid, corners in aruco_result.marker_corners.items():
        pts = corners.astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(vis, f"ID:{mid}", tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis
