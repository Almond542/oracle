import cv2
import numpy as np
from typing import List, Tuple, Optional

from dragon_oracle.config import OracleConfig
from dragon_oracle.grid import GridResult


class OverlayRenderer:
    def __init__(self, config: OracleConfig):
        self._line_color = config.overlay_line_color
        self._line_thickness = config.overlay_line_thickness
        self._number_color = config.overlay_number_color
        self._font_scale = config.overlay_number_font_scale
        self._alpha = config.overlay_alpha

    def render(
        self,
        frame: np.ndarray,
        spiral_path: List[Tuple[int, int]],
        grid_result: GridResult,
        homography: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw spiral overlay on the frame.

        If homography is provided, tile centers are inverse-warped back to
        original frame coordinates. Otherwise, draws directly using grid centers.
        """
        if not spiral_path or not grid_result.cells_by_pos:
            return frame

        # Get centers for each cell in spiral order
        warped_centers = []
        for r, c in spiral_path:
            cell = grid_result.cells_by_pos.get((r, c))
            if cell:
                warped_centers.append(cell.center)

        if not warped_centers:
            return frame

        # Transform to frame coordinates if homography provided
        if homography is not None:
            frame_pts = self._inverse_warp_points(
                np.array(warped_centers, dtype=np.float32), homography
            )
        else:
            frame_pts = np.array(warped_centers, dtype=np.float32)

        overlay = frame.copy()
        pts = frame_pts.astype(int)

        # Draw connecting lines (spiral path)
        for i in range(len(pts) - 1):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[i + 1])
            cv2.line(overlay, pt1, pt2, self._line_color, self._line_thickness)

        # Draw numbered circles
        for i, pt in enumerate(pts):
            center = tuple(pt)
            # Background circle
            cv2.circle(overlay, center, 16, self._line_color, -1)
            cv2.circle(overlay, center, 16, (0, 0, 0), 1)

            # Number text
            text = str(i + 1)
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 2
            )[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            cv2.putText(
                overlay, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
                self._number_color, 2,
            )

        # Alpha blend
        return cv2.addWeighted(overlay, self._alpha, frame, 1 - self._alpha, 0)

    def render_status(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Draw status text on the frame."""
        # Black background bar
        cv2.rectangle(frame, (0, 0), (len(text) * 14 + 20, 35), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        return frame

    def _inverse_warp_points(
        self, points: np.ndarray, homography: np.ndarray
    ) -> np.ndarray:
        """Map points from warped space back to original frame coordinates."""
        H_inv = np.linalg.inv(homography)
        n = len(points)
        # Convert to homogeneous coordinates
        ones = np.ones((n, 1), dtype=np.float32)
        pts_h = np.hstack([points, ones])  # Nx3
        # Transform
        mapped = (H_inv @ pts_h.T).T  # Nx3
        # De-homogenize
        mapped[:, 0] /= mapped[:, 2]
        mapped[:, 1] /= mapped[:, 2]
        return mapped[:, :2]
