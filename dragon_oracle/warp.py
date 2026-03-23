import cv2
import numpy as np

from dragon_oracle.config import OracleConfig


class BoardWarper:
    def __init__(self, config: OracleConfig):
        self._output_size = (config.warp_output_width, config.warp_output_height)

    def warp(self, frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """Apply perspective warp to extract the board region."""
        return cv2.warpPerspective(frame, homography, self._output_size)

    def inverse_warp_points(
        self, points: np.ndarray, homography: np.ndarray
    ) -> np.ndarray:
        """Map points from warped coordinates back to original frame coordinates.

        Args:
            points: Nx2 array of (x, y) in warped space.
            homography: the forward homography (original -> warped).

        Returns:
            Nx2 array of (x, y) in original frame space.
        """
        H_inv = np.linalg.inv(homography)
        n = len(points)
        ones = np.ones((n, 1), dtype=np.float32)
        pts_h = np.hstack([points.astype(np.float32), ones])
        mapped = (H_inv @ pts_h.T).T
        mapped[:, 0] /= mapped[:, 2]
        mapped[:, 1] /= mapped[:, 2]
        return mapped[:, :2]
