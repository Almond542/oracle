import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from dragon_oracle.config import OracleConfig


@dataclass
class ArucoResult:
    marker_corners: Dict[int, np.ndarray]  # marker_id -> 4x2 corner array
    marker_ids: List[int]
    homography: Optional[np.ndarray]  # 3x3 matrix, None if <4 markers found
    board_corners: Optional[np.ndarray]  # 4x2 ordered corners in frame coords


class ArucoDetector:
    def __init__(self, config: OracleConfig):
        dict_id = getattr(cv2.aruco, config.aruco_dict_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._target_ids: Set[int] = set(config.aruco_marker_ids)
        self._warp_size = (config.warp_output_width, config.warp_output_height)

    def detect(self, frame: np.ndarray) -> ArucoResult:
        """Detect ArUco markers and compute homography if 4 corners found."""
        corners, ids, _ = self._detector.detectMarkers(frame)

        marker_corners: Dict[int, np.ndarray] = {}
        detected_ids: List[int] = []

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                mid = int(marker_id)
                if mid in self._target_ids:
                    marker_corners[mid] = corners[i][0]  # 4x2 array
                    detected_ids.append(mid)

        homography = None
        board_corners = None

        if len(marker_corners) >= 4:
            # Get center of each marker
            centers = {
                mid: np.mean(crns, axis=0)
                for mid, crns in marker_corners.items()
            }
            board_corners = self._order_corners(centers)
            homography = self._compute_homography(board_corners)

        return ArucoResult(
            marker_corners=marker_corners,
            marker_ids=detected_ids,
            homography=homography,
            board_corners=board_corners,
        )

    def _order_corners(self, centers: Dict[int, np.ndarray]) -> np.ndarray:
        """Order marker centers as [top-left, top-right, bottom-right, bottom-left]."""
        pts = np.array(list(centers.values()), dtype=np.float32)

        # Sort by sum (x+y) for TL (smallest) and BR (largest)
        # Sort by diff (y-x) for TR (smallest) and BL (largest)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]  # top-left
        ordered[1] = pts[np.argmin(d)]  # top-right
        ordered[2] = pts[np.argmax(s)]  # bottom-right
        ordered[3] = pts[np.argmax(d)]  # bottom-left

        return ordered

    def _compute_homography(self, src_corners: np.ndarray) -> np.ndarray:
        """Compute homography from source corners to a rectangle."""
        dst = np.array(
            [
                [0, 0],
                [self._warp_size[0], 0],
                [self._warp_size[0], self._warp_size[1]],
                [0, self._warp_size[1]],
            ],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src_corners, dst)
        return H
