import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from dragon_oracle.config import OracleConfig


class FrameSource:
    """Base class for frame sources."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class CameraSource(FrameSource):
    """Live camera capture."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._target_size = (width, height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, self._target_size)
        return ret, frame

    def release(self) -> None:
        self._cap.release()


class StaticImageSource(FrameSource):
    """Iterates over a directory of images or a single image."""

    def __init__(self, path: str, target_size: Tuple[int, int] = (1280, 720),
                 loop: bool = True):
        self._target_size = target_size
        self._loop = loop
        self._index = 0

        p = Path(path)
        if p.is_dir():
            self._files = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        elif p.is_file():
            self._files = [p]
        else:
            self._files = []

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._files:
            return False, None

        if self._index >= len(self._files):
            if self._loop:
                self._index = 0
            else:
                return False, None

        img = cv2.imread(str(self._files[self._index]))
        self._index += 1

        if img is None:
            return False, None

        img = cv2.resize(img, self._target_size)
        return True, img

    @property
    def current_filename(self) -> str:
        if self._files and 0 < self._index <= len(self._files):
            return self._files[self._index - 1].name
        return ""

    @property
    def total_images(self) -> int:
        return len(self._files)


def create_source(config: OracleConfig,
                  static_path: Optional[str] = None) -> FrameSource:
    """Factory: returns CameraSource or StaticImageSource."""
    target_size = (config.process_width, config.process_height)
    if static_path:
        return StaticImageSource(static_path, target_size=target_size)
    return CameraSource(config.camera_index,
                        config.capture_width, config.capture_height)
