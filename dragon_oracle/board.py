"""
Card storage for the Dragon Oracle memory grid.

Stores JPEG thumbnails keyed by (row, col). No CV logic here.
"""
import base64
from typing import Dict, Tuple

import cv2
import numpy as np

from dragon_oracle.config import OracleConfig


class Board:
    def __init__(self, config: OracleConfig):
        self._rows = config.board_rows
        self._cols = config.board_cols
        self._card_w = config.card_store_w
        self._card_h = config.card_store_h
        self._padding = 6
        self._bg_color = (40, 40, 50)
        self._cards: Dict[Tuple[int, int], bytes] = {}

    def resize(self, rows: int, cols: int) -> None:
        """Change grid dimensions and clear all cards."""
        self._rows = rows
        self._cols = cols
        self._cards.clear()

    def auto_place(self, jpeg_bytes: bytes, row: int, col: int) -> str:
        """Place a card directly at (row, col).

        Returns "ok" or "occupied".
        """
        if (row, col) in self._cards:
            return "occupied"
        self._cards[(row, col)] = jpeg_bytes
        return "ok"

    def move_card(self, from_row: int, from_col: int,
                  to_row: int, to_col: int) -> str:
        """Move a placed card to a different cell.

        Returns "ok", "occupied", or "empty".
        """
        src = (from_row, from_col)
        dst = (to_row, to_col)
        if src not in self._cards:
            return "empty"
        if dst in self._cards:
            return "occupied"
        self._cards[dst] = self._cards.pop(src)
        return "ok"

    def build_grid_image(self) -> np.ndarray:
        """Render the grid as a BGR image."""
        p = self._padding
        cw, ch = self._card_w, self._card_h
        img_w = self._cols * (cw + p) + p
        img_h = self._rows * (ch + p) + p
        grid = np.full((img_h, img_w, 3), self._bg_color, dtype=np.uint8)

        for r in range(self._rows):
            for c in range(self._cols):
                x = p + c * (cw + p)
                y = p + r * (ch + p)
                if (r, c) in self._cards:
                    arr = np.frombuffer(self._cards[(r, c)], np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, (cw, ch),
                                         interpolation=cv2.INTER_AREA)
                        grid[y:y + ch, x:x + cw] = img
                    cv2.putText(grid, f"{r+1},{c+1}", (x + 2, y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                                (220, 220, 220), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(grid, (x, y), (x + cw - 1, y + ch - 1),
                                  (60, 60, 72), 1)
                    cv2.putText(grid, f"{r+1},{c+1}", (x + 2, y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                                (75, 75, 88), 1, cv2.LINE_AA)
        return grid

    def to_json(self) -> dict:
        grid: Dict[str, str] = {}
        for (r, c), jpeg_bytes in self._cards.items():
            grid[f"{r},{c}"] = base64.b64encode(jpeg_bytes).decode()
        return {
            "rows": self._rows,
            "cols": self._cols,
            "grid": grid,
            "revealed": len(self._cards),
            "total": self._rows * self._cols,
        }

    def clear(self) -> None:
        self._cards.clear()

    @property
    def revealed_count(self) -> int:
        return len(self._cards)

    @property
    def total_positions(self) -> int:
        return self._rows * self._cols

    def occupied_cells_set(self) -> set:
        return set(self._cards.keys())
