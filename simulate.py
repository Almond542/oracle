"""
Simulation: process test_images/1.jpg → 13.jpg in order,
accumulate cards on a single Board, save grid snapshots to sim_output/.
"""
import os
import sys

import cv2

from dragon_oracle.board import Board
from dragon_oracle.config import OracleConfig
from dragon_oracle.detect import detect_cards_with_positions


def main():
    cfg = OracleConfig()
    board = Board(cfg)

    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    out_dir  = os.path.join(os.path.dirname(__file__), "sim_output")
    os.makedirs(out_dir, exist_ok=True)

    # Sort by numeric filename: 1.jpg, 2.jpg, ..., 13.jpg
    files = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if not files:
        print(f"No .jpg files found in {test_dir}")
        sys.exit(1)

    print(f"{'Image':<10} {'Detected':>8} {'Placed':>8} {'Skipped':>8} {'Total':>8}")
    print("-" * 46)

    for img_file in files:
        path = os.path.join(test_dir, img_file)
        frame = cv2.imread(path)
        if frame is None:
            print(f"{img_file:<10} ERROR: could not read file")
            continue

        cards = detect_cards_with_positions(frame, cfg)
        placed_count  = 0
        skipped_count = 0

        for card in cards:
            result = board.auto_place(card["jpeg"], card["row"], card["col"])
            if result == "ok":
                placed_count += 1
            else:
                skipped_count += 1  # cell already occupied

        # Save grid snapshot after this image
        grid_img = board.build_grid_image()
        out_path = os.path.join(out_dir, f"grid_after_{os.path.splitext(img_file)[0]}.jpg")
        cv2.imwrite(out_path, grid_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        print(f"{img_file:<10} {len(cards):>8} {placed_count:>8} {skipped_count:>8} {board.revealed_count:>8}")

    print("-" * 46)
    print(f"Final: {board.revealed_count} / {board.total_positions} cells revealed")
    print(f"Snapshots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
