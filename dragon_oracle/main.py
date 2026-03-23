import argparse
import cv2

from dragon_oracle.config import OracleConfig
from dragon_oracle.capture import create_source, StaticImageSource
from dragon_oracle.aruco import ArucoDetector
from dragon_oracle.warp import BoardWarper
from dragon_oracle.tiles import TileDetector
from dragon_oracle.grid import GridClusterer
from dragon_oracle.spiral import spiral_order
from dragon_oracle.overlay import OverlayRenderer
from dragon_oracle import debug as dbg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dragon Oracle")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to image or directory for static mode",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index for live mode",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Show debug visualization windows")
    parser.add_argument("--no-aruco", action="store_true",
                        help="Skip ArUco detection, detect tiles on full frame")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OracleConfig(camera_index=args.camera, show_debug=args.debug)

    source = create_source(config, static_path=args.image)
    aruco_detector = ArucoDetector(config)
    warper = BoardWarper(config)
    tile_detector = TileDetector(config)
    grid_clusterer = GridClusterer(config)
    renderer = OverlayRenderer(config)

    paused = False

    while True:
        if not paused:
            ret, frame = source.read()
            if not ret:
                break
            display_frame = frame.copy()
        else:
            # Re-use last frame when paused
            pass

        status_parts = []

        # Show filename for static images
        if isinstance(source, StaticImageSource):
            status_parts.append(source.current_filename)

        if not args.no_aruco:
            aruco_result = aruco_detector.detect(frame)
            n_markers = len(aruco_result.marker_ids)
            status_parts.append(f"Markers: {n_markers}/4")

            if config.show_debug and aruco_result.marker_ids:
                cv2.imshow("ArUco", dbg.draw_aruco_markers(frame, aruco_result))

            if aruco_result.homography is not None:
                warped = warper.warp(frame, aruco_result.homography)
                tiles = tile_detector.detect(warped)
                homography = aruco_result.homography

                if config.show_debug:
                    cv2.imshow("Warped", warped)
                    if tiles:
                        cv2.imshow("Contours", dbg.draw_contours(warped, tiles))
            else:
                # Not enough markers — fall back to direct detection
                tiles = tile_detector.detect(frame)
                homography = None
        else:
            tiles = tile_detector.detect(frame)
            homography = None

        status_parts.append(f"Tiles: {len(tiles)}")

        if tiles:
            centers = [t.center for t in tiles]
            grid_result = grid_clusterer.cluster(centers)
            status_parts.append(
                f"Grid: {grid_result.num_rows}x{grid_result.num_cols}"
            )

            path = spiral_order(
                grid_result.grid_matrix,
                config.spiral_direction,
                config.spiral_start_corner,
            )
            status_parts.append(f"Spiral: {len(path)} steps")

            display_frame = renderer.render(
                display_frame, path, grid_result, homography
            )

            if config.show_debug:
                debug_img = frame if homography is None else warped
                cv2.imshow("Grid", dbg.draw_grid_lines(debug_img, grid_result))

        status = " | ".join(status_parts)
        display_frame = renderer.render_status(display_frame, status)
        cv2.imshow(config.window_name, display_frame)

        # Handle keyboard input
        wait_ms = 0 if (paused or isinstance(source, StaticImageSource)) else 1
        if isinstance(source, StaticImageSource) and not paused:
            wait_ms = 500  # auto-advance every 500ms for image slideshow

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or key == 27:  # q or Esc
            break
        elif key == ord("d"):
            config.show_debug = not config.show_debug
            if not config.show_debug:
                cv2.destroyWindow("Warped")
                cv2.destroyWindow("Contours")
                cv2.destroyWindow("Grid")
                cv2.destroyWindow("ArUco")
        elif key == ord("s"):
            config.spiral_direction = (
                "counterclockwise"
                if config.spiral_direction == "clockwise"
                else "clockwise"
            )
        elif key == ord(" "):  # space to pause/resume
            paused = not paused
        elif key == ord("n"):  # next image (static mode)
            paused = False

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
