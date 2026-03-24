from dataclasses import dataclass
from typing import Tuple


@dataclass
class OracleConfig:
    # Camera
    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720

    # Processing resolution
    process_width: int = 1280
    process_height: int = 720

    # ArUco
    aruco_dict_name: str = "DICT_4X4_50"
    aruco_marker_ids: Tuple[int, ...] = (0, 1, 2, 3)
    aruco_marker_size_cm: float = 3.0

    # Board warping
    warp_output_width: int = 800
    warp_output_height: int = 600

    # Tile detection
    tile_min_area: int = 1500
    tile_max_area: int = 8000
    tile_aspect_ratio_range: Tuple[float, float] = (0.5, 1.5)
    tile_blur_kernel: int = 5
    tile_min_dimension: int = 15
    tile_max_dimension: int = 80
    tile_min_rectangularity: float = 0.55
    tile_duplicate_distance: float = 25.0
    tile_canny_low: int = 20
    tile_canny_high: int = 80
    tile_hsv_sat_max: int = 100
    tile_hsv_val_min: int = 160
    tile_morph_close_iterations: int = 3

    # Grid clustering
    grid_cluster_tolerance: float = 0.4

    # Fixed board grid — the actual game board layout
    board_rows: int = 6
    board_cols: int = 10

    # Anchor-based grid snapping — fraction of row/col spacing for snap tolerance
    anchor_tolerance_factor: float = 0.4

    # Card classification — HSV range for face-down red/orange sun tiles
    # Covers both pink (old theme) and red (new theme)
    facedown_hue_range: Tuple[int, int] = (0, 18)
    facedown_sat_range: Tuple[int, int] = (70, 255)
    facedown_val_range: Tuple[int, int] = (100, 255)
    facedown_ratio_threshold: float = 0.20

    # Card memory
    card_store_size: Tuple[int, int] = (80, 100)  # normalized (w, h)

    # Spiral
    spiral_direction: str = "clockwise"
    spiral_start_corner: str = "top-left"

    # Overlay rendering
    overlay_line_color: Tuple[int, int, int] = (0, 255, 255)
    overlay_line_thickness: int = 3
    overlay_number_color: Tuple[int, int, int] = (255, 255, 255)
    overlay_number_font_scale: float = 0.7
    overlay_alpha: float = 0.6

    # Display
    window_name: str = "Dragon Oracle"
    show_debug: bool = False
