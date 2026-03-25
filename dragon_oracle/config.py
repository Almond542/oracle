from dataclasses import dataclass, field


@dataclass
class OracleConfig:
    # Grid dimensions (set during calibration, vary per level)
    board_rows: int = 6
    board_cols: int = 10

    # Card thumbnail storage size (pixels)
    card_store_w: int = 120
    card_store_h: int = 160

    # Screenshot capture region (set during calibration)
    region_left: int = 0
    region_top: int = 0
    region_width: int = 0
    region_height: int = 0

    # Detection tuning
    cell_inset_fraction: float = 0.15    # trim 15% from each cell edge (avoids grid lines)
    diff_threshold: float = 30.0         # per-pixel diff to count as "changed"
    changed_pixel_fraction: float = 0.25  # min fraction of changed pixels to trigger detection
