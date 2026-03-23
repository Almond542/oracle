import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class GridCell:
    row: int
    col: int
    center: Tuple[float, float]
    tile_index: int


@dataclass
class GridResult:
    cells: List[GridCell]
    num_rows: int
    num_cols: int
    grid_matrix: List[List[Optional[int]]]  # [row][col] -> tile_index or None
    row_positions: List[float]  # average Y for each row
    col_positions: List[float]  # average X for each column
    cells_by_pos: Dict[Tuple[int, int], GridCell] = field(default_factory=dict)


class GridClusterer:
    def __init__(self, config=None):
        self._tolerance = config.grid_cluster_tolerance if config else 0.4

    def cluster(self, tile_centers: List[Tuple[float, float]]) -> GridResult:
        """Cluster tile centroids into a row/column grid."""
        if not tile_centers:
            return GridResult([], 0, 0, [], [], [], {})

        centers = np.array(tile_centers)
        ys = centers[:, 1]
        xs = centers[:, 0]

        # Cluster Y values into rows, X values into columns
        row_labels, row_positions = self._cluster_1d(ys)
        col_labels, col_positions = self._cluster_1d(xs)

        num_rows = len(row_positions)
        num_cols = len(col_positions)

        # Build grid matrix
        grid_matrix: List[List[Optional[int]]] = [
            [None] * num_cols for _ in range(num_rows)
        ]

        cells = []
        cells_by_pos = {}

        for i, (cx, cy) in enumerate(tile_centers):
            row = row_labels[i]
            col = col_labels[i]

            # Handle conflicts — keep tile closest to cell center
            if grid_matrix[row][col] is not None:
                existing_idx = grid_matrix[row][col]
                ex, ey = tile_centers[existing_idx]
                target_x = col_positions[col]
                target_y = row_positions[row]
                existing_dist = (ex - target_x) ** 2 + (ey - target_y) ** 2
                new_dist = (cx - target_x) ** 2 + (cy - target_y) ** 2
                if new_dist >= existing_dist:
                    continue

            grid_matrix[row][col] = i
            cell = GridCell(row=row, col=col, center=(cx, cy), tile_index=i)
            cells_by_pos[(row, col)] = cell

        # Rebuild cells list from final grid assignments
        cells = list(cells_by_pos.values())

        return GridResult(
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            grid_matrix=grid_matrix,
            row_positions=row_positions,
            col_positions=col_positions,
            cells_by_pos=cells_by_pos,
        )

    def _cluster_1d(self, values: np.ndarray) -> Tuple[List[int], List[float]]:
        """Cluster 1D values into groups by gap analysis.

        Returns:
            labels: cluster index for each input value
            positions: mean value for each cluster
        """
        n = len(values)
        if n == 0:
            return [], []

        sorted_indices = np.argsort(values)
        sorted_vals = values[sorted_indices]

        if n == 1:
            return [0], [float(sorted_vals[0])]

        # Compute gaps between consecutive sorted values
        gaps = np.diff(sorted_vals)

        if len(gaps) == 0:
            return [0] * n, [float(np.mean(sorted_vals))]

        # Find threshold: separate within-cluster gaps from between-cluster gaps
        # Use the largest gap jump in sorted gaps
        median_gap = np.median(gaps)
        if median_gap == 0:
            median_gap = 1.0

        # Threshold: gaps larger than this indicate a new cluster
        # Use a factor of the median tile spacing
        threshold = median_gap * (1 + self._tolerance) * 2

        # Assign clusters
        cluster_id = 0
        cluster_assignments = np.zeros(n, dtype=int)
        cluster_assignments[sorted_indices[0]] = 0

        for i in range(1, n):
            if gaps[i - 1] > threshold:
                cluster_id += 1
            cluster_assignments[sorted_indices[i]] = cluster_id

        # Compute cluster positions (means)
        num_clusters = cluster_id + 1
        positions = []
        for c in range(num_clusters):
            mask = cluster_assignments == c
            positions.append(float(np.mean(values[mask])))

        labels = cluster_assignments.tolist()
        return labels, positions
