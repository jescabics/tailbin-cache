from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

from .grid import CacheGrid


BYTES_PER_VALUE = {
    "float64": 8,
    "float32": 4,
    "int16": 2,
}


def estimate_storage(grid: CacheGrid, Kmax: int, storage: str = "int16", compression_ratio: float = 1.0) -> Dict[str, Any]:
    storage = str(storage)
    if storage not in BYTES_PER_VALUE:
        raise ValueError(f"unknown storage {storage}")
    n_values = int(grid.n_tables) * int(Kmax + 1)
    raw_bytes = n_values * BYTES_PER_VALUE[storage]
    compressed_bytes = int(raw_bytes / max(float(compression_ratio), 1e-12))
    return {
        "n_parameter_points": int(grid.n_parameter_points),
        "n_alphas": int(len(grid.alphas)),
        "n_tables": int(grid.n_tables),
        "Kmax": int(Kmax),
        "storage": storage,
        "n_values": int(n_values),
        "raw_bytes": int(raw_bytes),
        "raw_GB": float(raw_bytes / 1e9),
        "compression_ratio_assumed": float(compression_ratio),
        "compressed_bytes_estimate": int(compressed_bytes),
        "compressed_GB_estimate": float(compressed_bytes / 1e9),
    }


def estimate_runtime(n_tables: int, seconds_per_table: float, n_jobs: int = 20) -> Dict[str, Any]:
    total = float(n_tables) * float(seconds_per_table)
    wall = total / max(1, int(n_jobs))
    return {
        "n_tables": int(n_tables),
        "seconds_per_table_assumed": float(seconds_per_table),
        "n_jobs": int(n_jobs),
        "total_cpu_seconds": total,
        "estimated_wall_seconds": wall,
        "estimated_wall_hours": wall / 3600.0,
        "estimated_wall_days": wall / 86400.0,
    }



def estimate_theta_bundled_runtime(grid: CacheGrid, seconds_per_bundle: float, n_jobs: int = 20) -> Dict[str, Any]:
    """Estimate runtime for build-theta-bundles mode.

    One bundle corresponds to a fixed (R,T,N,depth,alpha) and all theta_f values.
    """
    n_base_points = len(grid.R_values) * len(grid.T_values) * len(grid.N_values) * len(grid.depth_values)
    n_bundles = int(n_base_points * len(grid.alphas))
    total = float(n_bundles) * float(seconds_per_bundle)
    wall = total / max(1, int(n_jobs))
    return {
        "n_theta_values": int(len(grid.theta_values)),
        "n_base_points_excluding_theta": int(n_base_points),
        "n_bundles": int(n_bundles),
        "seconds_per_bundle_assumed": float(seconds_per_bundle),
        "n_jobs": int(n_jobs),
        "total_cpu_seconds": total,
        "estimated_wall_seconds": wall,
        "estimated_wall_hours": wall / 3600.0,
        "estimated_wall_days": wall / 86400.0,
        "amortized_seconds_per_theta_table": float(seconds_per_bundle) / max(1, int(len(grid.theta_values))),
    }
