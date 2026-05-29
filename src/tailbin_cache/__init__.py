"""Production cache builder for certified tail-bin CDF/z lookup tables.

The public API exposes grid/config helpers, error-budget dataclasses, the HDF5
reader, and the production HDF5 builder.  Internal modules contain the vendored
finite-depth PGF backend and coefficient-table construction machinery.
"""

from .grid import CacheGrid, ParameterPoint, default_alphas, grid_from_dict, write_default_config
from .builder import BuildConfig, ErrorBudget
from .hdf5_adaptive import HDF5AdaptiveCache
from .hdf5_alpha_monotone import build_alpha_monotone_hdf5_from_config
from .planner import plan_adaptive_cache
from .shards import balanced_shard_plan, write_balanced_shard_plan
from .estimate import estimate_storage, estimate_runtime, estimate_theta_bundled_runtime

__all__ = [
    "CacheGrid", "ParameterPoint", "default_alphas", "grid_from_dict", "write_default_config",
    "BuildConfig", "ErrorBudget",
    "HDF5AdaptiveCache", "build_alpha_monotone_hdf5_from_config",
    "plan_adaptive_cache", "balanced_shard_plan", "write_balanced_shard_plan",
    "estimate_storage", "estimate_runtime", "estimate_theta_bundled_runtime",
]
