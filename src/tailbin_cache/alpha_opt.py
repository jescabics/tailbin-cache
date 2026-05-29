
from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def read_count_threshold(alpha: float, depth: int) -> int:
    """Integer read-count threshold for the cumulative observed-frequency tail.

    The event read_count/depth >= alpha is equivalent to read_count >= ceil(alpha*depth).
    Any two alpha cutoffs with the same threshold are exactly equivalent for the
    read-sampling layer and therefore have identical CDF tables.
    """
    return int(max(0, min(int(depth), math.ceil(float(alpha) * int(depth) - 1e-15))))


def alpha_threshold_items(alphas: Iterable[float], depth: int) -> List[Tuple[int, float, int]]:
    return [(int(i), float(a), read_count_threshold(float(a), int(depth))) for i, a in enumerate(alphas)]
