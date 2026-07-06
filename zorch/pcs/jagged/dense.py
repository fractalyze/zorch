# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Log-area tier for the jagged commit and the verifier-point indicator."""
from __future__ import annotations

from zorch.utils.bits import log2_ceil_usize


def log_area_tier(total_area: int) -> int:
    """Log-area tier shared by the dense buffer (`log_m`) and the jagged
    indicator (`n_d`). The +1 keeps `t_L < 2^tier` strict (a power-of-two area
    would otherwise need t_L == 2^tier), so prefix sums always fit the bit
    width. One definition site: the opening seam requires `log_m == n_d`, so
    both sides must derive the tier identically."""
    return log2_ceil_usize(total_area) + 1
