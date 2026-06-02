# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Bit/log helpers for power-of-two sized data."""

from __future__ import annotations


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def log2_strict_usize(n: int) -> int:
    """Exact log2 of a power of two; raises ValueError otherwise."""
    if not is_power_of_two(n):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


def log2_ceil_usize(n: int) -> int:
    """ceil(log2 n); 0 for n <= 1."""
    if n <= 1:
        return 0
    return (n - 1).bit_length()
