# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Field-dtype helpers: the base prime field of a (possibly extension) dtype, the
naturals {0..n−1} built in it, the binary-field predicate, and the views between
an extension array and its base limbs."""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import zk_dtypes
from frx import Array, lax


def base_field(dtype: Any) -> Any:
    """The base prime field of `dtype`: `dtype` itself if already prime, else the
    subfield an extension is built over (`zk_dtypes.efinfo` raises for a
    non-extension dtype)."""
    try:
        return zk_dtypes.efinfo(dtype).base_field_dtype
    except ValueError:
        return dtype


def naturals(n: int, dtype: Any) -> Array:
    """`[0, 1, …, n−1]` in the base field of `dtype`, as a compile-time constant.

    Integer nodes are prime-field elements, so they are built there; an extension
    caller promotes at multiply time, which is cheaper than extension-typed nodes
    and byte-identical to embedding each node into the extension. Built as a
    constant, NOT `fnp.arange`: these nodes feed the fused round-poly kernels, whose
    bodies must stay straight-line element-wise (an `iota` is a forbidden op there,
    and an iota over an extension dtype is unsupported in the fork besides)."""
    return fnp.array(list(range(n)), base_field(dtype))


def is_binary_field(dtype: Any) -> bool:
    """True for the binary-field family (`binary_field_ghash`, `binary_field_t*`):
    GF(2^m), characteristic 2 — field addition is a bitwise XOR of the packed
    representation, and `lax.ntt` runs the LCH additive NTT for them."""
    return fnp.dtype(dtype).name.startswith("binary_field")


def to_limb_rows(values: Array) -> Array:
    """View an extension array as its base coefficients, one row per element:
    `(..., N)` extension -> `(..., N, degree)` base.

    The reinterpret is `lax.bitcast_convert_type`, so it stays on device and a
    caller's surrounding loop still traces as one jitted function — a host
    `np.asarray(...).view` round-trip forces eager execution instead. No bytes
    move; only the dtype and shape metadata change.

    The limb axis is left in place. A caller that wants the limbs contiguous in
    the trailing axis (a transcript batch, a hash leaf row) reshapes, which is
    free — the layout it wants is the caller's to state, not this function's to
    guess. `from_limb_rows` is the exact inverse.

    A base-field array is returned unchanged: it is already its own coefficients,
    one per element, and adding a length-1 axis would imply an extension that is
    not there.
    """
    dtype = values.dtype
    if base_field(dtype) == dtype:
        return values
    return lax.bitcast_convert_type(values, base_field(dtype))


def from_limb_rows(values: Array, dtype: Any) -> Array:
    """The exact inverse: `(..., N, degree)` base -> `(..., N)` elements of
    `dtype`, each row read as one element's coefficients.

    `dtype` must be given because limbs carry no record of what they were —
    a trailing axis of 12 could be 4 cubic elements or 3 quartic ones.

    The trailing axis must be exactly the extension degree; anything else is a
    layout error at the caller rather than something to pad, truncate, or
    silently regroup.
    """
    if base_field(dtype) == dtype:
        return values
    degree = zk_dtypes.efinfo(dtype).degree
    trailing = values.shape[-1] if values.ndim else 0
    if trailing != degree:
        raise ValueError(
            f"trailing axis must be the degree {degree} of "
            f"{fnp.dtype(dtype).name}, got {trailing}"
        )
    return lax.bitcast_convert_type(values, dtype)
