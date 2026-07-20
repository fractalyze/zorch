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


def to_base_limbs(values: Array) -> Array:
    """View an extension array as its base limbs, the extension axis expanded in
    place: an element becomes its `degree` contiguous coefficients.

    The reinterpret is `lax.bitcast_convert_type`, so it stays on device and a
    caller's surrounding loop still traces as one jitted function — a host
    `np.asarray(...).view` round-trip forces eager execution instead. The bitcast
    appends the limb axis; the reshape folds it back into the trailing axis to
    recover the contiguous layout that hash leaves and wire formats expect.

    A base-field array is returned unchanged: it is already its own limbs.
    """
    dtype = values.dtype
    if base_field(dtype) == dtype:
        return values
    limbs = lax.bitcast_convert_type(values, base_field(dtype))
    return limbs.reshape(*values.shape[:-1], -1)


def from_base_limbs(values: Array, dtype: Any) -> Array:
    """The inverse view: contiguous base limbs -> elements of `dtype`, each
    `degree` limbs read as one element's coefficients.

    The trailing axis must be a multiple of the extension degree; anything else
    is a layout error rather than something to pad or truncate.
    """
    if base_field(dtype) == dtype:
        return values
    degree = zk_dtypes.efinfo(dtype).degree
    trailing = values.shape[-1] if values.ndim else 1
    if trailing % degree:
        raise ValueError(
            f"trailing axis {trailing} is not a multiple of the degree {degree} "
            f"of {fnp.dtype(dtype).name}"
        )
    grouped = values.reshape(*values.shape[:-1], -1, degree)
    return lax.bitcast_convert_type(grouped, dtype)
