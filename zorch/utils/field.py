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


def split_limbs(values: Array) -> Array:
    """Split each extension element into its base limbs:
    `(..., N)` extension -> `(..., N, degree)` base.

    A view, not a copy — `lax.bitcast_convert_type` changes dtype and shape
    metadata on device, so a surrounding loop still traces as one jitted
    function. Reshape the result if the consumer wants the limbs contiguous in
    the trailing axis.

    A base-field array is returned unchanged, with no length-1 axis added.
    """
    dtype = values.dtype
    if base_field(dtype) == dtype:
        return values
    return lax.bitcast_convert_type(values, base_field(dtype))


def join_limbs(values: Array, dtype: Any) -> Array:
    """Join base limbs back into extension elements, the inverse of
    `split_limbs`: `(..., N, degree)` base -> `(..., N)` of `dtype`.

    `dtype` is required because limbs carry no record of what they were: a
    trailing axis of 12 could be 4 cubic elements or 3 quartic ones. The
    trailing axis must equal the degree.

    A base-field `dtype` returns the input unchanged.
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
