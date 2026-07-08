# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Field-dtype helpers: the base prime field of a (possibly extension) dtype, and
the naturals {0..n−1} built in it."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import zk_dtypes
from jax import Array


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
    constant, NOT `jnp.arange`: these nodes feed the fused round-poly kernels, whose
    bodies must stay straight-line element-wise (an `iota` is a forbidden op there,
    and an iota over an extension dtype is unsupported in the fork besides)."""
    return jnp.array(list(range(n)), base_field(dtype))
