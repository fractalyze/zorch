# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Field-kind helpers and a field-additive reduction that lowers on every
backend.

`field_sum` exists because a reduce-add over a binary field (`binary_field_*`,
i.e. GF(2^m)) has no lowering on the CUDA backend and hard-SIGSEGVs there
(zorch#400; sibling of the binary-field GPU-lowering gaps zorch#381 / zorch#399).
Elementwise field add lowers fine — only the *reduction* op is missing — so a
binary reduction folds its axis in a pairwise elementwise-add tree instead. This
is byte-identical to a reduce: characteristic-2 addition is associative and
commutative, so summation order is irrelevant. Prime / extension dtypes keep
`jnp.sum` (a single fused reduce); the tree is taken only where the native reduce
would crash.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array


def is_binary_field(dtype: Any) -> bool:
    """True for the binary-field family (`binary_field_ghash`, `binary_field_t*`).

    These are GF(2^m): characteristic 2, so field addition is a bitwise XOR of the
    packed representation and the multiplicative group has odd order (no 2^k-th
    roots of unity — `lax.ntt` runs the LCH additive NTT for them)."""
    return jnp.dtype(dtype).name.startswith("binary_field")


def _tree_sum_leading(x: Array) -> Array:
    """Reduce the LEADING axis (any length >= 1) by pairwise elementwise adds.

    Each pass adds the first half to the second half; an odd element rides
    through unpaired. O(log n) passes of a lowering-safe elementwise add."""
    n = x.shape[0]
    while n > 1:
        half = n // 2
        folded = x[:half] + x[half : 2 * half]
        if n & 1:  # odd length: carry the unpaired tail element into the next pass
            folded = jnp.concatenate([folded, x[2 * half :]], axis=0)
        x = folded
        n = x.shape[0]
    return x[0]


def field_sum(x: Array, axis: int | None = None) -> Array:
    """Sum `x` under field addition, contracting `axis` (all axes if None).

    Equivalent to `jnp.sum(x, axis=axis)` but lowers on the CUDA backend for
    binary fields, where a native reduce-add SIGSEGVs (see module docstring).
    Leading/trailing axes ride through, matching `jnp.sum`."""
    if not is_binary_field(x.dtype):
        return jnp.sum(x, axis=axis)
    if axis is None:
        return _tree_sum_leading(x.reshape(-1))
    return _tree_sum_leading(jnp.moveaxis(x, axis, 0))
