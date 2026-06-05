# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Deterministic random field elements for tests.

Draws canonical integers in [0, 2**30) (< every supported prime) and casts to
the field dtype, which Montgomery-encodes the canonical integer.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array


def rand_field(seed: int, shape: Sequence[int], dtype: Any) -> Array:
    ints = np.random.default_rng(seed).integers(0, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=dtype)


def rand_ext_field(seed: int, shape: Sequence[int], base: Any, ext: Any) -> Array:
    """Random extension-field tensor: an `ext` element is `k` base-field limbs,
    so draw `(*shape, k)` base elements and bitcast."""
    k = jnp.dtype(ext).itemsize // jnp.dtype(base).itemsize
    return rand_field(seed, (*shape, k), base).view(ext).reshape(shape)
