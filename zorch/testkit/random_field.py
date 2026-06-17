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
import zk_dtypes
from jax import Array


def _is_extension_dtype(dtype: Any) -> bool:
    """True for an algebraic extension field. `zk_dtypes.efinfo` is the extension
    registry the rest of zorch keys off (coding/reed_solomon.py, pcs/whir): it
    resolves extension metadata and raises for base fields and point types."""
    try:
        zk_dtypes.efinfo(dtype)
        return True
    except ValueError:
        return False


def rand_field(seed: int, shape: Sequence[int], dtype: Any) -> Array:
    # An extension element is k>1 base limbs, so casting a scalar canonical
    # integer embeds it into the base subfield (the k-1 higher coordinates stay
    # zero) -- a degenerate draw, not a generic extension element. Forbid it so
    # the footgun is caught at the call site; `rand_ext_field` draws the limbs.
    if _is_extension_dtype(dtype):
        raise ValueError(
            f"{np.dtype(dtype).name} is an extension field; rand_field would embed "
            f"a canonical integer into the base subfield (higher coordinates zero). "
            f"Use rand_ext_field(seed, shape, base, ext) for a generic element."
        )
    ints = np.random.default_rng(seed).integers(0, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=dtype)


def rand_ext_field(seed: int, shape: Sequence[int], base: Any, ext: Any) -> Array:
    """Random extension-field tensor: an `ext` element is `k` base-field limbs,
    so draw `(*shape, k)` base elements and bitcast."""
    k = jnp.dtype(ext).itemsize // jnp.dtype(base).itemsize
    return rand_field(seed, (*shape, k), base).view(ext).reshape(shape)
