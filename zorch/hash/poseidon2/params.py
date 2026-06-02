"""Poseidon2Params — the fully-free parameter surface (dataclass added in a later task)."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array


def _mds_external_default(width: int, dtype: Any) -> Array:
    """Canonical-for-width Poseidon2 external matrix: M[i][j] = M4[i%4][j%4] * (2 if same 4-block).

    Built list -> jnp.array so HLO sees a kConstant. Determined wholly by (width, dtype);
    carries no field/scheme identity. `width` must be a positive multiple of 4.
    """
    if width % 4 != 0:
        raise ValueError(f"external matrix default needs width % 4 == 0, got {width}")
    m4 = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
    mds = [[m4[i % 4][j % 4] * (2 if i // 4 == j // 4 else 1) for j in range(width)]
           for i in range(width)]
    return jnp.array(mds, dtype=dtype)
