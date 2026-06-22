"""Normal-form dense MDS layer — explicit field add/mul, no dot/reduce/gather.

The classic-Poseidon linear layer is a dense `mds @ state` applied every round,
written as a fixed, unrolled sum of column-scaled lanes so a round body stays
straight-line element-wise and fuses to one kernel: `jnp.dot`/`jnp.sum` lower to
a reduction (the `kInput` fusion boundary) and dynamic indexing to `gather`,
either of which splits the kernel. The MDS rides as integer literals (canonical
ints), so no field array is captured — required inside a name-routed
`fused_region`, where a closed-over array lifts to a leading operand and breaks
the emitter's operand ABI. This mirrors poseidon2's `apply_external_m4`.
"""

from __future__ import annotations

import functools
import operator

import jax.numpy as jnp
from jax import Array


def _unrolled_sum(terms: list[Array]) -> Array:
    return functools.reduce(operator.add, terms)


def apply_dense_mds(mds_rows: tuple[tuple[int, ...], ...], state: Array) -> Array:
    """Dense MDS layer `mds @ state`: row `i` is `sum_j mds[i][j] * state[j]`.

    `mds_rows` is the `width x width` matrix as canonical Python ints (rows of
    ints), so lanes scale by integer literals and no field array is captured —
    required inside a name-routed `fused_region`. The unrolled per-lane sum keeps
    the layer reduction-free (no `jnp.dot`/`jnp.sum`/gather), so the round body
    lowers to a single fused kernel.
    """
    w = state.shape[0]
    if state.ndim != 1 or w == 0 or len(mds_rows) != w:
        raise ValueError(
            f"dense MDS needs a 1-D state matching a square matrix, got state "
            f"{state.shape}, matrix rows {len(mds_rows)}"
        )
    return jnp.stack(
        [_unrolled_sum([mds_rows[i][j] * state[j] for j in range(w)]) for i in range(w)]
    )
