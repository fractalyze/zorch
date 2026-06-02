"""Normal-form linear layers — explicit field add/mul, no dot/reduce/gather.

`matrix @ state` and the internal layer `(J + Diag(d)) @ state` written as a
fixed, unrolled sum of column-scaled lanes. This keeps a round body straight-line
element-wise so it fuses to one kernel: `jnp.dot`/`jnp.sum` lower to a reduction
(the `kInput` fusion boundary) and dynamic indexing to `gather`, either of which
splits the kernel. Static lane indices lower to `slice`, not `gather`.
"""

from __future__ import annotations

import functools
import operator

from jax import Array


def _unrolled_sum(terms: list[Array]) -> Array:
    return functools.reduce(operator.add, terms)


def apply_matrix(matrix: Array, state: Array) -> Array:
    """`matrix @ state`, as the sum of each column scaled by its lane."""
    w = state.shape[0]
    return _unrolled_sum([matrix[:, j] * state[j] for j in range(w)])


def apply_internal(internal_diag: Array, state: Array) -> Array:
    """`(J + Diag(internal_diag)) @ state` = `sum(state) + internal_diag * state`.

    `J @ state` is the all-lanes sum broadcast to every lane; unrolling it keeps
    the layer reduction-free.
    """
    w = state.shape[0]
    total = _unrolled_sum([state[j] for j in range(w)])
    return total + internal_diag * state
