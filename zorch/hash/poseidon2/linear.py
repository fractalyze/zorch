"""Normal-form linear layers — explicit field add/mul, no dot/reduce/gather.

`matrix @ state` and the internal layer `(off_diag*J + Diag(d)) @ state` written as a
fixed, unrolled sum of column-scaled lanes. This keeps a round body straight-line
element-wise so it fuses to one kernel: `jnp.dot`/`jnp.sum` lower to a reduction
(the `kInput` fusion boundary) and dynamic indexing to `gather`, either of which
splits the kernel. Static lane indices lower to `slice`, not `gather`.
"""

from __future__ import annotations

import functools
import operator

import jax.numpy as jnp
from jax import Array

# Standard Poseidon2 external M4 (mirrors params.default_external_matrix). Kept as
# Python ints, not a field array: apply_external_standard scales lanes by these
# literals so it closes over no matrix. A closed-over array inside a
# jax.lax.composite is lifted to a leading operand, which would break a
# name-routed emitter's fixed operand ABI.
_M4 = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))


def _unrolled_sum(terms: list[Array]) -> Array:
    return functools.reduce(operator.add, terms)


def apply_matrix(matrix: Array, state: Array) -> Array:
    """`matrix @ state`, as the sum of each column scaled by its lane."""
    if state.ndim != 1 or matrix.shape != (state.shape[0], state.shape[0]):
        raise ValueError(
            f"need a square matrix matching 1-D state, got matrix {matrix.shape}, "
            f"state {state.shape}"
        )
    w = state.shape[0]
    return _unrolled_sum([matrix[:, j] * state[j] for j in range(w)])


def apply_external_standard(state: Array) -> Array:
    """Standard M4-circulant external layer `M[i][j] = M4[i%4][j%4]*(2 if same
    4-block)`, byte-equal to `apply_matrix(default_external_matrix(w, dtype),
    state)` but scaling lanes by integer literals so no matrix array is captured
    — required inside a name-routed `fused_region` (see `_M4`). `width` must be a
    positive multiple of 4.
    """
    w = state.shape[0]
    if state.ndim != 1 or w == 0 or w % 4 != 0:
        raise ValueError(
            f"standard external layer needs a 1-D state with width a positive "
            f"multiple of 4, got {state.shape}"
        )
    return jnp.stack(
        [
            _unrolled_sum(
                [
                    _M4[i % 4][j % 4] * (2 if i // 4 == j // 4 else 1) * state[j]
                    for j in range(w)
                ]
            )
            for i in range(w)
        ]
    )


def apply_internal(
    internal_diag: Array, state: Array, off_diag: Array | None = None
) -> Array:
    """The internal diffusion layer `(off_diag*J + Diag(internal_diag)) @ state`.

    Equals `off_diag*sum(state) + internal_diag*state`. `J @ state` is the
    all-lanes sum broadcast to every lane; unrolling it keeps the layer
    reduction-free. `off_diag` scales the all-ones (J) part; when None it is the
    identity, i.e. the plain `J + Diag(internal_diag)` form.
    """
    if state.ndim != 1 or internal_diag.shape != state.shape:
        raise ValueError(
            f"internal_diag and state must be matching 1-D vectors, got "
            f"internal_diag {internal_diag.shape}, state {state.shape}"
        )
    w = state.shape[0]
    total = _unrolled_sum([state[j] for j in range(w)])
    if off_diag is not None:
        total = off_diag * total
    return total + internal_diag * state
