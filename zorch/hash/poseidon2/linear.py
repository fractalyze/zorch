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


def apply_external_m4(state: Array, m4: tuple[tuple[int, ...], ...]) -> Array:
    """External layer `(I + J_blocks) ⊗ M4`: per-4-block M4, plus M4-image column
    sums across blocks — `M[i][j] = M4[i%4][j%4] * (2 if same 4-block)`.

    `m4` is the 4×4 base matrix as canonical Python ints, so lanes scale by integer
    literals and no field array is captured — required inside a name-routed
    `fused_region` (a closed-over array lifts to a leading operand and breaks the
    emitter ABI). The base M4 is not fixed: Plonky3's `circ(2,3,1,1)` and the
    HorizenLabs reference matrix are two valid choices, the caller's to pick. The
    2×-diagonal-block form matches references for width >= 8.
    """
    w = state.shape[0]
    if state.ndim != 1 or w == 0 or w % 4 != 0:
        raise ValueError(
            f"external layer needs a 1-D state with width a positive multiple of "
            f"4, got {state.shape}"
        )
    return jnp.stack(
        [
            _unrolled_sum(
                [
                    m4[i % 4][j % 4] * (2 if i // 4 == j // 4 else 1) * state[j]
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
