"""Shared normal-form linear helpers for the Poseidon family — explicit field
add/mul, no dot/reduce/gather.

Every Poseidon-family linear layer is written as a fixed, unrolled sum of
column-scaled lanes so a round body stays straight-line element-wise and fuses to
one kernel: `fnp.dot`/`fnp.sum` lower to a reduction (the `kInput` fusion
boundary) and dynamic indexing to `gather`, either of which splits the kernel.

A layer must also keep the number of times it reads its **chained** input — the
state, or the scalar a partial round threads through every lane — from scaling
with the width. Per-lane scalar expressions each re-derive that value, so where
rounds chain, the cost compounds per round rather than adding: measured ~3.6x per
round on the CPU backend, enough that a 22-round Goldilocks permutation never
finished (https://github.com/fractalyze/zorch/issues/565). Read the input once as
an array, or hoist its lanes once and index the hoisted list.

`unrolled_sum` is that summation primitive; `apply_matrix` is the dense
`matrix @ state` built on it for a **field-array** matrix. The scheme-specific
layers (`apply_dense_mds`, `apply_external_m4`, `apply_internal`,
`apply_sparse_partial`) live in the per-permutation `linear.py` and build on
`unrolled_sum` from here.
"""

from __future__ import annotations

import functools
import operator

from frx import Array


def unrolled_sum(terms: list[Array]) -> Array:
    """Sum `terms` as a left-fold of binary field adds: `((t0 + t1) + t2) + ...`.

    Deliberately not `fnp.sum`/`sum()`: those lower to a reduction op — the
    `kInput` fusion boundary that would split the round body's kernel. The fold
    stays a chain of straight-line element-wise adds, so the whole layer fuses.
    """
    return functools.reduce(operator.add, terms)


def apply_matrix(matrix: Array, state: Array) -> Array:
    """Dense `matrix @ state`, as the sum of each column scaled by its lane —
    `out[i] = sum_j matrix[i][j] * state[j]`.

    Written as an unrolled `unrolled_sum` rather than `matrix @ state`/`fnp.dot`
    on purpose: the matmul/dot lowers to a reduction (the `kInput` fusion
    boundary) that splits the round body's kernel; the column-scaled sum stays
    element-wise and fuses. Takes the matrix as a **field array** (for entries too
    large for an int64 literal); the integer-literal siblings are each
    permutation's `apply_dense_mds` / `apply_external_m4`.
    """
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D, got shape {state.shape}")
    w = state.shape[0]
    if matrix.shape != (w, w):
        raise ValueError(
            f"need a square matrix matching 1-D state, got matrix {matrix.shape}, "
            f"state {state.shape}"
        )
    return unrolled_sum([matrix[:, j] * state[j] for j in range(w)])
