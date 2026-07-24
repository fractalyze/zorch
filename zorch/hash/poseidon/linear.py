"""Normal-form Poseidon linear layers — explicit field add/mul, no dot/reduce/gather.

Every linear layer is a fixed, unrolled sum of column-scaled lanes so a round
body stays straight-line element-wise and fuses to one kernel: `fnp.dot`/`fnp.sum`
lower to a reduction (the `kInput` fusion boundary) and dynamic indexing to
`gather`, either of which splits the kernel.

Two matrix forms, by where the matrix rides:
- `apply_dense_mds` takes the MDS as **integer literals** (canonical ints) — no
  field array is captured, required inside a name-routed `fused_region` where a
  closed-over array lifts to a leading operand and breaks the emitter's ABI.
- `apply_matrix` / `apply_sparse_partial` take **field-array** matrices — for the
  optimized-sparse variant, whose dense full-field entries exceed an int64 literal
  and so cannot ride as ints; they stay field arrays, which the generic
  `zorch.fused_region` marker lifts to operands harmlessly (no name-routed ABI).
"""

from __future__ import annotations

import functools
import operator

import frx.numpy as fnp
from frx import Array


def _unrolled_sum(terms: list[Array]) -> Array:
    return functools.reduce(operator.add, terms)


def apply_dense_mds(mds_rows: tuple[tuple[int, ...], ...], state: Array) -> Array:
    """Dense MDS layer `mds @ state`: row `i` is `sum_j mds[i][j] * state[j]`.

    `mds_rows` is the `width x width` matrix as canonical Python ints (rows of
    ints), so lanes scale by integer literals and no field array is captured —
    required inside a name-routed `fused_region`. The unrolled per-lane sum keeps
    the layer reduction-free (no `fnp.dot`/`fnp.sum`/gather), so the round body
    lowers to a single fused kernel.
    """
    w = state.shape[0]
    if state.ndim != 1 or w == 0 or len(mds_rows) != w:
        raise ValueError(
            f"dense MDS needs a 1-D state matching a square matrix, got state "
            f"{state.shape}, matrix rows {len(mds_rows)}"
        )
    return fnp.stack(
        [_unrolled_sum([mds_rows[i][j] * state[j] for j in range(w)]) for i in range(w)]
    )


def apply_matrix(matrix: Array, state: Array) -> Array:
    """Dense layer `matrix @ state`, as the sum of each column scaled by its lane.

    The field-array sibling of `apply_dense_mds` — for a matrix whose entries are
    full-field (an int64 literal would overflow), so it rides as a field array."""
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D, got shape {state.shape}")
    w = state.shape[0]
    if matrix.shape != (w, w):
        raise ValueError(
            f"need a square matrix matching 1-D state, got matrix {matrix.shape}, "
            f"state {state.shape}"
        )
    return _unrolled_sum([matrix[:, j] * state[j] for j in range(w)])


def apply_sparse_partial(
    dot_row: Array,
    col_vec: Array,
    active: Array,
    tail: Array,
) -> Array:
    """The optimized-sparse partial round's linear layer, in normal form.

    With `a` the post-S-box lane-0 value (`active`) and `tail = state[1:]`:

        out[0]   = a*dot_row[0] + sum_{j>=1} tail[j-1]*dot_row[j]   (dense lane-0 row)
        out[t]   = tail[t-1] + a*col_vec[t-1]     for t = 1 .. width-1

    i.e. lane 0 gathers a full dot over the state while lanes 1.. only add a
    rank-1 correction from lane 0. `dot_row` (width) and `col_vec` (width-1) are
    field-array constants; the unrolled sum keeps the layer reduction- and
    gather-free. Uses `concatenate`, not `.at[0].set`, so no scatter splits the
    kernel.
    """
    if dot_row.ndim != 1 or col_vec.ndim != 1:
        raise ValueError(
            f"dot_row and col_vec must be 1-D, got {dot_row.shape}, {col_vec.shape}"
        )
    w = dot_row.shape[0]
    if col_vec.shape[0] != w - 1:
        raise ValueError(
            f"col_vec must have width-1 entries, got {col_vec.shape[0]} for width {w}"
        )
    out0 = _unrolled_sum(
        [dot_row[0] * active] + [dot_row[j] * tail[j - 1] for j in range(1, w)]
    )
    out_rest = fnp.stack([tail[t] + col_vec[t] * active for t in range(w - 1)])
    return fnp.concatenate([out0[None], out_rest])
