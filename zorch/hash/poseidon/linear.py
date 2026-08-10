"""Normal-form Poseidon linear layers — explicit field add/mul, no dot/reduce/gather.

Every linear layer is a fixed, unrolled sum of column-scaled lanes so a round
body stays straight-line element-wise and fuses to one kernel: `fnp.dot`/`fnp.sum`
lower to a reduction (the `kInput` fusion boundary) and dynamic indexing to
`gather`, either of which splits the kernel. The summation primitive
`unrolled_sum` and the field-array dense layer `apply_matrix` are shared with
poseidon2 in `zorch.hash.linear`; this module adds the Poseidon-specific forms.

Two matrix forms, by where the matrix rides:
- **Integer-literal** (`apply_dense_mds`, `apply_sparse_partial_ints`) — the
  matrix is canonical ints, so no field array is captured; required inside a
  name-routed `fused_region`, where a closed-over array lifts to a leading operand
  and breaks the emitter's ABI (the structure rides as an int64 marker attribute
  instead). The dedicated `zorch.sparse_poseidon` emitter's reference body uses
  these; entries past the int-literal staging cap are assembled in field
  arithmetic by `_scale`, so any u64-representable field works.
- **Field-array** (`apply_matrix` (shared), `apply_sparse_partial`) — the matrix
  stays a field array, which the generic `zorch.fused_region` marker lifts to an
  operand harmlessly (no name-routed ABI). This is the optimized-sparse variant's
  readable body when the dedicated emitter is absent, and the only form for fields
  whose entries exceed an int64 literal.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

from zorch.hash.linear import unrolled_sum

# Widest canonical constant that can ride as a bare Python-int literal: without
# x64, frx stages a plain int operand as an int32. A wider constant is
# assembled in field arithmetic by `_scale` instead.
_LITERAL_MAX = 2**31 - 1


def _scale(c: int, x: Array) -> Array:
    """`c * x` for a canonical int `c` of any width the field holds.

    Below `_LITERAL_MAX` this is the plain literal multiply. A wider `c` —
    Goldilocks canonical values reach `2^64 - 2^32` — cannot stage as an int
    literal, and staging it as an array constant would have the composite lift
    it to a leading operand and break the dedicated emitter's ABI. So it is
    rebuilt in field arithmetic from 16-bit literals: Horner over base `2^16`,
    seeded from the const-free field zero `x - x`. All intermediates are `< p`
    reductions of the same integer, so the assembled value is exactly `c`.
    """
    if c <= _LITERAL_MAX:
        return c * x
    limbs = []  # little-endian base-2^16
    v = c
    while v:
        limbs.append(v & 0xFFFF)
        v >>= 16
    acc = (x - x) + limbs[-1]
    for limb in reversed(limbs[:-1]):
        acc = acc * 0x10000 + limb
    return acc * x


def apply_dense_mds(mds_rows: tuple[tuple[int, ...], ...], state: Array) -> Array:
    """Dense MDS layer `mds @ state`: row `i` is `sum_j mds[i][j] * state[j]`.

    `mds_rows` is the `width x width` matrix as canonical Python ints (rows of
    ints), so lanes scale by integer literals and no field array is captured —
    required inside a name-routed `fused_region`. The unrolled per-lane sum keeps
    the layer reduction-free (no `fnp.dot`/`fnp.sum`/gather), so the round body
    lowers to a single fused kernel.
    """
    if state.ndim != 1:
        raise ValueError(f"dense MDS needs a 1-D state, got shape {state.shape}")
    w = state.shape[0]
    if w == 0 or len(mds_rows) != w:
        raise ValueError(
            f"dense MDS needs a 1-D state matching a square matrix, got state "
            f"{state.shape}, matrix rows {len(mds_rows)}"
        )
    # Hoist the lanes once. Indexing `state` inside both loops reads the chained
    # input w**2 times, which is the same per-lane fan-out the chained-input rule
    # in `zorch.hash.linear` forbids, squared.
    lanes = [state[j] for j in range(w)]
    return fnp.stack(
        [
            unrolled_sum([_scale(mds_rows[i][j], lanes[j]) for j in range(w)])
            for i in range(w)
        ]
    )


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
    if active.ndim != 0:
        raise ValueError(
            f"active (post-S-box lane 0) must be a scalar, got shape {active.shape}"
        )
    if tail.shape != (w - 1,):
        raise ValueError(
            f"tail (state[1:]) must have width-1 entries, got {tail.shape} for "
            f"width {w}"
        )
    # Array-shaped so `active` is read twice rather than once per lane — the
    # chained-input rule in `zorch.hash.linear`.
    prods = dot_row[1:] * tail
    out0 = unrolled_sum([dot_row[0] * active, *prods])
    out_rest = tail + col_vec * active
    return fnp.concatenate([out0[None], out_rest])


def apply_sparse_partial_ints(
    dot_row: tuple[int, ...],
    col_vec: tuple[int, ...],
    active: Array,
    tail: Array,
) -> Array:
    """Integer-literal twin of `apply_sparse_partial`, for the dedicated
    `zorch.sparse_poseidon` emitter's reference body.

    `dot_row` (width ints) and `col_vec` (width-1 ints) are canonical Python ints,
    so the lanes scale by integer literals and no field array is captured — a
    name-routed `fused_region` would lift a closed-over array to a leading operand
    and break the emitter's operand ABI (the sparse structure rides as an int64
    marker attribute instead). Same arithmetic and reduction-free normal form as
    `apply_sparse_partial`; this is the sparse-partial sibling of `apply_dense_mds`.

    Unlike its twin this reads `active` once per lane, against the chained-input
    rule in `zorch.hash.linear`. Broadcasting `active` to a vector first would
    hold the rule without capturing an array, but costs ~23% more traced
    equations, and nothing chains this body: every field that reaches it routes
    to the dedicated emitter, which replaces this reference form at compile
    time. Give it the broadcast if a consumer ever executes this shape.
    """
    w = len(dot_row)
    if len(col_vec) != w - 1:
        raise ValueError(
            f"col_vec must have width-1 entries, got {len(col_vec)} for width {w}"
        )
    if active.ndim != 0:
        raise ValueError(
            f"active (post-S-box lane 0) must be a scalar, got shape {active.shape}"
        )
    if tail.shape != (w - 1,):
        raise ValueError(
            f"tail (state[1:]) must have width-1 entries, got {tail.shape} for "
            f"width {w}"
        )
    out0 = unrolled_sum(
        [_scale(dot_row[0], active)]
        + [_scale(dot_row[j], tail[j - 1]) for j in range(1, w)]
    )
    out_rest = fnp.stack([tail[t] + _scale(col_vec[t], active) for t in range(w - 1)])
    return fnp.concatenate([out0[None], out_rest])
