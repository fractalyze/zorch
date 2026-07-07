# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Multi-round block binaries (xla#179 host-wall): K rounds + their FS
hops chained in one cached `jax.export` binary per (phase, K), dividing
the decoupled prove's per-round host dispatch cost by K."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, export

from zorch.logup_gkr._jagged_buffers import (
    _resize_zero,
)
from zorch.logup_gkr._jagged_composites import (
    _composite_fix_and_sum_boundary,
    _composite_fix_and_sum_dense,
    _composite_fix_and_sum_row,
    _composite_fix_last,
    _composite_sum_as_poly_row,
)
from zorch.logup_gkr._jagged_export import (
    _ROUND_SYM_MAX,
    _abst_scalars,
    _round_dispatch,
)
from zorch.logup_gkr._jagged_fs import (
    _fs_reduce,
)
from zorch.logup_gkr._jagged_types import (
    _InterpConsts,
    _Planes,
    _RoundScalars,
)
from zorch.transcript import (
    DuplexState,
    DuplexTranscript,
    _state_leaves,
)

# ============================================================================
# Multi-round blocks (xla#179 host-wall): the decoupled prove's ~330 rounds
# each pay one `call_exported` bind (~55us) plus one FS-zone dispatch -- a
# ~200 ms host wall while the GPU is busy well under half that. A block binary
# runs K uniform mid rounds (round compute + `_fs_reduce` FS hop, challenge
# chained in-trace) per bind, dividing the host wall by K and letting XLA fuse
# the inter-round repack glue that separate binaries must materialize. K is
# static (the in-binary loop unrolls), so the greedy ladder below keeps the
# binary census O(1): one binary per (phase, K, dtype-mix) -- never per shard,
# layer, or position. Under the fixed caps every operand shape is
# round-invariant, which is exactly what lets one K-block serve every stretch.
# 1 stays in the ladder so no stretch tail ever falls back to a single round
# plus its separate FS-zone dispatch -- a k=1 block is still one executable
# where the single-round path is two.
_ROUND_BLOCK_SIZES = (8, 4, 2, 1)


def _block_fs_key(transcript: DuplexTranscript, challenge_limbs: int) -> tuple:
    """The FS-config part of a block binary's cache key. The block bakes the
    permutation's constants and the fs backend into its trace (the singles
    never did -- their FS ran outside), so the key must carry them.
    `Poseidon2.__eq__/__hash__` are value-based, making the in-memory dict
    exact; these objects have no stable cross-process repr, which is why block
    binaries skip the on-disk cache (`_round_dispatch(disk=False)`)."""
    return (transcript.permutation, transcript.rate, transcript.fs, challenge_limbs)


def _dispatch_row_block(
    planes: _Planes,
    eq_row: Array,
    alpha: Array,
    row_counts: Array,
    eq_int: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_block: Array,
    transcript: DuplexTranscript,
    eval_point: Array,
    pos: Array,
    challenge_limbs: int,
    first: bool = False,
) -> tuple[
    Array, Array, _Planes, Array, DuplexTranscript, Array, Array, Array, Array, Array
]:
    """Dispatch `k = live_block.shape[0]` consecutive capped row rounds --
    each a `fix_and_sum_row` marker plus its `_fs_reduce` FS hop, the round
    challenge chained in-trace -- through ONE cached binary. Only the capped
    (width-preserving, `out_pairs is None`) route exists in block form: the
    exact layout changes width per round, so it keeps the single-round path.

    `first` makes iteration 0 the layer's round 0 (`sum_as_poly`, no fold, no
    eq_row change; `alpha` rides unused into the binary), so the round-0
    single AND its FS-zone dispatch fold into the block -- one executable per
    layer head instead of three.

    The transcript crosses the boundary as its five `DuplexState` leaves (the
    permutation / rate / fs metadata is baked into the trace and carried in
    the key via `_block_fs_key`). Returns the stacked `(k, DEGREE+1)` round
    polys, the `(k,)` challenges, and the advanced carry
    `(planes, eq_row, transcript, claim, pad_adj, z_cur, pos, last_r)`."""
    k = live_block.shape[0]
    state_ops = _state_leaves(transcript.state)
    operands = (
        planes,
        eq_row,
        alpha,
        row_counts,
        eq_int,
        scalars,
        live_block,
        state_ops,
        eval_point,
        pos,
    )
    key = (
        "row_block",
        k,
        first,
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        row_counts.shape,
        eq_int.shape,
        eval_point.shape,
        tuple(leaf.shape for leaf in state_ops),
        consts.naturals.shape[0],
        consts.naturals.dtype,
        _block_fs_key(transcript, challenge_limbs),
    )

    def build() -> export.Exported:
        pp, rr = export.symbolic_shape(
            "pp, rr",
            constraints=[
                "pp >= 1",
                f"pp <= {_ROUND_SYM_MAX}",
                "rr >= 1",
                f"rr <= {_ROUND_SYM_MAX}",
            ],
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((2 * pp,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((2 * rr,), eq_row.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            jax.ShapeDtypeStruct(row_counts.shape, row_counts.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct(live_block.shape, live_block.dtype),
            tuple(jax.ShapeDtypeStruct(s.shape, s.dtype) for s in state_ops),
            jax.ShapeDtypeStruct(eval_point.shape, eval_point.dtype),
            jax.ShapeDtypeStruct(pos.shape, pos.dtype),
        )
        template = transcript

        def fn(
            pl: _Planes,
            er: Array,
            al: Array,
            rc: Array,
            ei: Array,
            sc: _RoundScalars,
            lv: Array,
            st: tuple[Array, Array, Array, Array, Array],
            ep: Array,
            po: Array,
        ) -> tuple[
            Array,
            Array,
            _Planes,
            Array,
            tuple[Array, Array, Array, Array, Array],
            Array,
            Array,
            Array,
            Array,
            Array,
        ]:
            t = replace(template, state=DuplexState(*st))
            dtype = sc.claim.dtype
            pad_adj, z_cur, claim = sc.pad_adj, sc.z_cur, sc.claim
            prev = al
            polys: list[Array] = []
            rs: list[Array] = []
            # eq_adj / lam are row-stretch constants (the eq_adj swap happens
            # at the row->boundary handoff, outside any block), so each
            # iteration rebuilds the scalars bundle around the moving trio.
            for i in range(k):
                sci = _RoundScalars(sc.eq_adj, pad_adj, z_cur, claim, sc.lam)
                if first and i == 0:
                    # The layer's round 0: bare sum, no fold, eq_row untouched.
                    poly, pl = _composite_sum_as_poly_row(
                        pl, rc, er, ei, sci, consts, lv[i], None
                    )
                else:
                    poly, pl, er = _composite_fix_and_sum_row(
                        pl, er, prev, rc, ei, sci, consts, lv[i], None
                    )
                t, r, claim, pad_adj, z_cur, po = _fs_reduce(
                    poly, t, pad_adj, z_cur, ep, po, challenge_limbs, dtype
                )
                polys.append(poly)
                rs.append(r)
                prev = r
            return (
                jnp.stack(polys),
                jnp.stack(rs),
                pl,
                er,
                _state_leaves(t.state),
                claim,
                pad_adj,
                z_cur,
                po,
                prev,
            )

        return export.export(jax.jit(fn))(*abst)

    out = _round_dispatch(key, operands, build, disk=False)
    polys, rs, planes, eq_row, st, claim, pad_adj, z_cur, pos, prev = out
    return (
        polys,
        rs,
        planes,
        eq_row,
        replace(transcript, state=DuplexState(*st)),
        claim,
        pad_adj,
        z_cur,
        pos,
        prev,
    )


def _dispatch_int_block(
    planes: _Planes,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_block: Array,
    transcript: DuplexTranscript,
    eval_point: Array,
    pos: Array,
    challenge_limbs: int,
) -> tuple[
    Array, Array, _Planes, Array, DuplexTranscript, Array, Array, Array, Array, Array
]:
    """`_dispatch_row_block` for `k` consecutive capped dense interaction
    rounds (`fix_and_sum_int` + `_fs_reduce` each, challenge chained
    in-trace). The state and `eq_int` share the dense rounds' `4*g` symbol
    exactly like the single-round dispatch."""
    k = live_block.shape[0]
    state_ops = _state_leaves(transcript.state)
    operands = (planes, eq_int, alpha, scalars, live_block, state_ops, eval_point, pos)
    key = (
        "int_block",
        k,
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        eval_point.shape,
        tuple(leaf.shape for leaf in state_ops),
        consts.naturals.shape[0],
        consts.naturals.dtype,
        _block_fs_key(transcript, challenge_limbs),
    )

    def build() -> export.Exported:
        (g,) = export.symbolic_shape(
            "g", constraints=["g >= 1", f"g <= {_ROUND_SYM_MAX}"]
        )
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct((4 * g,), getattr(planes, f).dtype)
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct((4 * g,), eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct(live_block.shape, live_block.dtype),
            tuple(jax.ShapeDtypeStruct(s.shape, s.dtype) for s in state_ops),
            jax.ShapeDtypeStruct(eval_point.shape, eval_point.dtype),
            jax.ShapeDtypeStruct(pos.shape, pos.dtype),
        )
        template = transcript

        def fn(
            pl: _Planes,
            ei: Array,
            al: Array,
            sc: _RoundScalars,
            lv: Array,
            st: tuple[Array, Array, Array, Array, Array],
            ep: Array,
            po: Array,
        ) -> tuple[
            Array,
            Array,
            _Planes,
            Array,
            tuple[Array, Array, Array, Array, Array],
            Array,
            Array,
            Array,
            Array,
            Array,
        ]:
            t = replace(template, state=DuplexState(*st))
            dtype = sc.claim.dtype
            pad_adj, z_cur, claim = sc.pad_adj, sc.z_cur, sc.claim
            prev = al
            polys: list[Array] = []
            rs: list[Array] = []
            for i in range(k):
                sci = _RoundScalars(sc.eq_adj, pad_adj, z_cur, claim, sc.lam)
                poly, pl, ei = _composite_fix_and_sum_dense(
                    pl, ei, prev, sci, consts, lv[i]
                )
                t, r, claim, pad_adj, z_cur, po = _fs_reduce(
                    poly, t, pad_adj, z_cur, ep, po, challenge_limbs, dtype
                )
                polys.append(poly)
                rs.append(r)
                prev = r
            return (
                jnp.stack(polys),
                jnp.stack(rs),
                pl,
                ei,
                _state_leaves(t.state),
                claim,
                pad_adj,
                z_cur,
                po,
                prev,
            )

        return export.export(jax.jit(fn))(*abst)

    out = _round_dispatch(key, operands, build, disk=False)
    polys, rs, planes, eq_int, st, claim, pad_adj, z_cur, pos, prev = out
    return (
        polys,
        rs,
        planes,
        eq_int,
        replace(transcript, state=DuplexState(*st)),
        claim,
        pad_adj,
        z_cur,
        pos,
        prev,
    )


def _dispatch_boundary_block(
    planes: _Planes,
    eq_boundary: Array,
    eq_int: Array,
    alpha: Array,
    scalars: _RoundScalars,
    consts: _InterpConsts,
    live_block: Array,
    transcript: DuplexTranscript,
    eval_point: Array,
    pos: Array,
    challenge_limbs: int,
    final: bool,
) -> tuple[
    Array,
    Array,
    _Planes,
    Array,
    DuplexTranscript,
    Array,
    Array,
    Array,
    Array,
    Array,
    tuple[Array, Array, Array, Array] | None,
]:
    """Dispatch the row->interaction handoff plus `k-1` dense rounds through
    ONE cached binary: iteration 0 is the boundary round (bind the last row
    challenge over the still-unfolded `eq_boundary`, then resize the halved
    state down to the interaction cap in-trace), iterations 1.. are
    `fix_and_sum_int`, each with its `_fs_reduce` FS hop chained in-trace.
    With `final` (the block reaches the layer's last round) the tail also
    folds `fix_last`, returning the four pair openings -- the whole
    boundary+dense stretch of a typical layer collapses to one executable.

    `live_block` rows halve from the handoff's `1 << (niv-1)` pairs exactly
    like the loop's per-round `_dense_live_operand` sequence."""
    k = live_block.shape[0]
    state_ops = _state_leaves(transcript.state)
    operands = (
        planes,
        eq_boundary,
        eq_int,
        alpha,
        scalars,
        live_block,
        state_ops,
        eval_point,
        pos,
    )
    key = (
        "boundary_block",
        k,
        final,
        tuple(leaf.dtype for leaf in jax.tree_util.tree_leaves(operands)),
        planes.n0.shape,
        eq_boundary.shape,
        eq_int.shape,
        eval_point.shape,
        tuple(leaf.shape for leaf in state_ops),
        consts.naturals.shape[0],
        consts.naturals.dtype,
        _block_fs_key(transcript, challenge_limbs),
    )

    def build() -> export.Exported:
        # Concrete shapes, not the singles' symbolic dims: the in-trace
        # resize to the interaction cap compares the plane width against a
        # concrete cap, which shape polymorphism cannot decide -- and under
        # the fixed caps a symbol would bind exactly one size anyway, so the
        # census is identical (the shapes join the cache key above).
        abst = (
            _Planes(
                *(
                    jax.ShapeDtypeStruct(
                        getattr(planes, f).shape, getattr(planes, f).dtype
                    )
                    for f in ("n0", "n1", "d0", "d1")
                )
            ),
            jax.ShapeDtypeStruct(eq_boundary.shape, eq_boundary.dtype),
            jax.ShapeDtypeStruct(eq_int.shape, eq_int.dtype),
            jax.ShapeDtypeStruct((), alpha.dtype),
            _abst_scalars(scalars),
            jax.ShapeDtypeStruct(live_block.shape, live_block.dtype),
            tuple(jax.ShapeDtypeStruct(s.shape, s.dtype) for s in state_ops),
            jax.ShapeDtypeStruct(eval_point.shape, eval_point.dtype),
            jax.ShapeDtypeStruct(pos.shape, pos.dtype),
        )
        template = transcript

        def fn(
            pl: _Planes,
            eb: Array,
            ei: Array,
            al: Array,
            sc: _RoundScalars,
            lv: Array,
            st: tuple[Array, Array, Array, Array, Array],
            ep: Array,
            po: Array,
        ) -> tuple[Any, ...]:
            t = replace(template, state=DuplexState(*st))
            dtype = sc.claim.dtype
            pad_adj, z_cur, claim = sc.pad_adj, sc.z_cur, sc.claim
            interaction = ei.shape[0]
            prev = al
            polys: list[Array] = []
            rs: list[Array] = []
            for i in range(k):
                sci = _RoundScalars(sc.eq_adj, pad_adj, z_cur, claim, sc.lam)
                if i == 0:
                    poly, pl, _ = _composite_fix_and_sum_boundary(
                        pl, eb, prev, sci, consts, lv[i]
                    )
                    # The handoff halves [row] -> [row // 2]; the dense
                    # rounds run at the interaction cap, so resize in-trace
                    # (the live 2^niv prefix always survives -- the same
                    # contract as the single-round loop's resize).
                    pl = _Planes(
                        *(
                            _resize_zero(a, interaction)
                            for a in (pl.n0, pl.n1, pl.d0, pl.d1)
                        )
                    )
                else:
                    poly, pl, ei = _composite_fix_and_sum_dense(
                        pl, ei, prev, sci, consts, lv[i]
                    )
                t, r, claim, pad_adj, z_cur, po = _fs_reduce(
                    poly, t, pad_adj, z_cur, ep, po, challenge_limbs, dtype
                )
                polys.append(poly)
                rs.append(r)
                prev = r
            outs: list[Any] = [
                jnp.stack(polys),
                jnp.stack(rs),
                pl,
                ei,
                _state_leaves(t.state),
                claim,
                pad_adj,
                z_cur,
                po,
                prev,
            ]
            if final:
                head = _Planes(*(a[:2] for a in (pl.n0, pl.n1, pl.d0, pl.d1)))
                outs.extend(_composite_fix_last(head, prev))
            return tuple(outs)

        return export.export(jax.jit(fn))(*abst)

    out = _round_dispatch(key, operands, build, disk=False)
    if final:
        (polys, rs, pl, ei, st, claim, pad_adj, z_cur, po, prev, f0, f1, f2, f3) = out
        openings: tuple[Array, Array, Array, Array] | None = (f0, f1, f2, f3)
    else:
        polys, rs, pl, ei, st, claim, pad_adj, z_cur, po, prev = out
        openings = None
    return (
        polys,
        rs,
        pl,
        ei,
        replace(transcript, state=DuplexState(*st)),
        claim,
        pad_adj,
        z_cur,
        po,
        prev,
        openings,
    )
