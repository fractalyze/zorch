# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged PCS prover: commit + the S1-S6 opening seam.

`JaggedPcsProver` owns both halves of the prover:

* `commit` packs variable-height blocks into the dense stacked MLE `D`, commits
  it via the injected multilinear `PcsProver` (BaseFold), binds the jagged
  structure into the root, and retains `JaggedProverData` (the BaseFold witness,
  `D`, the layout, and the per-column prefix-sum bit tensor) for `open`.
* `open` runs the four-step opening that ties the prior pieces together — sample
  `z_col`, materialize the jagged indicator `J̃` (S2), run the outer Hadamard
  sumcheck `Σ_i D(i)·J̃(i) = claim` (S3), reprove `J̃(z_row, z_col, z_final)` via
  the inner jagged-assist sumcheck (S4), and open `D(z_final)` via the stacked
  BaseFold opening (S6) — all in one Fiat-Shamir transcript so the dual verifier
  replays in lockstep.

The seam requires the dense MLE and the jagged indicator to share a variable
count: `layout.log_m == cfg.n_d`. The dense buffer `D` is `2^log_m`; the
indicator `J̃` from `partial_eval` is `2^n_d`; the outer sumcheck Hadamard-pairs
them index-for-index (the `from_blocks` column-major packing puts column `c`'s
row `r` at flat index `t_c + r`, exactly where `J̃` is nonzero), so they must be
the same length. `open` asserts the match; the consumer picks
`log_stacking_height` to satisfy it (with this convention `K == 1`).
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_dataclass

from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData
from zorch.pcs.jagged.config import JaggedOpeningProof
from zorch.pcs.jagged.dense import JaggedLayout, from_blocks
from zorch.pcs.jagged.inner_sumcheck import prove_jagged_eval
from zorch.pcs.jagged.poly import (
    JaggedStaticConfig,
    build_jagged_layout,
    build_prefix_sums,
    msb_first_bits,
    partial_eval,
)
from zorch.pcs.jagged.stacked import stacked_open
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.prove import prove
from zorch.sumcheck.prover import SumcheckRound
from zorch.transcript import Transcript
from zorch.utils.bits import log2_ceil_usize

# Outer Hadamard sumcheck degree: Σ_i D(i)·J̃(i), a product of two MLEs.
_OUTER_DEGREE = 2


def _expand_col_heights(layout: JaggedLayout) -> list[int]:
    """Per-column heights for the indicator: each block `[h, w]` is `w` unit-width
    columns of height `h` (matching `from_blocks`' column-major packing)."""
    return [h for h, w in zip(layout.heights, layout.widths) for _ in range(w)]


def _offset_bit_tensor(
    col_heights: list[int], l_max: int, cfg: JaggedStaticConfig
) -> Array:
    """`(l_max+1, n_d)` prefix-sum bit tensor whose canonical int32 limb-0 holds
    each MSB-first bit, typed as `cfg.dtype`.

    `partial_eval` derives its integer scatter offsets by bitcasting the bit
    tensor to int32 and reading limb 0 — it needs the *canonical* bit there. A
    Montgomery field dtype encodes `astype(1)` as `R mod p` (limb 0 ≠ 1), so
    `build_jagged_layout`'s field-valued tensor (correct for the inner sumcheck's
    field arithmetic) misreads under that raw bitcast. Build a separate tensor
    by packing the bits into int32 limb 0 (other limbs zero) and bitcasting to
    the field dtype — its raw bytes give the right offsets regardless of the
    field's Montgomery-ness. Limb count is derived from the dtype's storage.
    """
    prefix = build_prefix_sums(col_heights)  # length len+1
    padded = prefix + [prefix[-1]] * (l_max - len(col_heights))  # empty-range pad
    bits = msb_first_bits(padded, cfg.n_d)  # (l_max+1, n_d) int
    # int32 limbs per field element (4 for a 128-bit EF, 1 for a 32-bit base).
    probe = jax.lax.bitcast_convert_type(jnp.zeros((1,), cfg.dtype), jnp.int32)
    n_limbs = probe.shape[-1] if probe.ndim > 1 else 1
    limbs = np.zeros((bits.shape[0], cfg.n_d, n_limbs), dtype=np.int32)
    limbs[..., 0] = bits
    return jax.lax.bitcast_convert_type(jnp.asarray(limbs), cfg.dtype)


def _indicator_inputs(
    layout: JaggedLayout,
) -> tuple[Array, Array, JaggedStaticConfig]:
    """`(col_prefix_sums, col_prefix_sums_offsets, cfg)` for a committed region.

    `col_prefix_sums` is the field-valued bit tensor `build_jagged_layout`
    produces (correct for the inner sumcheck + `eval_jagged_mle` arithmetic);
    `col_prefix_sums_offsets` is the canonical-limb sibling `partial_eval`'s raw
    offset bitcast needs (see `_offset_bit_tensor`). `l_max` is the exact column
    count `L = Σ widths`; `n_r` covers the tallest column. The
    `cfg.n_d == layout.log_m` requirement is enforced in `open` (commit itself
    does not need it).
    """
    col_heights = _expand_col_heights(layout)
    l_max = max(len(col_heights), 1)
    n_r = max(log2_ceil_usize(max(col_heights, default=1)), 1)
    col_prefix_sums, cfg = build_jagged_layout(
        col_heights, l_max=l_max, n_r=n_r, dtype=layout.dtype
    )
    offsets = _offset_bit_tensor(col_heights, l_max, cfg)
    return col_prefix_sums, offsets, cfg


@partial(
    register_dataclass,
    data_fields=[
        "basefold_prover_data",
        "dense",
        "col_prefix_sums",
        "col_prefix_sums_offsets",
    ],
    meta_fields=["layout", "cfg"],
)
@dataclass(frozen=True)
class JaggedProverData:
    """Retained witness from `JaggedPcsProver.commit`, everything `open` needs.

    basefold_prover_data: the underlying BaseFold witness over the dense `[S, K]`
        matrix (the stacked opening reads it).
    dense: the dense MLE `D`, shape `(2^log_m,)` (= `(2^n_d,)`); the outer
        sumcheck Hadamard-pairs it with the indicator and the stacked opening
        proves `D(z_final)`.
    col_prefix_sums: the `(l_max+1, n_d)` field-valued prefix-sum bit tensor (the
        S4 inner sumcheck + `eval_jagged_mle` read it as field elements).
    col_prefix_sums_offsets: the canonical-limb sibling for `partial_eval`'s raw
        offset bitcast (S2) — see `_offset_bit_tensor`.
    layout / cfg: static structure metadata (jagged layout + indicator config).

    A registered pytree (layout/cfg static) so it threads a `@jit` boundary.
    """

    basefold_prover_data: BasefoldProverData
    dense: Array
    col_prefix_sums: Array
    col_prefix_sums_offsets: Array
    layout: JaggedLayout
    cfg: JaggedStaticConfig


class JaggedPcsProver:
    """Commits variable-height blocks via an injected multilinear `PcsProver` and
    opens them at a row point, binding the jagged structure into the commitment.

    `sponge` / `compressor` perform the structure bind (the jagged layer owns it,
    not the PCS). The jit'd device commit zone is built once in `__init__`,
    capturing `prover` / `sponge` / `compressor`, keyed on the MLE + structure
    shapes, so it compiles once per (area tier, block count).
    """

    _commit: Any  # jax.jit-wrapped device-commit closure, built in __init__

    def __init__(
        self, prover: BasefoldProver, sponge: Sponge, compressor: Compression
    ) -> None:
        if sponge.out != compressor.chunk:
            raise ValueError(
                f"sponge digest ({sponge.out}) must equal compressor chunk "
                f"({compressor.chunk}) for the structure bind"
            )
        if compressor.arity != 2:
            raise ValueError(
                f"compressor.arity must be 2 for the (root, structure_hash) "
                f"bind, got {compressor.arity}"
            )
        self.prover = prover
        self.sponge = sponge
        self.compressor = compressor

        @jax.jit
        def _commit(mle: Array, structure: Array) -> tuple[Array, Any]:
            # The seam commits a Sequence of column MLEs; BaseFold binds them
            # jointly under one root (a matrix commitment), which the structure
            # bind below hashes against.
            columns = [mle[:, j] for j in range(mle.shape[1])]
            root, prover_data = prover.commit(columns)
            structure_hash = sponge.hash(structure)
            bound = compressor.compress(jnp.stack([root, structure_hash]))
            return bound, prover_data

        self._commit = _commit

    @property
    def bf_prover(self) -> BasefoldProver:
        return self.prover

    def commit(
        self, blocks: list[Array], *, log_stacking_height: int
    ) -> tuple[Array, JaggedProverData]:
        """Pack + commit; return `(commitment, JaggedProverData)`.

        `from_blocks` (host) derives the tier and the stacked MLE `[S, K]`; the
        `@jit` zone commits it via `pcs` and binds the structure hash into the
        root. The retained `JaggedProverData` carries the dense buffer, the
        layout, and the per-column prefix-sum bit tensor for `open`.
        """
        packed, layout = from_blocks(blocks, log_stacking_height=log_stacking_height)
        mle = packed.reshape(layout.K, layout.S).T  # [S, K]
        structure = _structure_vec(layout)
        col_prefix_sums, offsets, cfg = _indicator_inputs(layout)
        bound, basefold_prover_data = self._commit(mle, structure)
        prover_data = JaggedProverData(
            basefold_prover_data=basefold_prover_data,
            dense=packed,
            col_prefix_sums=col_prefix_sums,
            col_prefix_sums_offsets=offsets,
            layout=layout,
            cfg=cfg,
        )
        return bound, prover_data

    def open(
        self,
        prover_data: JaggedProverData,
        z_row: Array,
        column_claims: Array,
        transcript: Transcript,
    ) -> tuple[JaggedOpeningProof, Transcript]:
        """Open the committed region at `z_row`.

        `column_claims` is the `(L,)` vector of per-column claims
        `{MLE_c(z_row)}` (one per unit-width column, `L = Σ widths`), in the
        flat column order `from_blocks` packs — passed for transcript symmetry
        with `verify` (it samples `z_col` then folds the claim); zorch's stock
        sumcheck is seed-free, so the prover derives the round polys straight
        from `D`·`J̃` and never reads `column_claims`. Returns
        `(proof, transcript)`.
        """
        del column_claims  # symmetry with verify; the seed-free sumcheck needs it not
        cfg = prover_data.cfg
        layout = prover_data.layout
        col_prefix_sums = prover_data.col_prefix_sums
        if cfg.n_d != layout.log_m:
            raise ValueError(
                f"jagged open requires cfg.n_d == layout.log_m; got n_d={cfg.n_d}, "
                f"log_m={layout.log_m}. Choose log_stacking_height so the dense MLE "
                f"and the jagged indicator share a variable count (log_m == n_d)."
            )

        # S1: sample z_col. The transcript squeezes base-field scalars; embed them
        # in the indicator's extension field so the jagged arithmetic stays
        # single-field (base·EF mixed-stacks fail to promote inside the branching
        # program). z_col must be sampled here (not in verify alone) so both
        # transcripts advance identically before the outer sumcheck.
        transcript, z_col = transcript.sample(cfg.n_c)
        z_col = z_col.astype(cfg.dtype)

        # S2: materialize the jagged indicator J̃(z_row, z_col, ·). `partial_eval`
        # reads its scatter offsets from the canonical-limb tensor.
        indicator = partial_eval(
            prover_data.col_prefix_sums_offsets, z_row, z_col, cfg=cfg
        )

        # S3: outer Hadamard sumcheck Σ_i D(i)·J̃(i) = claim.
        folded, transcript, msgs = prove(
            SumcheckRound(_OUTER_DEGREE), [prover_data.dense, indicator], transcript
        )
        outer_sumcheck_polys = msgs.round_poly  # [n_d, degree+1]
        # zorch's stock sumcheck folds MSB-first (buf[:half] = first variable),
        # so the challenge stream is already MSB→LSB — the order `eval_mle` /
        # the stacked opening / the inner BP all read. (The whir-zorch reference
        # folds LSB-first with insert-at-front and reverses; zorch does not.)
        z_final = msgs.challenge.astype(cfg.dtype)
        dense_eval = folded[0][0]  # D(z_final)
        outer_final_eval = folded[0][0] * folded[1][0]  # D(z_final)·J̃(z_final)

        # S4: inner jagged-assist sumcheck reproving J̃(z_row, z_col, z_final).
        inner_proof, inner_claimed_sum, transcript = prove_jagged_eval(
            col_prefix_sums, z_row, z_col, z_final, cfg=cfg, transcript=transcript
        )

        # S6: stacked BaseFold opening of D(z_final).
        dense_eval2, column_values, basefold_proof, transcript = stacked_open(
            self.bf_prover,
            prover_data.basefold_prover_data,
            z_final,
            layout,
            transcript,
        )
        # dense_eval2 is D(z_final) recomputed from the column evals; it must
        # equal the outer sumcheck's reduced D(z_final).
        del dense_eval2

        # The un-bound BaseFold root the jagged commitment binds; the verifier
        # re-derives the bind from it and feeds it to the stacked verify.
        basefold_root = prover_data.basefold_prover_data.digest_layers[-1][0]

        proof = JaggedOpeningProof(
            outer_sumcheck_polys=outer_sumcheck_polys,
            outer_final_eval=outer_final_eval,
            inner_proof=inner_proof,
            inner_claimed_sum=inner_claimed_sum,
            dense_eval=dense_eval,
            column_values=column_values,
            basefold_root=basefold_root,
            basefold_proof=basefold_proof,
        )
        return proof, transcript


def _structure_vec(layout: JaggedLayout) -> Array:
    """[num_blocks, heights..., widths...] as a base-field row, for binding.

    Counts are bounded by M_max = 2^log_m < p for all supported field primes,
    so they round-trip through the field dtype without modular reduction.
    """
    vals = [len(layout.heights), *layout.heights, *layout.widths]
    return jnp.asarray(vals, dtype=layout.dtype)


def _compress_column_claims(column_claims: Array, z_col: Array) -> Array:
    """claim = Σ_c eq(z_col, c)·column_claims[c].

    `column_claims` holds `L = Σ widths` real per-column claims; the eq weights
    span the padded `2^n_c` column hypercube, so take the first `L` weights.
    EF sum via a trace-time fold (never `jnp.sum` over EF — ZKX abort)."""
    weights = expand_eq_to_hypercube(z_col, jnp.ones((), z_col.dtype))  # (2^n_c,)
    L = column_claims.shape[0]
    terms = [weights[c] * column_claims[c] for c in range(L)]
    return functools.reduce(operator.add, terms)
