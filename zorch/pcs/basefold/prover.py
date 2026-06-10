# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold prover — the commit slice of the multilinear PCS on the `pcs` seam.

`commit` is the low-degree extension of each column (`FoldableCode.encode`; the
native-NTT Reed-Solomon today) followed by a Merkle commit of the codeword rows.
Unlike `kzg`/`fri` — which commit each polynomial in the batch independently and
return one root per poly — BaseFold is a **matrix commitment**: the columns share
one code domain and the Merkle leaves are codeword *rows* spanning all columns, so
the whole batch binds under a single root.

`open` is the BaseFold batch open: it reduces one or more *separately committed*
matrices, evaluated at a shared point, to a single FRI. Each matrix's columns are
combined with the others' by a staggered partial-Lagrange RLC into one codeword
and one MLE; an interleaved sumcheck folds that MLE while the FRI folds the
codeword by the same per-round challenge, the round committing the *pre-fold*
layer's conjugate-pair leaves before sampling the fold challenge. A single matrix
is the degenerate one-round batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import batch_staggered, sample_staggered_coeffs
from zorch.pcs.basefold.config import BasefoldCommitment, BasefoldProof
from zorch.pcs.fold import (
    PairCommittedLayer,
    PreFoldPairCommitRound,
    open_rows,
    sample_positions,
)
from zorch.poly.multilinear import eval_mle, mle_fold
from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver
    from zorch.round import ProverRound


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["digest_layers", "mle", "codeword"],
    meta_fields=["widths"],
)
@dataclass(frozen=True)
class BasefoldProverData:
    """Retained witness from `BasefoldProver.commit`: the Merkle digest layers
    over the codeword, the message-domain MLE `[S, K]` (the sumcheck folds it),
    the codeword `[block_len, K]` (the fold halves it and Merkle opens it), plus
    per-column widths. A pytree so `commit`/`open` ride a `@jit` zone."""

    digest_layers: list[Array]
    mle: Array  # [S, K] message-domain columns
    codeword: Array  # [block_len, K] codeword (Merkle leaves = its rows)
    widths: tuple[int, ...]


def _sumcheck_msg(mle: Array, claim: Array, zs: Array) -> tuple[Array, Array]:
    """The degree-1 sumcheck message `(s(0), s(1))` for the variable bound this
    round (`zs[-1]`), from the running MLE and claim. `mle_fold(., 0)` fixes the
    bound variable to 0 (the additive fold coincides with the multilinear
    partial-eval at beta=0), so zero_val is the sumcheck s(0); one_val is
    recovered from the running claim."""
    zero_mle = mle_fold(mle, jnp.zeros((), zs.dtype))
    rest = zs[:-1]
    zero_val = eval_mle(zero_mle, rest) if rest.shape[0] > 0 else zero_mle[0]
    one_val = (claim - zero_val) / zs[-1] + zero_val
    return zero_val, one_val


# (codeword, running MLE, running claim, unbound z suffix) — the suffix shrinks
# with the MLE, so the round needs no explicit round index.
_OpenCarry = tuple[Array, Array, Array, Array]


@dataclass(frozen=True)
class _SumcheckPairFoldRound(Round):
    """One interleaved-sumcheck round of the batch open: emit + observe the
    degree-1 sumcheck message, run the shared pre-fold pair-commit tail (commit
    the layer's pairs, observe the root, sample β, fold the codeword), then fold
    the MLE and reduce the running claim by the tail's β — sumcheck and codeword
    fold by the same challenge. msg = (sumcheck message, the tail's
    `PairCommittedLayer`)."""

    tail: PreFoldPairCommitRound

    def __call__(
        self, carry: _OpenCarry, transcript: Transcript
    ) -> tuple[_OpenCarry, Transcript, tuple[tuple[Array, Array], PairCommittedLayer]]:
        cw, mle, claim, zs = carry
        zero_val, one_val = _sumcheck_msg(mle, claim, zs)
        t = transcript.observe(jnp.stack([zero_val, one_val]))
        cw, t, layer = self.tail(cw, t)
        mle = mle_fold(mle, layer.beta)
        claim = zero_val + layer.beta * one_val
        return (cw, mle, claim, zs[:-1]), t, ((zero_val, one_val), layer)


@dataclass(frozen=True)
class BasefoldProver:
    """BaseFold PCS prover (`PcsProver`). `code` fixes the per-column message
    length (= the MLE height `S`); `tree` commits the codeword rows."""

    code: FoldableCode
    tree: MerkleTree
    num_queries: int = 4  # query repetitions; placeholder, not soundness-calibrated

    def commit(
        self, polys: Sequence[Array]
    ) -> tuple[BasefoldCommitment, BasefoldProverData]:
        return _commit_body(self.code, self.tree, list(polys))

    def open(
        self,
        prover_data: BasefoldProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, BasefoldProof, Transcript]:
        """Open a single committed matrix — the degenerate one-round batch, the
        `PcsProver` seam shape. Returns `(values, proof, transcript)` with
        `values` the matrix's per-column evaluations `[K]`."""
        values, proof, t = self.open_batch([prover_data], points, transcript)
        return values[0], proof, t

    def open_batch(
        self,
        rounds: Sequence[BasefoldProverData],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[list[Array], BasefoldProof, Transcript]:
        """Batch-open the committed matrices `rounds` at the shared point.

        Returns `(values, proof, transcript)` where `values[r]` is round `r`'s
        per-column evaluations `[w_r]`. A single-element `rounds` is the
        degenerate (un-batched) open.
        """
        if len(points) != 1:
            raise ValueError(
                f"BaseFold opens the matrices at one shared point, got {len(points)}"
            )
        if not rounds:
            raise ValueError("BaseFold opens at least one committed matrix, got none")
        z = points[0]  # (log_S,)
        num_vars = z.shape[0]
        # Eager shape guards, ahead of the jit zone (mirrors the verifier).
        if num_vars < 1:
            raise ValueError("BaseFold opens over at least one variable, got none")
        for pd in rounds:
            if pd.mle.shape[0] != (1 << num_vars):
                raise ValueError(
                    f"point dimension {num_vars} doesn't match MLE height "
                    f"{pd.mle.shape[0]} (expected 2^{num_vars})"
                )
        return _open_batch_body(self, list(rounds), z, transcript)


# Jitted commit body: standalone (outside the jagged seam's enclosing jit),
# an eager commit dispatches the per-column encode ffts and the Merkle
# fused_region op-by-op (#214). Module-level with the static keys compared by
# value, so same-config instances (one per test, in practice) share one trace;
# inside an enclosing jit it traces straight through. Keyed on code + tree
# rather than the prover: commit never reads num_queries, so provers differing
# only there must not compile twice.
@partial(jax.jit, static_argnames=("code", "tree"))
def _commit_body(
    code: FoldableCode, tree: MerkleTree, polys: list[Array]
) -> tuple[BasefoldCommitment, BasefoldProverData]:
    # The columns share one message length S; encode each column separately
    # (encode lowers to lax.fft today, which requires 1-D input on
    # extension-field dtypes, so the batched transpose trick doesn't
    # generalise). O(K) encodes — fine at current column counts; revisit
    # if K grows. Stack the codewords into [n, K].
    mle = jnp.stack(polys, axis=1)
    codeword = jnp.stack([code.encode(p) for p in polys], axis=1)
    root, layers = tree.commit(codeword)
    return root, BasefoldProverData(
        digest_layers=layers, mle=mle, codeword=codeword, widths=(len(polys),)
    )


# Jitted open body: an eager replay re-traces the per-round pair-leaf
# `open_rows` vmaps per call (issue #186). Module-level with the prover as the
# static key — by value (#214), so same-config instances (one per test, in
# practice) share one trace; unlike `commit` — which the jagged seam also
# reaches inside its enclosing jit — `open` is reached eagerly via
# `stacked_open`.
@partial(jax.jit, static_argnames=("prover",))
def _open_batch_body(
    prover: BasefoldProver,
    rounds: Sequence[BasefoldProverData],
    z: Array,
    transcript: Transcript,
) -> tuple[list[Array], BasefoldProof, Transcript]:
    dtype = z.dtype
    num_vars = z.shape[0]
    t = transcript
    # 1. Bind every commitment root into the transcript (the FS commit step,
    #    mirroring `fri`). `verify` observes the same roots in the same order.
    for pd in rounds:
        t = t.observe(pd.digest_layers[-1][0])

    # 2. Per-matrix, per-column evaluations at z; observe each as sampled.
    values = []
    for pd in rounds:
        vals = eval_mle(pd.mle, z, axis=0)  # (w_r,)
        t = t.observe(vals)
        values.append(vals)

    # 3. Staggered partial-Lagrange RLC over the total column width, then
    #    collapse every matrix's columns into one MLE / codeword / claim.
    total_width = sum(int(pd.mle.shape[1]) for pd in rounds)
    t, coeffs = sample_staggered_coeffs(t, total_width, dtype)
    current_mle = batch_staggered([pd.mle for pd in rounds], coeffs)  # (S,)
    cw = batch_staggered([pd.codeword for pd in rounds], coeffs)  # (n,)
    current_claim = batch_staggered(values, coeffs)  # scalar
    # Domain separation: bind the fold-round count (mirrors the reference).
    t = t.observe(jnp.asarray(num_vars, dtype))

    # 4. Interleaved sumcheck + pre-fold pair-leaf FRI fold, num_vars rounds.
    #    Every round commits its pre-fold layer; the final folded codeword
    #    (length `blowup`) is the cleartext final poly.
    carry: _OpenCarry = (cw, current_mle, current_claim, z)
    carry, t, msgs = fold_rounds(
        _SumcheckPairFoldRound(PreFoldPairCommitRound(prover.code, prover.tree)),
        carry,
        t,
        num_vars,
    )
    final_poly = carry[0]
    uni_msgs = [uni for uni, _ in msgs]
    layers = [layer for _, layer in msgs]
    # Bind the cleartext final codeword before sampling queries, so the query
    # positions depend on it (the IOPP terminal binding; `verify` mirrors).
    t = t.observe(final_poly)

    # 5. Query phase: shared positions; open every matrix at the full index
    #    and every fold layer's pair-leaf at its halved index.
    n = prover.code.block_len
    t, positions = sample_positions(t, n, prover.num_queries)
    a = prover.code.layer_positions(positions, num_vars)
    component_openings = [
        open_rows(prover.tree, pd.codeword, pd.digest_layers, positions)
        for pd in rounds
    ]
    query_openings = [
        open_rows(prover.tree, layer.leaves, layer.digest_layers, a[i])
        for i, layer in enumerate(layers)
    ]
    proof = BasefoldProof(
        uni_msgs,
        [layer.root for layer in layers],
        final_poly,
        component_openings,
        query_openings,
    )
    return values, proof, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[
        PcsProver[BasefoldCommitment, BasefoldProverData, BasefoldProof]
    ] = BasefoldProver
    _fold_round: type[ProverRound] = _SumcheckPairFoldRound
