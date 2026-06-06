# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold prover — the commit slice of the multilinear PCS on the `pcs` seam.

`commit` is the low-degree extension of each column (`FoldableCode.encode`; the
native-NTT Reed-Solomon today) followed by a Merkle commit of the codeword rows.
Unlike `kzg`/`fri` — which commit each polynomial in the batch independently and
return one root per poly — BaseFold is a **matrix commitment**: the columns share
one code domain and the
Merkle leaves are codeword *rows* spanning all columns, so the whole batch binds
under a single root. That single-root commitment is what the jagged structure bind
(`JaggedPcsProver`) hashes against. Both halves are already-fused substrate ops, so the
commit is one `@jit`-able device zone with no host sync.

`open` is the interleaved-sumcheck BaseFold opening: it evaluates the matrix's K
columns at one shared point, RLC-batches them into a single codeword, then folds
the MLE and the codeword by the same per-round challenge (a sumcheck interleaved
with the code's fold) and proves the folded codeword with a query phase.
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
from zorch.pcs.basefold.config import (
    BasefoldCommitment,
    BasefoldProof,
    sample_rlc_coeffs,
)
from zorch.pcs.fold import CommitFoldRound, CommittedLayer, open_query_phase
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
    over the codeword, the message-domain MLE `[S, K]` (the sumcheck folds
    it), the codeword `[block_len, K]` (the fold halves it and Merkle opens it),
    plus per-column widths. A pytree so `commit`/`open` ride a `@jit` zone."""

    digest_layers: list[Array]
    mle: Array  # [S, K] message-domain columns
    codeword: Array  # [block_len, K] codeword (Merkle leaves = its rows)
    # len-1 today (one MLE per commit); reserved for batch-commit in P3.
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
class _SumcheckFoldRound(Round):
    """One interleaved-sumcheck round: emit + observe the degree-1 sumcheck
    message, run the shared commit-and-fold tail, then fold the MLE and reduce
    the running claim by the tail's β — the sumcheck and the codeword fold by
    the same challenge. msg = (sumcheck message, the tail's `CommittedLayer`)."""

    tail: CommitFoldRound

    def __call__(
        self, carry: _OpenCarry, transcript: Transcript
    ) -> tuple[_OpenCarry, Transcript, tuple[tuple[Array, Array], CommittedLayer]]:
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
        # The columns share one message length S; encode each column separately
        # (encode lowers to lax.fft today, which requires 1-D input on
        # extension-field dtypes, so the batched transpose trick doesn't
        # generalise). O(K) encodes — fine at current column counts; revisit
        # if K grows. Stack the codewords into [n, K].
        mle = jnp.stack(polys, axis=1)
        codeword = jnp.stack([self.code.encode(p) for p in polys], axis=1)
        root, layers = self.tree.commit(codeword)
        return root, BasefoldProverData(
            digest_layers=layers, mle=mle, codeword=codeword, widths=(len(polys),)
        )

    def open(
        self,
        prover_data: BasefoldProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, BasefoldProof, Transcript]:
        if len(points) != 1:
            raise ValueError(
                f"BaseFold opens the matrix at one shared point, got {len(points)}"
            )
        z = points[0]  # (log_S,)
        mle, codeword = prover_data.mle, prover_data.codeword  # [S,K], [n,K]
        dtype = z.dtype
        K = mle.shape[1]
        num_vars = z.shape[0]
        if num_vars < 1:
            raise ValueError("BaseFold opens over at least one variable, got none")
        if mle.shape[0] != (1 << num_vars):
            raise ValueError(
                f"point dimension {num_vars} doesn't match MLE height "
                f"{mle.shape[0]} (expected 2^{num_vars})"
            )
        t = transcript
        # Bind the matrix commitment root into the transcript so every fold/query
        # challenge depends on it (the FS commit step, mirroring `fri`). `verify`
        # observes the same root.
        t = t.observe(prover_data.digest_layers[-1][0])

        # 1. Per-column evals (one eq-expansion, batched over the K columns), then
        #    RLC-batch the K columns into one MLE/codeword.
        values = eval_mle(mle, z, axis=0)  # (K,)
        t = t.observe(values)
        t, coeffs = sample_rlc_coeffs(t, K, dtype)
        current_mle = (mle * coeffs).sum(axis=1)  # (S,)
        cw = (codeword * coeffs).sum(axis=1)  # (n,)
        current_claim = (values * coeffs).sum()

        # 2. Interleaved sumcheck + codeword fold (num_vars rounds): the first
        #    num_vars−1 ride the shared commit-and-fold round; the final round
        #    is peeled — its folded layer goes in the clear, no commit.
        carry: _OpenCarry = (cw, current_mle, current_claim, z)
        carry, t, msgs = fold_rounds(
            _SumcheckFoldRound(CommitFoldRound(self.code, self.tree)),
            carry,
            t,
            num_vars - 1,
        )
        cw, current_mle, current_claim, zs = carry
        zero_val, one_val = _sumcheck_msg(current_mle, current_claim, zs)
        t = t.observe(jnp.stack([zero_val, one_val]))
        t, beta = t.sample()
        final_layer = self.code.fold(cw, beta.reshape(()))
        t = t.observe(final_layer)
        uni_msgs = [uni for uni, _ in msgs] + [(zero_val, one_val)]
        layers = [layer for _, layer in msgs]

        # 3. Query phase (natural order, shared with fri).
        t, comp, layer_opens = open_query_phase(
            self.tree, t, codeword, prover_data.digest_layers, layers, self.num_queries
        )
        proof = BasefoldProof(
            uni_msgs, [layer.root for layer in layers], final_layer, comp, layer_opens
        )
        return values, proof, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[
        PcsProver[BasefoldCommitment, BasefoldProverData, BasefoldProof]
    ] = BasefoldProver
    _fold_round: type[ProverRound] = _SumcheckFoldRound
