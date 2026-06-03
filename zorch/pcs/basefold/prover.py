# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold prover — the commit slice of the multilinear PCS on the `pcs` seam.

`commit` is the RS low-degree extension (the native NTT via `ReedSolomon`) of each
column followed by a Merkle commit of the codeword rows. Unlike `kzg`/`fri` — which
commit each polynomial in the batch independently and return one root per poly —
BaseFold is a **matrix commitment**: the columns share one RS domain and the
Merkle leaves are codeword *rows* spanning all columns, so the whole batch binds
under a single root. That single-root commitment is what the jagged structure bind
(`JaggedPcs`) hashes against. Both halves are already-fused substrate ops, so the
commit is one `@jit`-able device zone with no host sync.

`open` is the interleaved-sumcheck BaseFold opening: it evaluates the matrix's K
columns at one shared point, RLC-batches them into a single codeword, then folds
the MLE and the codeword by the same per-round challenge (a sumcheck interleaved
with a natural-order FRI fold) and proves the folded codeword with a query phase.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from zorch.coding.fri import fri_fold
from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree, Opening
from zorch.pcs.basefold.config import BasefoldProof, sample_rlc_coeffs
from zorch.pcs.fri.config import LayerOpening, query_layer_indices, sample_positions
from zorch.poly.multilinear import eval_mle, mle_fold
from zorch.transcript import Transcript


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["digest_layers", "mle", "codeword"],
    meta_fields=["widths"],
)
@dataclass(frozen=True)
class BasefoldProverData:
    """Retained witness from `BasefoldProver.commit`: the Merkle digest layers
    over the RS codeword, the message-domain MLE `[S, K]` (the sumcheck folds
    it), the codeword `[S*blowup, K]` (the FRI folds it and Merkle-opens it),
    plus per-column widths. A pytree so `commit`/`open` ride a `@jit` zone."""

    digest_layers: list[Array]
    mle: Array  # [S, K] message-domain columns
    codeword: Array  # [S*blowup, K] RS codeword (Merkle leaves = its rows)
    # len-1 today (one MLE per commit); reserved for batch-commit in P3.
    widths: tuple[int, ...]


@dataclass(frozen=True)
class BasefoldProver:
    """BaseFold PCS prover (`PcsProver`). `rs` fixes the per-column message length
    (= the MLE height `S`); `tree` commits the codeword rows."""

    rs: ReedSolomon
    tree: MerkleTree
    num_queries: int = 4  # query repetitions; placeholder, not soundness-calibrated

    def commit(self, polys: Sequence[Array]) -> tuple[Array, BasefoldProverData]:
        # The columns share one RS message length S; encode each column separately
        # (lax.fft on extension-field dtypes requires 1-D input, so the batched
        # transpose trick doesn't generalise). O(K) NTTs — fine at current column
        # counts; revisit if K grows. Stack the codewords into [n, K].
        mle = jnp.stack(polys, axis=1)
        codeword = jnp.stack([self.rs.encode(p) for p in polys], axis=1)
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
        n = codeword.shape[0]
        num_vars = z.shape[0]
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

        # 2. Interleaved sumcheck + codeword fold (num_vars rounds).
        uni_msgs, layer_roots, layer_mats, layer_dls = [], [], [], []
        for r in range(num_vars):
            last = z[-(r + 1)]
            rest = z[: -(r + 1)] if r + 1 < num_vars else z[:0]
            # mle_fold(., 0) fixes the bound variable to 0 (the additive fold
            # coincides with the multilinear partial-eval at beta=0), so zero_val
            # is the sumcheck s(0); one_val is recovered from the running claim.
            zero_mle = mle_fold(current_mle, jnp.zeros((), dtype))
            zero_val = eval_mle(zero_mle, rest) if rest.shape[0] > 0 else zero_mle[0]
            one_val = (current_claim - zero_val) / last + zero_val
            uni_msgs.append((zero_val, one_val))
            t = t.observe(jnp.stack([zero_val, one_val]))
            t, beta = t.sample()
            beta = beta.reshape(())
            cw = fri_fold(cw, beta)
            current_mle = mle_fold(current_mle, beta)
            current_claim = zero_val + beta * one_val
            if r < num_vars - 1:
                m = cw.reshape(-1, 1)
                root, dl = self.tree.commit(m)
                layer_mats.append(m)
                layer_dls.append(dl)
                layer_roots.append(root)
                t = t.observe(root)
            else:
                final_layer = cw
                t = t.observe(final_layer)

        # 3. Query phase (natural order, mirrors fri/prover._open_one).
        t, positions = sample_positions(t, n, self.num_queries)
        a = query_layer_indices(positions, n, num_vars)
        half0 = n >> 1

        def open_batch(mtx: Array, dl: list[Array], idx: Array) -> Opening:
            return jax.vmap(lambda i: self.tree.open(mtx, dl, i))(idx)

        comp = LayerOpening(
            open_batch(codeword, prover_data.digest_layers, a[0]),
            open_batch(codeword, prover_data.digest_layers, a[0] + half0),
        )
        layer_opens = []
        for layer in range(1, num_vars):
            half = n >> (layer + 1)
            m, dl = layer_mats[layer - 1], layer_dls[layer - 1]
            layer_opens.append(
                LayerOpening(
                    open_batch(m, dl, a[layer]),
                    open_batch(m, dl, a[layer] + half),
                )
            )
        proof = BasefoldProof(uni_msgs, layer_roots, final_layer, comp, layer_opens)
        return values, proof, t
