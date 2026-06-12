# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WHIR prover — the `PcsProver` half of the multilinear PCS.

`commit` Reed-Solomon-encodes the polynomial's hypercube evaluations and binds the
codeword rows under a query-strided Merkle root. `open` runs the WHIR rounds: each
round folds the MLE by `k_whir` sumcheck sub-folds, re-encodes the folded MLE as a
fresh (shrinking) RS codeword, commits it, answers one out-of-domain point, and
opens the previous round's codeword at strided query cosets — accumulating the
out-of-domain and in-domain constraints into the weight polynomial. The last round
sends the folded coefficients in the clear.

Unlike fri/basefold this never folds a codeword in place, so it does not use
`pcs/fold.py`; the round driver is WHIR's own. See `config.py` for the divergence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jax import Array, lax
from zk_dtypes import efinfo

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.pcs.fold import sample_positions
from zorch.pcs.whir._math import (
    eq_table,
    pow2_powers,
    query_gamma_powers,
    round_code,
)
from zorch.pcs.whir.config import WhirCommitment, WhirParams, WhirProof
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle, mle_evals_to_coeffs
from zorch.poly.univariate import eval_coeffs
from zorch.sumcheck.prover import fold_pair
from zorch.transcript import GrindingTranscript, Transcript, sample_challenge

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["mle", "codeword", "digest_layers"],
    meta_fields=[],
)
@dataclass(frozen=True)
class WhirProverData:
    """Retained witness from `WhirProver.commit`: the message-domain columns
    `mle` `(S, num_polys)` the open μ-combines then sumcheck-folds, the initial RS
    codeword matrix `(block_len, num_polys)` (its strided rows are the Merkle
    leaves the round-0 queries open), and the codeword's Merkle digest layers. A
    pytree so `commit`/`open` ride a `@jit` zone."""

    mle: Array  # (S, num_polys)
    codeword: Array  # (block_len, num_polys)
    digest_layers: list[Array]


@dataclass(frozen=True)
class WhirProver:
    """WHIR PCS prover (`PcsProver`). `code` is the initial-round RS encoder (the
    round driver re-encodes at shrinking sizes); `tree` commits the codeword's
    `2^k_whir`-row query cosets; `params` carries the per-round knobs."""

    code: ReedSolomon
    tree: StridedMerkleTree
    params: WhirParams

    def commit(self, polys: Sequence[Array]) -> tuple[WhirCommitment, WhirProverData]:
        """Bind a batch of multilinears sharing one point, each given by its
        `2^m` hypercube evaluations. Each column is RS-encoded; the codeword
        matrix binds under one query-strided Merkle root (a matrix commitment,
        like BaseFold). `open` reduces the columns to one polynomial by a μ-power
        random linear combination."""
        if not polys:
            raise ValueError("WHIR commits at least one polynomial, got none")
        for p in polys:
            if p.ndim != 1:
                raise ValueError(f"each polynomial must be 1-D, got ndim {p.ndim}")
            if p.shape[0] != self.code.message_len:
                raise ValueError(
                    f"polynomial length {p.shape[0]} != code message_len "
                    f"{self.code.message_len}"
                )
        return _commit_body(self.code, self.tree, list(polys))

    def open(
        self,
        prover_data: WhirProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, WhirProof, Transcript]:
        """Open the committed batch at the single point `points[0]`, threading
        Fiat-Shamir. Returns `(values, proof, transcript)` with `values` the
        per-column evaluations `f̂ᵢ(z)` `(num_polys,)`."""
        if len(points) != 1:
            raise ValueError(f"WHIR opens at one point, got {len(points)}")
        z = points[0]
        m = z.shape[0]
        k = self.params.k_whir
        num_rounds = len(self.params.num_queries)
        if num_rounds * k != m:
            raise ValueError(
                f"num_rounds·k_whir ({num_rounds}·{k}) must equal num_variables "
                f"({m})"
            )
        return _open_body(self, prover_data, z, transcript)


# Jitted commit body, keyed on code + tree by value (#214): standalone, an eager
# commit dispatches the encode NTT + Merkle fused_region op-by-op. Each column's
# evals become coefficients and RS-encode (one batched NTT over the leading axis);
# the `(block_len, num_polys)` codeword matrix commits under one strided root whose
# `2^k_whir` strided rows one query opens. `mle` (the message-domain columns) and
# the codeword matrix are retained for `open`.
@partial(jax.jit, static_argnames=("code", "tree"))
def _commit_body(
    code: ReedSolomon, tree: StridedMerkleTree, polys: list[Array]
) -> tuple[WhirCommitment, WhirProverData]:
    mle = jnp.stack(polys, axis=1)  # (S, num_polys)
    codeword = code.encode(mle_evals_to_coeffs(mle.T)).T  # (block_len, num_polys)
    root, layers = tree.commit(codeword)
    return root, WhirProverData(mle=mle, codeword=codeword, digest_layers=layers)


# Jitted open body (prover the static key by value): one round driver, not
# `fold_rounds` over a FoldableCode. Each round runs `k_whir` degree-2 sumcheck
# folds of `Σ f̂·ŵ`, re-encodes the folded MLE as a fresh shrinking RS codeword
# (committed via the strided tree, out-of-domain sampled), opens the queried
# codeword at strided cosets, then folds the out-of-domain and in-domain query
# constraints into the weight by γ powers. The last round sends `f̂`'s
# coefficients in the clear.
@partial(jax.jit, static_argnames=("prover",))
def _open_body(
    prover: WhirProver,
    prover_data: WhirProverData,
    z: Array,
    transcript: Transcript,
) -> tuple[Array, WhirProof, Transcript]:
    code, tree, params = prover.code, prover.tree, prover.params
    k = params.k_whir
    num_rounds = len(params.num_queries)
    m = z.shape[0]
    ef, base = z.dtype, code.dtype
    limbs = efinfo(ef).degree
    one = jnp.ones((), ef)

    # Bind the commitment + per-column evaluations, then reduce the batch to one
    # polynomial by a μ-power RLC of the columns (eval_coeffs over the column
    # axis = Σ μⁱ·colᵢ). A single committed column is the degenerate batch.
    values = eval_mle(prover_data.mle, z, axis=0)  # (num_polys,)
    t = transcript.observe(prover_data.digest_layers[-1][0])  # bind initial root
    t = t.observe(values)
    t, mu = sample_challenge(t, ef, limbs)
    f_evals = eval_coeffs(prover_data.mle.astype(ef), mu)  # (S,) combined column
    w_evals = expand_eq_to_hypercube(z, one)

    sumcheck_polys: list[Array] = []
    folding_pow_witnesses: list[Array] = []
    query_pow_witnesses: list[Array] = []
    codeword_roots: list[Array] = []
    ood_values: list[Array] = []
    codeword_openings = []
    initial_opening = None
    final_poly = f_evals  # overwritten in the last round

    # The codeword the current round opens: round 0 the initial commit, later
    # rounds the previous round's re-encode.
    cur_codeword, cur_layers, cur_code = (
        prover_data.codeword,
        prover_data.digest_layers,
        code,
    )

    for r in range(num_rounds):
        is_last = r == num_rounds - 1
        alphas: list[Array] = []
        for _ in range(k):
            f0, f1 = f_evals[0::2], f_evals[1::2]
            w0, w1 = w_evals[0::2], w_evals[1::2]
            s1 = (f1 * w1).sum()
            s2 = ((f1 + f1 - f0) * (w1 + w1 - w0)).sum()
            s = jnp.stack([s1, s2])
            t = t.observe(s)
            sumcheck_polys.append(s)
            t, wit = cast(GrindingTranscript, t).grind(params.folding_pow_bits)
            folding_pow_witnesses.append(wit)
            t, alpha = sample_challenge(t, ef, limbs)
            alphas.append(alpha)
            f_evals = fold_pair(f0, f1, alpha)
            w_evals = fold_pair(w0, w1, alpha)

        g_coeffs = mle_evals_to_coeffs(f_evals)
        # Placeholders kept correctly-typed (no Optional): overwritten when this
        # is not the last round, unread otherwise.
        z0 = one
        next_codeword, next_layers, next_code = cur_codeword, cur_layers, cur_code
        if not is_last:
            next_code = round_code(code, r + 1, k, rate_increase=params.rate_increase)
            next_codeword = lax.bitcast_convert_type(next_code.encode(g_coeffs), base)
            root, next_layers = tree.commit(next_codeword)
            t = t.observe(root)
            codeword_roots.append(root)
            t, z0 = sample_challenge(t, ef, limbs)
            y0 = eval_coeffs(g_coeffs, z0)
            t = t.observe(y0)
            ood_values.append(y0)
        else:
            t = t.observe(g_coeffs)
            final_poly = g_coeffs

        t, qwit = cast(GrindingTranscript, t).grind(params.query_pow_bits)
        query_pow_witnesses.append(qwit)
        stride = cur_code.block_len >> k
        t, positions = sample_positions(t, stride, params.num_queries[r])
        opening = jax.vmap(lambda i: tree.open(cur_codeword, cur_layers, i))(positions)
        if r == 0:
            initial_opening = opening
        else:
            codeword_openings.append(opening)

        t, gamma = sample_challenge(t, ef, limbs)
        if not is_last:
            dim = m - (r + 1) * k
            x_roots = cur_code.domain()[positions].astype(ef)  # (Q,) coset bases

            def _query_weight(x_root: Array) -> Array:
                zi = pow2_powers(x_root, k + 1)[-1]  # x_root^(2^k), folded-domain pt
                return eq_table(pow2_powers(zi, dim))

            # γ folds OOD then each query into the weight; queries are independent,
            # so one vmap + a γ-power-weighted reduction, not a Python loop.
            gpows = query_gamma_powers(gamma, params.num_queries[r])
            query_tables = jax.vmap(_query_weight)(x_roots)  # (Q, 2^dim)
            w_evals = (
                w_evals
                + gamma * eq_table(pow2_powers(z0, dim))
                + (gpows[:, None] * query_tables).sum(0)
            )
            cur_codeword, cur_layers, cur_code = next_codeword, next_layers, next_code

    assert initial_opening is not None
    proof = WhirProof(
        sumcheck_polys=sumcheck_polys,
        codeword_roots=codeword_roots,
        ood_values=ood_values,
        folding_pow_witnesses=folding_pow_witnesses,
        query_pow_witnesses=query_pow_witnesses,
        initial_opening=initial_opening,
        codeword_openings=codeword_openings,
        final_poly=final_poly,
    )
    return values, proof, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[PcsProver[WhirCommitment, WhirProverData, WhirProof]] = WhirProver
