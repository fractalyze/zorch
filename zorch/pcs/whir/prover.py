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
from zorch.commit.merkle import Opening
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.pcs.whir._math import (
    eq_table,
    pow2_powers,
    query_gamma_powers,
    round_code,
    sample_query_positions,
)
from zorch.pcs.whir.config import WhirCommitment, WhirParams, WhirProof
from zorch.pcs.whir.scheme import EqWhirScheme, WhirScheme
from zorch.poly.multilinear import mle_evals_to_coeffs
from zorch.poly.univariate import eval_coeffs
from zorch.sumcheck.domain import fold
from zorch.transcript import GrindingTranscript, Transcript, sample_challenge
from zorch.utils.bits import log2_strict_usize

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
    `2^k_whir`-row query cosets; `params` carries the per-round knobs; `scheme`
    supplies the initial message + weight maps (default: a plain multilinear
    opening — see `scheme.py`)."""

    code: ReedSolomon
    tree: StridedMerkleTree
    params: WhirParams
    scheme: WhirScheme = EqWhirScheme()

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
        """Open a single committed matrix at `points[0]` — the degenerate
        one-commitment μ-batch, the `PcsProver` seam shape. Returns
        `(values, proof, transcript)` with `values` the matrix's per-column
        evaluations `f̂ᵢ(z)` `(W,)`."""
        return self.open_batch([prover_data], points, transcript)

    def open_batch(
        self,
        prover_datas: Sequence[WhirProverData],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, WhirProof, Transcript]:
        """μ-batch-open the committed matrices at the shared point `points[0]`,
        threading Fiat-Shamir.

        The committed columns of every commitment join into one `(S, ΣWᵢ)`
        message — first commitment's columns first — over which the scheme's
        μ-power combine runs; each commitment keeps its own codeword tree and
        round 0 opens them all. Returns `(values, proof, transcript)` with
        `values` the per-column evaluations across all commitments `(ΣWᵢ,)`. A
        single-element `prover_datas` is the degenerate (un-batched) open."""
        datas = list(prover_datas)
        if not datas:
            raise ValueError("WHIR opens at least one commitment, got none")
        if len(points) != 1:
            raise ValueError(f"WHIR opens at one point, got {len(points)}")
        z = points[0]
        m = z.shape[0]
        k = self.params.k_whir
        num_rounds = len(self.params.num_queries)
        if not (0 < num_rounds * k <= m):
            raise ValueError(
                f"num_rounds·k_whir ({num_rounds}·{k}) must fold between 1 and "
                f"num_variables ({m}) inclusive"
            )
        return _open_body(self, datas, z, transcript)


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


# The open body is an EAGER driver over jitted compute islands, not one `@jit`
# zone. WHIR grinds proof-of-work between every sumcheck fold and before every
# query phase, and `Transcript.grind` (unlike `check_witness`) validates its
# witness on the host — so it cannot be traced. Wrapping the whole driver in
# `@jit` therefore breaks at any `pow_bits > 0` (the production case). Instead the
# pure-array work fuses inside per-phase islands (each its own kernel; the
# transcript's `observe`/`sample` are already jitted internally), the grinds run
# eagerly on the host between them, and a Python loop threads the rounds. This
# mirrors the structure the openvm reference prover uses to stay byte-exact under
# real grinds.


@partial(jax.jit, static_argnames=("prover",))
def _island_initial(prover: WhirProver, mle: Array, z: Array) -> Array:
    """The per-column claimed evaluations the proof opens to."""
    return prover.scheme.claimed_values(mle, z)


@partial(jax.jit, static_argnames=("prover",))
def _island_tables(
    prover: WhirProver, mle: Array, z: Array, mu: Array
) -> tuple[Array, Array]:
    """The initial sumcheck message `f̂` (μ-combined columns) and weight `ŵ`."""
    return prover.scheme.combined_f_evals(mle, mu), prover.scheme.initial_weight(z)


@jax.jit
def _island_round_poly(f_evals: Array, w_evals: Array) -> Array:
    """The degree-2 sumcheck message `[s(1), s(2)]` of `Σ f̂·ŵ`."""
    f0, f1 = f_evals[0::2], f_evals[1::2]
    w0, w1 = w_evals[0::2], w_evals[1::2]
    s1 = (f1 * w1).sum()
    s2 = ((f1 + f1 - f0) * (w1 + w1 - w0)).sum()
    return jnp.stack([s1, s2])


@jax.jit
def _island_fold(f_evals: Array, w_evals: Array, alpha: Array) -> tuple[Array, Array]:
    """Fold both tables at `alpha` (LSB-first, halving the length)."""
    return (
        fold(f_evals, alpha, msb=False),
        fold(w_evals, alpha, msb=False),
    )


@jax.jit
def _island_coeffs(f_evals: Array) -> Array:
    """The folded MLE's coefficients `ĝ` (re-encoded, or sent in the clear)."""
    return mle_evals_to_coeffs(f_evals)


@partial(jax.jit, static_argnames=("prover", "r"))
def _island_reencode(
    prover: WhirProver, g_coeffs: Array, r: int
) -> tuple[Array, Array, list[Array]]:
    """RS-encode `ĝ` as round `r`'s fresh codeword and strided-Merkle-commit it."""
    code = prover.code
    next_code = round_code(
        code, r + 1, prover.params.k_whir, rate_increase=prover.params.rate_increase
    )
    codeword = lax.bitcast_convert_type(next_code.encode(g_coeffs), code.dtype)
    root, layers = prover.tree.commit(codeword)
    return root, codeword, layers


@jax.jit
def _island_ood(g_coeffs: Array, z0: Array) -> Array:
    """`ĝ` at the out-of-domain point (the coefficients as a univariate at `z0`)."""
    return eval_coeffs(g_coeffs, z0)


@partial(jax.jit, static_argnames=("prover",))
def _island_query_open(
    prover: WhirProver, codeword: Array, layers: list[Array], positions: Array
) -> Opening:
    """Open the queried codeword at every strided query coset."""
    return jax.vmap(lambda i: prover.tree.open(codeword, layers, i))(positions)


@partial(jax.jit, static_argnames=("prover", "r"))
def _island_weight_update(
    prover: WhirProver,
    w_evals: Array,
    gamma: Array,
    z0: Array,
    positions: Array,
    r: int,
) -> Array:
    """Fold round `r`'s out-of-domain and in-domain query constraints into the
    weight by γ powers (the OOD term takes `γ¹`, queries `γ²…`)."""
    code, params = prover.code, prover.params
    k = params.k_whir
    # Variables left in the weight after this round's folds. Derived from the
    # table rather than `num_rounds·k` so a final residual (num_rounds·k < m,
    # the rate-increasing case) sizes the OOD/query eq tables correctly; equals
    # `m − (r+1)·k`, the former hardcoded value, when the fold is full.
    dim = log2_strict_usize(w_evals.shape[0])
    cur_code = round_code(code, r, k, rate_increase=params.rate_increase)
    x_roots = cur_code.domain()[positions].astype(z0.dtype)  # (Q,) coset bases

    def _query_weight(x_root: Array) -> Array:
        zi = pow2_powers(x_root, k + 1)[-1]  # x_root^(2^k), folded-domain point
        return eq_table(pow2_powers(zi, dim))

    # Queries are independent, so one vmap + a γ-power-weighted reduction.
    gpows = query_gamma_powers(gamma, params.num_queries[r])
    query_tables = jax.vmap(_query_weight)(x_roots)  # (Q, 2^dim)
    return (
        w_evals
        + gamma * eq_table(pow2_powers(z0, dim))
        + (gpows[:, None] * query_tables).sum(0)
    )


def _open_body(
    prover: WhirProver,
    prover_datas: list[WhirProverData],
    z: Array,
    transcript: Transcript,
) -> tuple[Array, WhirProof, Transcript]:
    code, params = prover.code, prover.params
    k = params.k_whir
    num_rounds = len(params.num_queries)
    ef = z.dtype
    limbs = efinfo(ef).degree

    # Multi-commitment μ-batch: the committed columns of every commitment join
    # into one `(S, ΣWᵢ)` message — first commitment's columns first — over which
    # the scheme's μ-power combine runs (a single commitment is the length-1
    # case, byte-identical). Each commitment keeps its own codeword tree; round 0
    # opens them all (below).
    mle = jnp.concatenate([pd.mle for pd in prover_datas], axis=1)

    # Bind the commitment + per-column evaluations; the scheme supplies the claimed
    # values, the μ-combined initial message, and the weight (plain MLE + eq by
    # default; prismalinear + möbius for SWIRL).
    values = _island_initial(prover, mle, z)  # (ΣWᵢ,)
    # Bind the commitment + claimed values (the scheme decides how — the default
    # absorbs both; a consumer whose outer protocol already bound the commitment
    # no-ops this to stay byte-exact).
    t = prover.scheme.bind(transcript, prover_datas[0].digest_layers[-1][0], values)
    t, mu_wit = cast(GrindingTranscript, t).grind(params.mu_pow_bits)
    t, mu = sample_challenge(t, ef, limbs)
    f_evals, w_evals = _island_tables(prover, mle, z, mu)

    sumcheck_polys: list[Array] = []
    folding_pow_witnesses: list[Array] = []
    query_pow_witnesses: list[Array] = []
    codeword_roots: list[Array] = []
    ood_values: list[Array] = []
    codeword_openings = []
    initial_openings: list[Opening] = []
    final_poly = f_evals  # overwritten in the last round

    # The codeword later rounds (r ≥ 1) open: the previous round's single
    # re-encode, set at the end of each round. Round 0 opens the committed
    # codewords directly from `prover_datas` (below), so this initial value is
    # just a placeholder until the first re-encode.
    cur_codeword, cur_layers = prover_datas[0].codeword, prover_datas[0].digest_layers

    for r in range(num_rounds):
        is_last = r == num_rounds - 1
        for _ in range(k):
            s = _island_round_poly(f_evals, w_evals)
            t = t.observe(s)
            sumcheck_polys.append(s)
            t, wit = cast(GrindingTranscript, t).grind(params.folding_pow_bits)
            folding_pow_witnesses.append(wit)
            t, alpha = sample_challenge(t, ef, limbs)
            f_evals, w_evals = _island_fold(f_evals, w_evals, alpha)

        g_coeffs = _island_coeffs(f_evals)
        z0 = jnp.ones((), ef)  # placeholder; set + read only when not the last round
        next_codeword, next_layers = cur_codeword, cur_layers
        if not is_last:
            root, next_codeword, next_layers = _island_reencode(prover, g_coeffs, r)
            t = t.observe(root)
            codeword_roots.append(root)
            t, z0 = sample_challenge(t, ef, limbs)
            y0 = _island_ood(g_coeffs, z0)
            t = t.observe(y0)
            ood_values.append(y0)
        else:
            t = t.observe(g_coeffs)
            final_poly = g_coeffs

        t, qwit = cast(GrindingTranscript, t).grind(params.query_pow_bits)
        query_pow_witnesses.append(qwit)
        stride = (
            round_code(code, r, k, rate_increase=params.rate_increase).block_len >> k
        )
        t, positions = sample_query_positions(
            t, stride, params.num_queries[r], code.dtype
        )
        if r == 0:
            # Open every commitment's codeword at the shared query cosets.
            initial_openings = [
                _island_query_open(prover, pd.codeword, pd.digest_layers, positions)
                for pd in prover_datas
            ]
        else:
            codeword_openings.append(
                _island_query_open(prover, cur_codeword, cur_layers, positions)
            )

        t, gamma = sample_challenge(t, ef, limbs)
        if not is_last:
            w_evals = _island_weight_update(prover, w_evals, gamma, z0, positions, r)
            cur_codeword, cur_layers = next_codeword, next_layers

    assert initial_openings
    proof = WhirProof(
        mu_pow_witness=mu_wit,
        sumcheck_polys=sumcheck_polys,
        codeword_roots=codeword_roots,
        ood_values=ood_values,
        folding_pow_witnesses=folding_pow_witnesses,
        query_pow_witnesses=query_pow_witnesses,
        initial_openings=initial_openings,
        codeword_openings=codeword_openings,
        final_poly=final_poly,
    )
    return values, proof, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _pcs_prover: type[PcsProver[WhirCommitment, WhirProverData, WhirProof]] = WhirProver
