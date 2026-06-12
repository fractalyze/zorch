# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WHIR verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` replays the WHIR rounds from the transcript: it re-derives the folding,
out-of-domain and query challenges in the prover's order, checks each round's
degree-2 sumcheck message against the running claim, rebuilds the strided query
openings from the committed roots, folds each opened `2^k_whir` coset as a small
MLE at the round's folding challenges (the binary k-fold consistency), and closes
on the final-poly constraint. It holds only the public params (`code` for the RS
geometry, `tree` for the Merkle config) — never the prover's retained codeword.
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
    binary_k_fold,
    interp_quadratic_012,
    pow2_powers,
    query_gamma_powers,
    round_code,
)
from zorch.pcs.whir.config import WhirCommitment, WhirParams, WhirProof
from zorch.pcs.whir.scheme import EqWhirScheme, WhirScheme
from zorch.poly.eq import eval_eq
from zorch.poly.univariate import eval_coeffs
from zorch.transcript import GrindingTranscript, Transcript, sample_challenge

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


@dataclass(frozen=True)
class WhirVerifier:
    """WHIR PCS verifier (`PcsVerifier`). `code`/`tree`/`params`/`scheme` must
    match the prover's."""

    code: ReedSolomon
    tree: StridedMerkleTree
    params: WhirParams
    scheme: WhirScheme = EqWhirScheme()

    def verify(
        self,
        commitment: WhirCommitment,
        points: Sequence[Array],
        values: Array,
        proof: WhirProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Return `(ok, transcript)` where `ok` is a scalar boolean array."""
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
        # Fail loud on a structurally malformed proof — a short list would let the
        # round loop silently skip checks (mirrors BasefoldVerifier.verify_batch).
        lengths = {
            "sumcheck_polys": (len(proof.sumcheck_polys), num_rounds * k),
            "folding_pow_witnesses": (len(proof.folding_pow_witnesses), num_rounds * k),
            "query_pow_witnesses": (len(proof.query_pow_witnesses), num_rounds),
            "codeword_roots": (len(proof.codeword_roots), num_rounds - 1),
            "ood_values": (len(proof.ood_values), num_rounds - 1),
            "codeword_openings": (len(proof.codeword_openings), num_rounds - 1),
        }
        bad = {
            name: got_exp
            for name, got_exp in lengths.items()
            if got_exp[0] != got_exp[1]
        }
        if bad:
            raise ValueError(f"malformed WHIR proof: {bad} (got, expected)")
        num_polys = proof.initial_opening.row.shape[-1]
        if values.ndim != 1 or values.shape[0] != num_polys:
            raise ValueError(
                f"values must be 1-D of length num_polys ({num_polys}), got shape "
                f"{values.shape}"
            )
        return _verify_body(self, commitment, z, values, proof, transcript)


@partial(jax.jit, static_argnames=("verifier",))
def _verify_body(
    verifier: WhirVerifier,
    commitment: WhirCommitment,
    z: Array,
    values: Array,
    proof: WhirProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    code, tree, params = verifier.code, verifier.tree, verifier.params
    k = params.k_whir
    num_rounds = len(params.num_queries)
    m = z.shape[0]
    ef = z.dtype
    limbs = efinfo(ef).degree
    one = jnp.ones((), ef)

    # Mirror the prover: bind commitment + per-column values, sample μ, and take
    # the running claim as the μ-power combine of the columns' claimed evals.
    t = transcript.observe(commitment)
    t = t.observe(values)
    t, ok = cast(GrindingTranscript, t).check_witness(
        params.mu_pow_bits, proof.mu_pow_witness
    )
    t, mu = sample_challenge(t, ef, limbs)
    claim = eval_coeffs(values, mu)

    all_alphas: list[Array] = []
    z0s: list[Array] = []
    gammas: list[Array] = []
    query_roots: list[Array] = []  # x_root per query, per round (folded-domain base)
    sc_idx = 0
    cur_root = commitment
    cur_code = code

    for r in range(num_rounds):
        is_last = r == num_rounds - 1
        alphas: list[Array] = []
        for _ in range(k):
            s = proof.sumcheck_polys[sc_idx]
            t = t.observe(s)
            t, okw = cast(GrindingTranscript, t).check_witness(
                params.folding_pow_bits, proof.folding_pow_witnesses[sc_idx]
            )
            ok = ok & okw
            t, alpha = sample_challenge(t, ef, limbs)
            alphas.append(alpha)
            # s(0) = claim − s(1); reduce the claim to s(α).
            claim = interp_quadratic_012(claim - s[0], s[0], s[1], alpha)
            sc_idx += 1
        all_alphas.extend(alphas)

        if not is_last:
            t = t.observe(proof.codeword_roots[r])
            t, z0 = sample_challenge(t, ef, limbs)
            z0s.append(z0)
            y0 = proof.ood_values[r]
            t = t.observe(y0)
        else:
            t = t.observe(proof.final_poly)

        t, okw = cast(GrindingTranscript, t).check_witness(
            params.query_pow_bits, proof.query_pow_witnesses[r]
        )
        ok = ok & okw
        stride = cur_code.block_len >> k
        t, positions = sample_positions(t, stride, params.num_queries[r])

        opening = proof.initial_opening if r == 0 else proof.codeword_openings[r - 1]
        rebuilt = jax.vmap(tree.reconstruct_root)(positions, opening)
        ok = ok & jnp.all(rebuilt == cur_root)

        # Fold each opened coset to the round-folded poly's value at the query
        # point; coset points {x·ω_k^j} are the queried domain gathered at the
        # strided coset indices (conjugates a half apart).
        domain = cur_code.domain()
        coset_idx = positions[:, None] + stride * jnp.arange(1 << k)
        coset_pts = domain[coset_idx]  # (Q, 2^k)
        # Round 0 opens the committed matrix — μ-combine its columns (mirroring
        # the prover's batch combine); later rounds open the single EF re-encode
        # stored as base-field limbs — bitcast back.
        if r == 0:
            coset_vals = eval_coeffs(opening.row.astype(ef), mu)  # (Q, 2^k)
        else:
            coset_vals = lax.bitcast_convert_type(opening.row, ef)  # (Q, 2^k)
        ys = jax.vmap(lambda v, p: binary_k_fold(v, alphas, p))(coset_vals, coset_pts)
        query_roots.append(domain[positions].astype(ef))

        t, gamma = sample_challenge(t, ef, limbs)
        gammas.append(gamma)
        # γ folds OOD then each query into the claim; queries are independent, so
        # a γ-power-weighted reduction, not a Python loop.
        gpows = query_gamma_powers(gamma, params.num_queries[r])
        if not is_last:
            claim = claim + proof.ood_values[r] * gamma
        claim = claim + (ys * gpows).sum()

        if not is_last:
            cur_root = proof.codeword_roots[r]
            cur_code = round_code(code, r + 1, k, rate_increase=params.rate_increase)

    # Final constraint: the running claim equals the original opening term plus
    # every round's γ-weighted out-of-domain and in-domain consistency terms,
    # each tying the final polynomial to a point the fold reduced to.
    final_poly = proof.final_poly
    prefix = verifier.scheme.final_prefix(z, jnp.stack(all_alphas))
    acc = prefix * final_poly[0]
    j = k
    for r in range(num_rounds):
        gamma = gammas[r]
        alpha_slc = all_alphas[j:m]
        rem = m - j  # remaining variables after this round's folds

        def _consistency(point: Array) -> Array:
            """The constraint term for a point reduced to round `r`: eq of the
            remaining folds with the point's powers-of-two, times the final poly
            at the point folded through the rest."""
            pp = pow2_powers(point, rem + 1)
            eq = eval_eq(jnp.stack(alpha_slc), jnp.stack(pp[:-1])) if rem else one
            return eq * eval_coeffs(final_poly, pp[-1])

        if r != num_rounds - 1:
            acc = acc + gamma * _consistency(z0s[r])
        # Closes once at the end over the static query count, so a straight-line
        # `for` (conventions.md "Loops"): the per-query `eval_coeffs` contracts a
        # fixed `final_poly` against a per-point power vector, which doesn't `vmap`
        # cleanly (a one-batched-operand dot).
        gpows = query_gamma_powers(gamma, params.num_queries[r])
        for qi in range(params.num_queries[r]):
            zi = pow2_powers(query_roots[r][qi], k + 1)[-1]
            acc = acc + gpows[qi] * _consistency(zi)
        j += k

    ok = ok & (acc == claim)
    return ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[WhirCommitment, WhirProof]] = WhirVerifier
