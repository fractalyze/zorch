# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""IPA prover: commit and open by log-n basis folding.

`commit` is the Pedersen MSM `P = ⟨a, G⟩ = msm(coeffs, basis)`. `open` proves
`p(x) = ⟨a, b⟩` for `b = (1, x, …, x^{n-1})` by the Bulletproofs/Halo fold. The
opening first squeezes a seed challenge ξ₀ binding `(P, x, v)` and scales the
inner-product generator to `h' = U·ξ₀` (arkworks `ipa_pc`'s `h_prime`); each of
the `k = log₂ n` rounds then sends two cross-term group elements

    L_j = ⟨a_hi, G_lo⟩ + ⟨a_hi, b_lo⟩·h'
    R_j = ⟨a_lo, G_hi⟩ + ⟨a_lo, b_hi⟩·h'

absorbs them into the Fiat-Shamir transcript, samples a challenge `u_j`, and folds
all three vectors in half

    a ← a_lo + a_hi·u_j⁻¹
    b ← b_lo + b_hi·u_j
    G ← G_lo + G_hi·u_j

The L/R labeling (`L` pairs `a_hi` with `G_lo`) and the no-inverse fold (the low
half carried unscaled, the high half scaled by `u_j`) are arkworks `ipa_pc`'s
convention — the same the check polynomial `h(X) = ∏(1 + u_j·X^…)` and the
decider's final-key MSM are written against (see `math.py` and zorch#339).
Folding continues until each vector collapses to a single element. Each cross
term is one `lax.msm` (the `h'` term folded in as one extra (scalar, point) pair),
so the only raw EC arithmetic is the basis fold `G_lo + G_hi·u` — vectorized
scalar-mul and point-add, with the result converted back to affine each round to
keep the point representation (and thus the next round's `lax.msm` input) stable.
The fold is a `lax.scan` over the round count, not a Python unroll: the unroll's
static-slice fold fuses cleanly but recompiles per size (compile grows linearly in
`k = log₂ n`), while the scan compiles in O(1). The scan's fixed-shape carry keeps
a/b/G at full size n and reads the collapsing half with a masked `dynamic_slice`,
byte-identical to the shrinking fold (`0·P = identity`) — trading the unroll's
static-slice fusion for the `dynamic_slice`/scatter fusion boundaries the scan
needs (a compile-time-for-fusion trade, not a claim of one fused kernel). Warm
runtime is unchanged at tested sizes (FS-permute-bound); the `valid_count` msm
operand removes the resulting k·n mask padding (see `_open_one`, zorch#344).

Scope: one base-field polynomial per opening, power-of-two length. `open` is
transparent (no blinding); the hiding/zk `_open_one_zk` below blinds the witness
and opens an `s`-randomized commitment. A demonstration of the seam, not a
hardened prover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import jax.numpy as jnp
from jax import Array, lax

from zorch.pcs.ipa.challenger import (
    IpaChallenger,
    TranscriptChallenger,
    ZkIpaChallenger,
)
from zorch.pcs.ipa.config import IpaCommitment, IpaProof, IpaZkProof
from zorch.pcs.ipa.setup import IpaKey
from zorch.poly.univariate import powers
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver

_Ch = TypeVar("_Ch", bound=IpaChallenger)
_ZkCh = TypeVar("_ZkCh", bound=ZkIpaChallenger)


def _hiding_commit(v: Array, r: Array, basis: Array, s: Array) -> Array:
    """Hiding Pedersen commit `⟨v, G⟩ + r·s` — the blinding generator `s` rides the
    MSM as one extra `(r, s)` pair, so the commitment (or the blinding polynomial's)
    randomness is folded in with no separate point-add."""
    return lax.msm(
        jnp.concatenate([v, r[None]]),
        jnp.concatenate([basis[: v.shape[0]], s[None]]),
    )


@dataclass(frozen=True)
class IpaProverData:
    """Retained witness from `IpaProver.commit`: the coefficient vectors (to drive
    the fold in `open`) and the commitments (the opening's Fiat-Shamir binds them
    as part of the statement). Holds references to the (immutable) inputs — no
    polynomial data is copied."""

    coeffs: tuple[Array, ...]
    commitments: Array  # G1 affine [K] — P_j per poly, bound into the FS seed


@dataclass(frozen=True)
class IpaProver:
    key: IpaKey

    def commit(self, polys: Sequence[Array]) -> tuple[IpaCommitment, IpaProverData]:
        """Pedersen-commit a batch of coefficient vectors: `P_j = ⟨a_j, G⟩`.
        Returns the stacked G1 commitments and the prover data (coeffs plus the
        commitments, which the opening's Fiat-Shamir binds)."""
        commitments = jnp.stack(
            [lax.msm(c, self.key.basis[: c.shape[0]]) for c in polys]
        )
        return commitments, IpaProverData(tuple(polys), commitments)

    def commit_zk(
        self, polys: Sequence[Array], randomnesses: Sequence[Array]
    ) -> tuple[IpaCommitment, IpaProverData]:
        """Hiding Pedersen commit: `P_j = ⟨a_j, G⟩ + r_j·s`. The blinding `r_j·s`
        makes the commitment hiding; `_open_one_zk` opens such a commitment to a
        zero-knowledge proof, removing the blinding inside the fold (the
        `− combined_rand·s` term, which is why the commitment must carry `r_j·s` to
        begin with). Requires the key's blinding generator `key.s`."""
        s = self.key.s
        if s is None:
            raise ValueError("zk commit requires the blinding generator key.s")
        if len(polys) != len(randomnesses):
            raise ValueError(
                f"batch mismatch: {len(polys)} polys vs "
                f"{len(randomnesses)} randomnesses"
            )
        commitments = jnp.stack(
            [
                _hiding_commit(c, r, self.key.basis, s)
                for c, r in zip(polys, randomnesses)
            ]
        )
        return commitments, IpaProverData(tuple(polys), commitments)

    def open(
        self,
        prover_data: IpaProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, list[IpaProof], Transcript]:
        """Open poly `j` at `points[j]`. Returns `(values, proofs, transcript)`
        with `values[j] = p_j(points[j])` and one `IpaProof` per opening. Wraps the
        transcript in the default `TranscriptChallenger` (the zorch-native FS) and
        threads it through every opening; a byte-exact consumer drives `_open_one`
        with its own `IpaChallenger` instead."""
        if len(prover_data.coeffs) != len(points):
            raise ValueError(
                f"batch mismatch: {len(prover_data.coeffs)} polys vs "
                f"{len(points)} points"
            )
        fs = TranscriptChallenger(transcript, prover_data.coeffs[0].dtype)
        values, proofs = [], []
        for commitment, coeffs, x in zip(
            prover_data.commitments, prover_data.coeffs, points
        ):
            fs, value, proof = _open_one(self.key, commitment, coeffs, x, fs)
            values.append(value)
            proofs.append(proof)
        return jnp.stack(values), proofs, fs.transcript


def _open_one(
    key: IpaKey, commitment: Array, coeffs: Array, x: Array, fs: _Ch
) -> tuple[_Ch, Array, IpaProof]:
    """Fold one (poly, point) to a proof, deriving challenges through the injected
    `fs`. Returns `(challenger, value, proof)`. Challenger-generic so an
    accumulation consumer can drive it with an arkworks-faithful `IpaChallenger`
    — which, like `TranscriptChallenger`, must be a JAX pytree (the fold carries it
    through the `lax.scan` below)."""
    n = coeffs.shape[0]
    k = log2_strict_usize(n)
    hn = n // 2
    affine = key.basis.dtype  # the point representation msm consumes
    one = jnp.ones((), dtype=coeffs.dtype)
    fq0 = jnp.zeros((), dtype=coeffs.dtype)  # scalar fill for the masked halves
    idx = jnp.arange(hn)  # active-half mask base (loop-invariant, hoisted)

    a = coeffs
    b = powers(x, n)
    g = key.basis[:n]
    value = jnp.sum(a * b)  # ⟨a, b⟩ = p(x)

    # Seed ξ₀ binds the statement (P, x, v) and scales the inner-product generator
    # to h' = U·ξ₀ (arkworks h_prime). The cross-term inner products absorb the ξ₀
    # factor (one extra (scalar, point) pair `(⟨·,·⟩·ξ₀, U)`) rather than forming
    # h' as its own point.
    fs, xi0 = fs.seed(commitment, x, value)

    # The fold is a `lax.scan` over the round count, not a Python unroll: the unroll
    # recompiles per size and its compile time grows ~linearly in k (zorch#344
    # measured ≈23 s/round for an on-device-FS open, so a 2²⁰ open extrapolates to
    # minutes), whereas the scan compiles in O(1). A scan needs a fixed-shape carry,
    # so the half-collapsing a/b/G stay full size n; round j reads the active half at
    # [half_j : half_j+hn] (half_j = n>>(j+1)) with a `dynamic_slice` and masks the
    # inactive tail (idx ≥ half_j) of the msm scalars to 0 — byte-identical to the
    # shrinking unroll (0·P = identity), and the folded carry's tail is never read
    # again (round j+1 reads only [0:half_j/2]). The L/R cross terms ride the carry as
    # `.at[j].set` buffers — `lax.scan` cannot auto-stack G1-affine outputs (no
    # zero-affine constant to init the accumulator); each slot is seeded from a real
    # point (commitment) and overwritten.
    lr0 = jnp.broadcast_to(commitment, (k,))

    def _round(
        carry: tuple[Array, Array, Array, _Ch, Array, Array], j: Array
    ) -> tuple[tuple[Array, Array, Array, _Ch, Array, Array], None]:
        a, b, g, fs, ls, rs = carry
        half = lax.shift_right_logical(jnp.int32(n), j + 1)
        active = idx < half
        a_lo, b_lo, g_lo = a[:hn], b[:hn], g[:hn]
        a_hi = lax.dynamic_slice(a, (half,), (hn,))
        b_hi = lax.dynamic_slice(b, (half,), (hn,))
        g_hi = lax.dynamic_slice(g, (half,), (hn,))
        a_lo_m = jnp.where(active, a_lo, fq0)
        a_hi_m = jnp.where(active, a_hi, fq0)

        # arkworks L/R: L pairs a_hi with G_lo, R pairs a_lo with G_hi; the
        # inner-product cross term rides h' = U·ξ₀ (the ·ξ₀ on the U scalar). Only the
        # a-side scalars need masking — `a_*_m` is already 0 past half, so its product
        # with the unmasked b half is too.
        lj = lax.msm(
            jnp.concatenate([a_hi_m, (jnp.sum(a_hi_m * b_lo) * xi0)[None]]),
            jnp.concatenate([g_lo, key.u[None]]),
        )
        rj = lax.msm(
            jnp.concatenate([a_lo_m, (jnp.sum(a_lo_m * b_hi) * xi0)[None]]),
            jnp.concatenate([g_hi, key.u[None]]),
        )
        fs, uj = fs.challenge(lj, rj)
        uj_inv = one / uj

        # Basis fold G_lo + u·G_hi: scalar-mul widens the high half to the jacobian
        # accumulator, so lift the unscaled low half into that same representation
        # before the point-add, then narrow back to affine for the next round's msm
        # input. Convert explicitly to `g_hi·uj`'s dtype rather than via `· one`: a
        # scalar-mul by one is folded away under `@jit`, leaving the low half affine
        # and the point-add a representation mismatch (affine + jacobian).
        g_hi_scaled = g_hi * uj
        a = a.at[:hn].set(a_lo + a_hi * uj_inv)
        b = b.at[:hn].set(b_lo + b_hi * uj)
        g = g.at[:hn].set(
            lax.convert_element_type(
                lax.convert_element_type(g_lo, g_hi_scaled.dtype) + g_hi_scaled, affine
            )
        )
        return (a, b, g, fs, ls.at[j].set(lj), rs.at[j].set(rj)), None

    (a, _, _, fs, ls, rs), _ = lax.scan(
        _round, (a, b, g, fs, lr0, lr0), jnp.arange(k, dtype=jnp.int32)
    )
    return fs, value, IpaProof(ls, rs, a[0])


def _open_one_zk(
    key: IpaKey,
    commitment: Array,
    coeffs: Array,
    x: Array,
    hiding_poly: Array,
    hiding_rand: Array,
    commitment_randomness: Array,
    fs: _ZkCh,
) -> tuple[_ZkCh, Array, IpaZkProof]:
    """Hiding/zk open of one (poly, point): commit a blinding polynomial, fold it
    into the statement under one extra challenge, then run the *same* fold as
    `_open_one` on the blinded `(commitment, coeffs)`. The blinding hides the
    witness while preserving the opened value — the blinding polynomial is shifted
    to vanish at `x`, so `mod_coeffs(x) = coeffs(x)`. Challenger-generic, like
    `_open_one`; the byte-exact arkworks variant lives in the consumer's
    `ZkIpaChallenger`. Requires the key's blinding generator `key.s`."""
    s = key.s
    if s is None:
        raise ValueError("zk opening requires the blinding generator key.s")
    n = coeffs.shape[0]
    one = jnp.ones((), dtype=coeffs.dtype)
    b = powers(x, n)
    value = jnp.sum(coeffs * b)  # p(x); the blinding below preserves it

    # Shift the blinding poly to vanish at x (keeps the value), Pedersen-commit it
    # under the blinding generator s, and squeeze the one hiding challenge hc. The
    # shift is an elementwise select on the constant term (not a scatter) so the
    # open stays one fused kernel.
    raw_eval = jnp.sum(hiding_poly * b)
    hiding_poly = jnp.where(jnp.arange(n) == 0, hiding_poly - raw_eval, hiding_poly)
    hiding_comm = _hiding_commit(hiding_poly, hiding_rand, key.basis, s)
    fs, hc = fs.hiding_challenge(commitment, hiding_comm, x, value)

    # Fold the blinding into the statement: the blinded coefficients, the
    # accumulated randomness, and the blinded commitment the fold actually opens
    # (commitment + hc·hiding_comm − s·combined_rand).
    mod_coeffs = coeffs + hc * hiding_poly
    combined_rand = commitment_randomness + hc * hiding_rand
    mod_commitment = lax.msm(
        jnp.stack([one, hc, -combined_rand]),
        jnp.stack([commitment, hiding_comm, s]),
    )

    fs, _, proof = _open_one(key, mod_commitment, mod_coeffs, x, fs)
    return fs, value, IpaZkProof(proof.l, proof.r, proof.a, hiding_comm, combined_rand)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsProver[IpaCommitment, IpaProverData, list[IpaProof]]] = IpaProver
