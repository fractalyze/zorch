# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""IPA prover: commit and open by log-n basis folding.

`commit` is the Pedersen MSM `P = ⟨a, G⟩ = msm(coeffs, basis)`. `open` proves
`p(x) = ⟨a, b⟩` for `b = (1, x, …, x^{n-1})` by the Bulletproofs/Halo fold: each
of the `k = log₂ n` rounds sends two cross-term group elements

    L_j = ⟨a_lo, G_hi⟩ + ⟨a_lo, b_hi⟩·U
    R_j = ⟨a_hi, G_lo⟩ + ⟨a_hi, b_lo⟩·U

absorbs them into the Fiat-Shamir transcript, samples a challenge `u_j`, and folds
all three vectors in half

    a ← a_lo·u_j   + a_hi·u_j⁻¹
    b ← b_lo·u_j⁻¹ + b_hi·u_j
    G ← G_lo·u_j⁻¹ + G_hi·u_j

until each collapses to a single element. Each cross term is one `lax.msm` (the U
term folded in as one extra (scalar, point) pair), so the only raw EC arithmetic
is the basis fold `G_lo·u⁻¹ + G_hi·u` — vectorized scalar-mul and point-add, with
the result converted back to affine each round to keep the point representation
(and thus the next round's `lax.msm` input) stable. The fold is a Python `for`
over the static round count, so each round lowers to one fused kernel (the same
shape as the FRI prover's fold loop), not a `lax.scan` carry.

Scope: one base-field polynomial per opening, power-of-two length, no hiding
(`U`-blinding omitted, matching the study note). A demonstration of the seam, not
a hardened prover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, lax

from zorch.pcs.ipa.config import IpaCommitment, IpaProof
from zorch.pcs.ipa.math import _check_pow2, inner_powers
from zorch.pcs.ipa.setup import IpaKey
from zorch.transcript import Transcript, sample_challenge

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver


@dataclass(frozen=True)
class IpaProverData:
    """Retained witness from `IpaProver.commit`: the coefficient vectors, kept to
    drive the fold in `open`. Holds references to the (immutable) inputs — no
    polynomial data is copied."""

    coeffs: tuple[Array, ...]


@dataclass(frozen=True)
class IpaProver:
    key: IpaKey

    def commit(self, polys: Sequence[Array]) -> tuple[IpaCommitment, IpaProverData]:
        """Pedersen-commit a batch of coefficient vectors: `P_j = ⟨a_j, G⟩`.
        Returns the stacked G1 commitments and the coeffs as prover data."""
        commitments = [lax.msm(c, self.key.basis[: c.shape[0]]) for c in polys]
        return jnp.stack(commitments), IpaProverData(tuple(polys))

    def open(
        self,
        prover_data: IpaProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, list[IpaProof], Transcript]:
        """Open poly `j` at `points[j]`. Returns `(values, proofs, transcript)`
        with `values[j] = p_j(points[j])` and one `IpaProof` per opening, the
        transcript threaded through every round's challenge."""
        if len(prover_data.coeffs) != len(points):
            raise ValueError(
                f"batch mismatch: {len(prover_data.coeffs)} polys vs "
                f"{len(points)} points"
            )
        values, proofs = [], []
        t = transcript
        for coeffs, x in zip(prover_data.coeffs, points):
            t, value, proof = _open_one(self.key, coeffs, x, t)
            values.append(value)
            proofs.append(proof)
        return jnp.stack(values), proofs, t


def _open_one(
    key: IpaKey, coeffs: Array, x: Array, transcript: Transcript
) -> tuple[Transcript, Array, IpaProof]:
    """Fold one (poly, point) to a proof. Returns `(transcript, value, proof)`."""
    n = coeffs.shape[0]
    k = _check_pow2(n)
    affine = key.basis.dtype  # the point representation msm consumes
    one = jnp.ones((), dtype=coeffs.dtype)

    a = coeffs
    b = inner_powers(x, n)
    g = key.basis[:n]
    value = jnp.sum(a * b)  # ⟨a, b⟩ = p(x)

    ls, rs = [], []
    t = transcript
    for _ in range(k):
        m = a.shape[0] // 2
        a_lo, a_hi = a[:m], a[m:]
        b_lo, b_hi = b[:m], b[m:]
        g_lo, g_hi = g[:m], g[m:]

        # Each cross term: an MSM over the half-basis with the inner-product
        # value folded in as one extra (scalar, point) pair against U.
        cl = lax.msm(
            jnp.concatenate([a_lo, jnp.sum(a_lo * b_hi)[None]]),
            jnp.concatenate([g_hi, key.u[None]]),
        )
        cr = lax.msm(
            jnp.concatenate([a_hi, jnp.sum(a_hi * b_lo)[None]]),
            jnp.concatenate([g_lo, key.u[None]]),
        )
        t, uj = sample_challenge(t.observe(jnp.stack([cl, cr])), coeffs.dtype)
        uj_inv = one / uj

        a = a_lo * uj + a_hi * uj_inv
        b = b_lo * uj_inv + b_hi * uj
        g = lax.convert_element_type(g_lo * uj_inv + g_hi * uj, affine)
        ls.append(cl)
        rs.append(cr)

    return t, value, IpaProof(jnp.stack(ls), jnp.stack(rs), a[0])


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsProver[IpaCommitment, IpaProverData, list[IpaProof]]] = IpaProver
