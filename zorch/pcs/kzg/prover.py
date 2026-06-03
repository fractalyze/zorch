# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""KZG prover: commit and open, both backed by `lax.msm`.

`commit` is `C = Σ aᵢ·[τⁱ]₁ = msm(coeffs, powers_g1)`; `open` at `z` is the same
MSM over the quotient `q(x) = (f(x) − f(z))/(x − z)`, with `f(z)` falling out of
the same synthetic division as the remainder. Both run entirely on the GPU: the
field arithmetic is fusion-ready normal form (the quotient recurrence) and the
MSM lowers to `stablehlo.msm`, a dedicated GPU kernel — so neither hits the
LLVM-NVPTX codegen cliff that fusing raw EC arithmetic would. Polynomials are
taken in the **coefficient basis** (KZG's commitment is over powers of τ); an
evaluation-form input must be interpolated to coefficients first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, lax

from zorch.pcs.kzg.setup import KzgProvingKey
from zorch.transcript import Transcript


def _quotient_and_eval(coeffs: Array, z: Array) -> tuple[Array, Array]:
    """Synthetic division of `f` by `(x − z)`, where `coeffs[i]` is the coefficient
    of `xⁱ` (ascending). Returns `(q, f(z))`: `q` the ascending quotient coeffs and
    the remainder `f(z)`. Degree is static, so this unrolls as a Python loop (the
    repo's leaf-helper idiom — fusion-ready, no `lax.scan` carry)."""
    n = coeffs.shape[0]
    if n < 2:
        raise ValueError(f"opening needs degree >= 1 (>= 2 coeffs), got {n}")
    q = [None] * (n - 1)
    carry = coeffs[n - 1]  # leading quotient coeff
    q[n - 2] = carry
    for i in range(n - 2, 0, -1):
        carry = coeffs[i] + z * carry
        q[i - 1] = carry
    fz = coeffs[0] + z * carry
    return jnp.stack(q), fz


@dataclass(frozen=True)
class KzgProver:
    pk: KzgProvingKey

    def commit(self, polys: Sequence[Array]) -> tuple[Array, list[Array]]:
        """Commit a batch of coefficient-basis polynomials. Returns the stacked G1
        commitments and the coeffs as prover data (kept to build quotients)."""
        commitments = [lax.msm(c, self.pk.powers_g1[: c.shape[0]]) for c in polys]
        return jnp.stack(commitments), list(polys)

    def open(
        self,
        prover_data: Sequence[Array],
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, Array, Transcript]:
        """Open poly `j` at `points[j]`. Returns `(values, proofs, transcript)`.
        KZG runs no fold rounds, so the transcript passes through unchanged for a
        single (poly, point) each; batching many openings into one proof (a γ
        sampled from the transcript) is a later extension."""
        if len(prover_data) != len(points):
            raise ValueError(
                f"batch mismatch: {len(prover_data)} polys vs {len(points)} points"
            )
        values, proofs = [], []
        for coeffs, z in zip(prover_data, points):
            q, fz = _quotient_and_eval(coeffs, z)
            proofs.append(lax.msm(q, self.pk.powers_g1[: q.shape[0]]))
            values.append(fz)
        return jnp.stack(values), jnp.stack(proofs), transcript
