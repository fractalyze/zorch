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
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, lax

from zorch.pcs.kzg.config import KzgCommitment, KzgProof
from zorch.pcs.kzg.setup import KzgProvingKey
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsProver


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
class KzgProverData:
    """Retained witness from `KzgProver.commit`: the coefficient vectors, kept to
    build the opening quotients. The tuple holds references to the (immutable)
    input arrays — no polynomial data is copied."""

    coeffs: tuple[Array, ...]


@dataclass(frozen=True)
class KzgProver:
    pk: KzgProvingKey

    def commit(self, polys: Sequence[Array]) -> tuple[KzgCommitment, KzgProverData]:
        """Commit a batch of coefficient-basis polynomials. Returns the stacked G1
        commitments and the coeffs as prover data (kept to build quotients)."""
        commitments = [lax.msm(c, self.pk.powers_g1[: c.shape[0]]) for c in polys]
        return jnp.stack(commitments), KzgProverData(tuple(polys))

    def open(
        self,
        prover_data: KzgProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, KzgProof, Transcript]:
        """Open poly `j` at `points[j]`. Returns `(values, proofs, transcript)`.
        KZG runs no fold rounds, so the transcript passes through unchanged for a
        single (poly, point) each; batching many openings into one proof (a γ
        sampled from the transcript) is a later extension."""
        if len(prover_data.coeffs) != len(points):
            raise ValueError(
                f"batch mismatch: {len(prover_data.coeffs)} polys vs "
                f"{len(points)} points"
            )
        values, proofs = [], []
        for coeffs, z in zip(prover_data.coeffs, points):
            q, fz = _quotient_and_eval(coeffs, z)
            proofs.append(lax.msm(q, self.pk.powers_g1[: q.shape[0]]))
            values.append(fz)
        return jnp.stack(values), jnp.stack(proofs), transcript


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/pcs.md "Instance anatomy".
    _: type[PcsProver[KzgCommitment, KzgProverData, KzgProof]] = KzgProver
