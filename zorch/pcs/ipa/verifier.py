# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""IPA verifier, split into its cheap half and its one expensive MSM.

The verifier's work divides cleanly (study note §1.4/§2.3):

- **cheap (O(log n))** — replay the round challenges from `L_j, R_j`, fold the
  commitment to the left-hand point
  `Q = P + v·U + Σ_j (u_j·L_j + u_j⁻¹·R_j)`, and evaluate `b = h(x)` from the
  challenges alone. `reduce_opening` does exactly this and returns an
  `IpaReducedClaim` — the deferred statement "`Q == a·G_final + a·b·U` where
  `G_final = ⟨s, G⟩`", carried as its succinct witness (the challenges) without
  ever touching the size-`n` basis.
- **expensive (O(n))** — the one MSM `G_final = ⟨s, G⟩`. `verify` pays it inline
  to settle a single opening; an accumulation scheme instead *keeps* the
  `IpaReducedClaim`, folds many of them, and pays one MSM at the very end (the
  decider). That reuse is the whole reason this split is a public seam and not a
  private helper.

The point combinations are `lax.msm`s (GPU-only on this stack); the field-only
pieces (`b = h(x)`, the size-`n` `s`) come from `math.py`, which a CPU test
exercises without the EC path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import jax.numpy as jnp
from jax import Array, lax

from zorch.pcs.ipa.challenger import IpaChallenger, TranscriptChallenger
from zorch.pcs.ipa.config import IpaCommitment, IpaProof
from zorch.pcs.ipa.math import challenge_vector, eval_challenge_poly
from zorch.pcs.ipa.setup import IpaKey
from zorch.transcript import Transcript

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier

_Ch = TypeVar("_Ch", bound=IpaChallenger)


@dataclass(frozen=True)
class IpaReducedClaim:
    """The cheap reduction of one IPA opening — everything but the size-`n` MSM.

    `combined` is the folded left-hand point `Q`; `u` are the round challenges
    (the succinct representation of the check polynomial `h`, and of `s` via
    `math.challenge_vector`); `a` is the proof's collapsed coefficient; `b` is
    `h(x)`, the collapsed evaluation. The deferred statement is
    `combined == a·⟨s, G⟩ + a·b·U`; settling it needs only the one MSM `⟨s, G⟩`,
    which `verify` runs and an accumulator defers."""

    combined: Array  # G1 affine — Q = P + v·U + Σ (u·L + u⁻¹·R)
    u: Array  # scalar field [k] — round challenges
    a: Array  # scalar field — collapsed coefficient (from the proof)
    b: Array  # scalar field — h(x), the collapsed evaluation


def reduce_opening(
    key: IpaKey,
    commitment: Array,
    point: Array,
    value: Array,
    proof: IpaProof,
    fs: _Ch,
) -> tuple[_Ch, IpaReducedClaim]:
    """The O(log n) half of verification for one opening: replay challenges, fold
    the commitment to `Q`, and evaluate `b = h(point)`. Touches no size-`n` MSM —
    the seam an accumulation scheme reuses to defer the expensive check. Derives
    challenges through the injected `fs` (challenger-generic), so an accumulation
    consumer drives it with the same arkworks-faithful FS as its prover."""
    k = proof.l.shape[0]
    one = jnp.ones((), dtype=value.dtype)

    fs = fs.seed(commitment, point, value)
    us, us_inv = [], []
    for j in range(k):
        fs, uj = fs.challenge(proof.l[j], proof.r[j])
        us.append(uj)
        us_inv.append(one / uj)
    u = jnp.stack(us)
    u_inv = jnp.stack(us_inv)

    # Q = P + v·U + Σ_j (u_j·L_j + u_j⁻¹·R_j), one MSM over O(log n) points.
    scalars = [one, value]
    pts = [commitment, key.u]
    for j in range(k):
        scalars += [u[j], u_inv[j]]
        pts += [proof.l[j], proof.r[j]]
    combined = lax.msm(jnp.stack(scalars), jnp.stack(pts))

    b = eval_challenge_poly(u, point)
    return fs, IpaReducedClaim(combined, u, proof.a, b)


def settle(key: IpaKey, claim: IpaReducedClaim) -> Array:
    """Pay one reduced claim's deferred debt: the size-`n` MSM `G_final = ⟨s, G⟩`
    and the final point identity `Q == a·G_final + a·b·U`. Returns a scalar bool.
    Factored out so `verify` and an accumulation decider settle by one code
    path."""
    s = challenge_vector(claim.u)
    g_final = lax.msm(s, key.basis[: s.shape[0]])
    rhs = lax.msm(jnp.stack([claim.a, claim.a * claim.b]), jnp.stack([g_final, key.u]))
    return jnp.all(claim.combined == rhs)


@dataclass(frozen=True)
class IpaVerifier:
    key: IpaKey

    def verify(
        self,
        commitment: IpaCommitment,
        points: Sequence[Array],
        values: Array,
        proof: list[IpaProof],
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Check each opening: reduce to `Q` (cheap), then settle the deferred
        MSM (expensive). Returns `(all_ok, transcript)`. Wraps the transcript in the
        default `TranscriptChallenger` (zorch-native FS); a byte-exact consumer
        drives `reduce_opening` with its own `IpaChallenger`."""
        k = commitment.shape[0]
        if not len(points) == values.shape[0] == len(proof) == k:
            raise ValueError(
                f"batch mismatch: commitment={k}, points={len(points)}, "
                f"values={values.shape[0]}, proof={len(proof)}"
            )
        fs = TranscriptChallenger(transcript, values.dtype)
        oks = []
        for c, x, v, pf in zip(commitment, points, values, proof):
            fs, claim = reduce_opening(self.key, c, x, v, pf, fs)
            oks.append(settle(self.key, claim))
        return jnp.all(jnp.stack(oks)), fs.transcript


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[IpaCommitment, list[IpaProof]]] = IpaVerifier
