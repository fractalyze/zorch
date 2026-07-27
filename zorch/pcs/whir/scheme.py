# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The WHIR opening scheme — the four scheme-specific maps the round driver
delegates, so a consumer can byte-match a specific reference (e.g.
openvm-stark-backend's SWIRL) without forking the driver.

WHIR's round machinery (sumcheck folds, per-round RS re-encode + out-of-domain
sample, strided query consistency, final constraint) proves
`claim = Σ_x f̂(x)·ŵ(x)` for *whatever* initial message `f̂` and weight `ŵ` it is
handed — it is agnostic to how those are built. Four things, and only four,
depend on the scheme:

1. how the committed columns become the initial sumcheck message `f̂`
   (`combined_f_evals`) — a plain MLE for the self-test, the prismalinear
   eval→coeff RS message for SWIRL;
2. how the columns' *claimed evaluations* are read off at the opening point
   (`claimed_values`);
3. the initial weight table `ŵ` the sumcheck folds (`initial_weight`) — plain
   `eq(z, ·)` here, SWIRL's möbius-adjusted `eq` there;
4. the matching closed form of that weight at the fully-folded point, the
   final-constraint prefix (`final_prefix`).

Everything else (the out-of-domain and per-query weight updates, which are plain
`eq` in every known WHIR variant) stays in the driver. A scheme instance is a
`@jit` static key on the prover/verifier, so it must be a frozen, hashable value;
its methods are pure and run inside the driver's `@jit` zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import frx.numpy as fnp
from frx import Array

from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.poly.univariate import eval_coeffs
from zorch.transcript import TranscriptT


@runtime_checkable
class WhirScheme(Protocol):
    """The scheme-specific maps of a WHIR opening. Implementations are frozen,
    hashable (they ride a prover/verifier `@jit` static key) and their methods are
    jit-traceable pure functions. `mle` is the committed columns `(S, num_polys)`,
    `z` the opening point `(m,)`, `mu` the batch-combine challenge, `alphas` the
    `(m,)` stack of per-fold sumcheck challenges in fold order."""

    def bind(
        self, transcript: TranscriptT, commitment: Array, values: Array
    ) -> TranscriptT:
        """Bind the commitment and claimed values into the transcript before μ is
        sampled. The default absorbs both — a standalone PCS must commit to what it
        opens. A consumer whose larger protocol already bound the commitment in an
        earlier stage (so WHIR opens against an existing commitment) overrides this
        to a no-op, keeping the Fiat-Shamir stream byte-exact with that reference."""
        ...

    def claimed_values(self, mle: Array, z: Array) -> Array:
        """The per-column claimed evaluations `(num_polys,)` the proof opens to."""
        ...

    def combined_f_evals(self, mle: Array, mu: Array) -> Array:
        """The initial sumcheck message `f̂` `(S,)` — the columns reduced to one
        polynomial by the μ-power batch combine."""
        ...

    def initial_weight(self, z: Array) -> Array:
        """The initial weight table `ŵ` `(2^m,)` the first round's sumcheck folds;
        its inner product with `f̂` is the opened claim."""
        ...

    def final_prefix(self, z: Array, alphas: Array) -> Array:
        """The final-constraint contribution of the initial weight: that weight
        evaluated through every fold, i.e. the closed form of `initial_weight`'s
        multilinear at the fold challenges."""
        ...


@dataclass(frozen=True)
class EqWhirScheme:
    """The default scheme — a plain multilinear opening at a point. The columns are
    used as MLEs directly, the weight is `eq(z, ·)` (so the claim is the MLE
    evaluated at `z`), and the final prefix is `eq(z, ᾱ)` with the folds bound
    LSB-first (hence the reversal, mirroring the `[0::2]/[1::2]` fold order). This
    is the self-test scheme and the behaviour the driver had before the seam."""

    def bind(
        self, transcript: TranscriptT, commitment: Array, values: Array
    ) -> TranscriptT:
        return transcript.observe(commitment).observe(values)

    def claimed_values(self, mle: Array, z: Array) -> Array:
        return eval_mle(mle, z, axis=0)

    def combined_f_evals(self, mle: Array, mu: Array) -> Array:
        return eval_coeffs(mle.astype(mu.dtype), mu)

    def initial_weight(self, z: Array) -> Array:
        return expand_eq_to_hypercube(z, fnp.ones((), z.dtype))

    def final_prefix(self, z: Array, alphas: Array) -> Array:
        return eval_eq(z, alphas[::-1])
