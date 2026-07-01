# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The IPA-PC challenge source — the Fiat-Shamir seam the reuse path injects.

`reduce_opening`/`settle` (the verifier reuse seam) and the prover fold derive
their challenges through an `IpaChallenger` rather than a `zorch.transcript`
directly. That indirection is what lets an accumulation consumer drive the fold
with an **arkworks-faithful** Fiat-Shamir — a fresh domain-separated sponge per
round, the previous challenge re-absorbed, a nonnative truncated squeeze — which
does not fit zorch's running `Transcript` (`observe`/`sample`) shape at all (see
the accumulation-zorch IPA-PC port and zorch#339). zorch ships the running-
transcript default below; the byte-exact arkworks challenger lives in the
consumer, matching the scheme-agnostic split of zorch#295.

`IpaProver.open` / `IpaVerifier.verify` stay `Transcript`-typed `PcsProver` /
`PcsVerifier` methods: they wrap the transcript in the default challenger here, so
the public seam is unchanged and the injection point is the challenger-generic
free functions (`reduce_opening`, the prover's `_open_one`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, Self

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_dataclass

from zorch.transcript import Transcript, sample_challenge


class IpaChallenger(Protocol):
    """Derives the IPA fold challenges. `seed` binds the opening statement
    `(commitment, point, value)` before the rounds and returns the *seed challenge*
    ξ₀, which scales the inner-product generator into `h' = U·ξ₀` (arkworks
    `ipa_pc`'s `h_prime = svk.h·ξ₀`); `challenge` absorbs a round's cross terms
    `l`, `r` and returns the fold challenge. Both also return the advanced
    challenger (threaded functionally, like `Transcript`).

    An implementation must be a registered JAX pytree — its device-resident
    Fiat-Shamir state the data leaves, its config the static meta — because the
    prover's fold (`_open_one`) carries the challenger through a `lax.scan`."""

    def seed(
        self, commitment: Array, point: Array, value: Array
    ) -> tuple[Self, Array]: ...

    def challenge(self, l: Array, r: Array) -> tuple[Self, Array]: ...


class ZkIpaChallenger(IpaChallenger, Protocol):
    """An `IpaChallenger` that also derives the *hiding challenge* of the zk/hiding
    opening: the one extra challenge squeezed from the statement and the blinding
    commitment `(commitment, hiding_comm, point, value)` before the rounds. It is
    what the prover/verifier fold the blinding into `(commitment, coeffs)` with
    (the arkworks hiding fold), and is squeezed once — it does not enter the
    per-round challenge list. Separate from `IpaChallenger` so the transparent
    path's challengers need not implement it."""

    def hiding_challenge(
        self, commitment: Array, hiding_comm: Array, point: Array, value: Array
    ) -> tuple[Self, Array]: ...


@partial(register_dataclass, data_fields=["transcript"], meta_fields=["dtype"])
@dataclass(frozen=True)
class TranscriptChallenger:
    """The default `IpaChallenger`: zorch's own running `DuplexTranscript`. `seed`
    binds the statement `(commitment, point, value)` and squeezes the seed
    challenge ξ₀ (the inner-product generator scale `h' = U·ξ₀`); `challenge`
    observes the round's cross terms and squeezes one `dtype` challenge;
    `hiding_challenge` squeezes the zk opening's pre-fold blinding challenge. This is
    the zorch-native FS, NOT arkworks-byte-exact, and serves as the default
    `ZkIpaChallenger` as well as `IpaChallenger`.

    A JAX pytree (`transcript` is the data leaf — itself a pytree; `dtype` is
    static) so the prover's fold carries it through its `lax.scan` (`_open_one`)."""

    transcript: Transcript
    dtype: Any  # the challenge field (a zk_dtypes scalar-field dtype)

    def seed(
        self, commitment: Array, point: Array, value: Array
    ) -> tuple[TranscriptChallenger, Array]:
        t = self.transcript.observe(commitment).observe(jnp.stack([point, value]))
        t, xi0 = sample_challenge(t, self.dtype)
        return TranscriptChallenger(t, self.dtype), xi0

    def challenge(self, l: Array, r: Array) -> tuple[TranscriptChallenger, Array]:
        t, u = sample_challenge(self.transcript.observe(jnp.stack([l, r])), self.dtype)
        return TranscriptChallenger(t, self.dtype), u

    def hiding_challenge(
        self, commitment: Array, hiding_comm: Array, point: Array, value: Array
    ) -> tuple[TranscriptChallenger, Array]:
        """Squeeze the hiding challenge over `(commitment, hiding_comm, point,
        value)` — the zorch-native (NOT arkworks-byte-exact) read; the byte-exact
        version lives in the accumulation consumer's challenger."""
        t = self.transcript.observe(jnp.stack([commitment, hiding_comm])).observe(
            jnp.stack([point, value])
        )
        t, hc = sample_challenge(t, self.dtype)
        return TranscriptChallenger(t, self.dtype), hc
