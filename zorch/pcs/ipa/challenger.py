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
from typing import Any, Protocol, Self

import jax.numpy as jnp
from jax import Array

from zorch.transcript import Transcript, sample_challenge


class IpaChallenger(Protocol):
    """Derives the IPA fold challenges. `seed` binds the opening statement
    `(commitment, point, value)` before the rounds; `challenge` absorbs a round's
    cross terms `l`, `r` and returns the fold challenge. Both return the advanced
    challenger (threaded functionally, like `Transcript`)."""

    def seed(self, commitment: Array, point: Array, value: Array) -> Self: ...

    def challenge(self, l: Array, r: Array) -> tuple[Self, Array]: ...


@dataclass(frozen=True)
class TranscriptChallenger:
    """The default `IpaChallenger`: zorch's own running `DuplexTranscript`. `seed`
    binds the statement (closing the gap that the bare fold left it unbound);
    `challenge` observes the round's cross terms and squeezes one `dtype`
    challenge — byte-for-byte the fold's prior per-round Fiat-Shamir, now with the
    statement mixed in. This is the zorch-native FS, NOT arkworks-byte-exact."""

    transcript: Transcript
    dtype: Any  # the challenge field (a zk_dtypes scalar-field dtype)

    def seed(
        self, commitment: Array, point: Array, value: Array
    ) -> TranscriptChallenger:
        t = self.transcript.observe(commitment).observe(jnp.stack([point, value]))
        return TranscriptChallenger(t, self.dtype)

    def challenge(self, l: Array, r: Array) -> tuple[TranscriptChallenger, Array]:
        t, u = sample_challenge(self.transcript.observe(jnp.stack([l, r])), self.dtype)
        return TranscriptChallenger(t, self.dtype), u
