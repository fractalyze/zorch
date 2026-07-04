# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round-owned message-observe framing: a sumcheck round may absorb its round poly
per-element scalar (`KIND_SCALAR`, no length prefix) instead of the default
count-prefixed slice, matching a byte challenger's element-at-a-time convention
(flock's `Challenger.observe_f128` / `sample_f128`). The `scalar_framing` flag is
field-agnostic; koalabear stands in here for a byte-framed binary field.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.sha256_field_transcript import Sha256FieldTranscript
from zorch.sumcheck.prover import INF, SumcheckRound, observe_and_sample_msg, prove
from zorch.sumcheck.testing import product
from zorch.sumcheck.verifier import (
    InfDomainSumcheckRound,
)
from zorch.sumcheck.verifier import (
    SumcheckRound as VerifySumcheckRound,
)
from zorch.testkit.random_field import rand_field
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont

# A verifier round: (claim, msg, transcript) -> (reduced_claim, transcript, r, ok).
_VerifierRound = Callable[
    [Array, Array, Transcript], tuple[Array, Transcript, Array, Array]
]


def _msg(vals: Sequence[int]) -> Array:
    return jnp.asarray(vals, dtype=jnp.uint32).view(KB)


def _verifier_challenges(
    vround: _VerifierRound,
    claim: Array,
    proof: Array,
    n: int,
    *,
    domain: bytes = b"reuse",
) -> Array:
    """Thread a verifier round over the proof from a fresh transcript seeded with
    the same `domain` the prover used, returning the challenges it derived.

    The per-round identity `ok` is deliberately NOT asserted: over koalabear a
    `Sha256FieldTranscript` samples raw bytes `.view(dtype)`, which for a prime
    field is non-canonical ~half the time (>= the modulus) and breaks the
    interpolation identity -- a koalabear stand-in artifact, not a framing one (the
    slice path breaks identically). flock's GF(2^128) has no such issue: every
    16-byte pattern is a valid element. What this seam guards is Fiat-Shamir
    lockstep -- prover and verifier absorb the same byte stream and derive the same
    challenges -- which holds regardless of canonicity."""
    transcript: Transcript = Sha256FieldTranscript.new(domain, KB)
    challenges = []
    for i in range(n):
        claim, transcript, r, _ok = vround(claim, proof[i], transcript)
        challenges.append(r)
    return jnp.stack(challenges)


class FramingHelperTest(absltest.TestCase):
    def test_slice_framing_matches_observe_and_sample(self) -> None:
        # scalar_framing=False is the transcript's own count-prefixed slice framing
        # -- a direct observe_and_sample, byte-identical to today's driver.
        msg = _msg([1, 2, 3])
        t0 = Sha256FieldTranscript.new(b"framing", KB)
        got_t, got_r = observe_and_sample_msg(t0, msg, 1, scalar_framing=False)
        want_t, want_r = t0.observe_and_sample(msg, 1)
        self.assertTrue(bool(jnp.array_equal(got_r, want_r)))
        got = cast(Sha256FieldTranscript, got_t)
        self.assertTrue(bool(jnp.array_equal(got.state.h, want_t.state.h)))

    def test_scalar_framing_matches_per_element_observe_scalar(self) -> None:
        # scalar_framing=True observes each element via observe_scalar (no length
        # prefix) and squeezes via sample_scalar -- flock's per-element convention.
        msg = _msg([1, 2, 3])
        t0 = Sha256FieldTranscript.new(b"framing", KB)
        got_t, got_r = observe_and_sample_msg(t0, msg, 1, scalar_framing=True)
        ref = t0
        for i in range(msg.shape[0]):
            ref = ref.observe_scalar(msg[i])
        ref, want_r = ref.sample_scalar()
        self.assertTrue(bool(jnp.array_equal(got_r, want_r)))
        got = cast(Sha256FieldTranscript, got_t)
        self.assertTrue(bool(jnp.array_equal(got.state.h, ref.state.h)))

    def test_scalar_and_slice_framing_diverge(self) -> None:
        # The two framings are not the same stream: the length prefix the slice path
        # absorbs shifts every downstream challenge.
        msg = _msg([1, 2, 3])
        t0 = Sha256FieldTranscript.new(b"framing", KB)
        _, r_slice = observe_and_sample_msg(t0, msg, 1, scalar_framing=False)
        _, r_scalar = observe_and_sample_msg(t0, msg, 1, scalar_framing=True)
        self.assertFalse(bool(jnp.array_equal(r_slice, r_scalar)))


class FramingReuseTest(absltest.TestCase):
    def test_scalar_framing_applied_and_locksteps_natural(self) -> None:
        # A scalar_framing product round drives prove's scan over a scalar-capable
        # transcript, and the scalar-framed verifier stays in Fiat-Shamir lockstep.
        # Asserts (1) scalar framing is actually applied -- prover challenges differ
        # from the slice-framed prove on the same factors; (2) prover and verifier
        # derive identical challenges every round (a prover-scalar / verifier-slice
        # mismatch would desync here).
        n = 6
        factors = [rand_field(1, (1 << n,), KB), rand_field(2, (1 << n,), KB)]
        _, _, scalar = prove(
            SumcheckRound(degree=2, scalar_framing=True),
            factors,
            Sha256FieldTranscript.new(b"reuse", KB),
        )
        _, _, slice_ = prove(
            SumcheckRound(degree=2, scalar_framing=False),
            factors,
            Sha256FieldTranscript.new(b"reuse", KB),
        )
        self.assertFalse(bool(jnp.array_equal(scalar.challenge, slice_.challenge)))
        challenges = _verifier_challenges(
            VerifySumcheckRound(degree=2, scalar_framing=True),
            jnp.sum(product(factors)),
            scalar.round_poly,
            n,
        )
        self.assertTrue(bool(jnp.array_equal(scalar.challenge, challenges)))

    def test_scalar_framing_locksteps_inf_domain(self) -> None:
        # Scalar framing composes with the round-owned ∞-domain: a
        # SumcheckRound(domain=(1, INF), scalar_framing=True) sends the 2-element
        # (s(1), s(∞)) message per-element scalar and the scalar-framed ∞-verifier
        # stays in lockstep -- the closest koalabear stand-in for flock's actual
        # round (∞-trick + per-element scalar framing).
        n = 6
        factors = [rand_field(3, (1 << n,), KB), rand_field(4, (1 << n,), KB)]
        _, _, msgs = prove(
            SumcheckRound(degree=2, domain=(1, INF), scalar_framing=True),
            factors,
            Sha256FieldTranscript.new(b"reuse", KB),
        )
        self.assertEqual(msgs.round_poly.shape, (n, 2))  # (s(1), s(inf)) per round
        challenges = _verifier_challenges(
            InfDomainSumcheckRound(degree=2, scalar_framing=True),
            jnp.sum(product(factors)),
            msgs.round_poly,
            n,
        )
        self.assertTrue(bool(jnp.array_equal(msgs.challenge, challenges)))


if __name__ == "__main__":
    absltest.main()
