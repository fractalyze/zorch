# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π_eval — completeness, the constant-coefficient claim, and its soundness.

Structural correctness without goldens, like `opening_test`. The statement
this layer proves is weaker than the one below it, and the tests are built
around exactly that gap: a witness whose `F_u` has a zero constant
coefficient but a nonzero remainder must *verify* (it is the whole point),
while one whose constant coefficient is nonzero must be rejected. The
honest instance is therefore constructed to have a genuinely nonzero
`F_u(s1, m)` whose constant coefficient vanishes.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import HostSha256
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteHashTranscript, ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.challenge import ChallengeParams
from zorch.lnp.eval import AbdlopEval, EvalProof
from zorch.lnp.opening import AbdlopOpening

_SPLIT_Q = (4294967197,)
_D = 64
_KAPPA, _ETA, _K = 2, 59, 32
_CHALLENGE = ChallengeParams(d=_D, kappa=_KAPPA, eta=_ETA, k=_K, fail_prob=2.0**-40)

# n rows, m1 committed / m2 randomness columns, ℓ messages, λ garbage
# terms, M linear functions.
_ROWS, _M1, _M2, _ELL, _LAM, _M = 2, 2, 2, 1, 2, 2

_GAMMA = 14.0
_T = _ETA * float(np.sqrt(_M1 * _D))
_STD = _GAMMA * _T
_REP = float(np.exp(14.0 / _GAMMA + 1.0 / (2.0 * _GAMMA**2)))


def _ring() -> HostSplitRing:
    return HostSplitRing(_SPLIT_Q, _D)


def _uniform_stack(
    ring: HostSplitRing, rng: np.random.Generator, *lead: int
) -> np.ndarray:
    return np.stack(
        [
            rng.integers(0, q, size=(*lead, ring.d), dtype=np.uint64)
            for q in ring.q_moduli
        ],
        axis=-2,
    )


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return ByteHashTranscript.new(b"lnp-eval-test", HostSha256()).observe_bytes(tag)


def _eval(ring: HostSplitRing, **overrides: object) -> AbdlopEval:
    """The extended scheme is the whole trick: the inner opening's BDLOP
    half carries ℓ + λ messages, because `m‖g` is what it opens."""
    scheme = AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=_M1,
        randomness_cols=_M2,
        messages=_ELL + _LAM,
        beta1_inf=1,
        beta2_inf=1,
    )
    params: dict[str, object] = dict(
        s1_std=_STD, s2_std=_STD, rep1=_REP, rep2=_REP, challenge=_CHALLENGE
    )
    params.update(overrides)
    return AbdlopEval(AbdlopOpening(scheme, **params), lam=_LAM)  # type: ignore[arg-type]


class _Instance:
    """One honest Π_eval statement: publics, witness, commitment, and M
    linear functions whose constant coefficients vanish while the functions
    themselves do not."""

    def __init__(self, seed: int) -> None:
        self.ring = _ring()
        self.protocol = _eval(self.ring)
        rng = np.random.default_rng(seed)
        self.a1 = _uniform_stack(self.ring, rng, _ROWS, _M1)
        self.a2 = _uniform_stack(self.ring, rng, _ROWS, _M2)
        self.b = _uniform_stack(self.ring, rng, _ELL, _M2)
        self.bg = _uniform_stack(self.ring, rng, _LAM, _M2)
        self.s1 = rng.integers(-1, 2, size=(_M1, _D)).astype(np.int64)
        self.s2 = rng.integers(-1, 2, size=(_M2, _D)).astype(np.int64)
        self.message = _uniform_stack(self.ring, rng, _ELL)
        self.rng = rng

        # The commitment covers m‖g, but g is drawn per proof — the message
        # half is committed here and `prove` appends t_g.
        scheme = self.protocol.opening.scheme
        s1_ring = self.ring.from_signed_stack(self.s1)
        s2_ring = self.ring.from_signed_stack(self.s2)
        self.t_a = self.ring.add(
            self.ring.matvec(self.a1, s1_ring), self.ring.matvec(self.a2, s2_ring)
        )
        self.t_b = self.ring.add(self.ring.matvec(self.b, s2_ring), self.message)
        assert scheme.messages == _ELL + _LAM

        # F_u(s1, m) = Fs1_u·s1 + Fm_u·m − target_u. Pick the matrices, then
        # solve for the target that makes each F_u a *nonzero* ring element
        # with a zero constant coefficient: subtract off only the constant
        # coefficient of the raw value.
        self.fs1 = _uniform_stack(self.ring, rng, _M, _M1)
        self.fm = _uniform_stack(self.ring, rng, _M, _ELL)
        raw = self.ring.add(
            self.ring.matvec(self.fs1, s1_ring), self.ring.matvec(self.fm, self.message)
        )
        self.target = np.zeros_like(raw)
        self.target[..., 0] = raw[..., 0]
        self.raw = raw

    def prove(
        self, tag: bytes = b"", **overrides: np.ndarray
    ) -> tuple[EvalProof, ByteTranscript]:
        args = dict(
            a1=self.a1,
            a2=self.a2,
            b=self.b,
            bg=self.bg,
            fs1=self.fs1,
            fm=self.fm,
            target=self.target,
        )
        args.update(overrides)
        return self.protocol.prove(
            s1=self.s1,
            s2=self.s2,
            message=self.message,
            rng=self.rng,
            transcript=_transcript(tag),
            **args,
        )

    def verify(
        self, proof: EvalProof, tag: bytes = b"", **overrides: np.ndarray
    ) -> bool:
        args = dict(
            a1=self.a1,
            a2=self.a2,
            b=self.b,
            bg=self.bg,
            fs1=self.fs1,
            fm=self.fm,
            target=self.target,
            t_a=self.t_a,
            t_b=self.t_b,
        )
        args.update(overrides)
        ok, _ = self.protocol.verify(proof=proof, transcript=_transcript(tag), **args)
        return ok


class EvalStatementTest(absltest.TestCase):
    def test_the_honest_statement_is_the_weaker_one(self) -> None:
        """Guards the whole suite: the instance must exercise the gap
        between Π_eval and Π_many. Every F_u has a zero constant
        coefficient (what this layer proves) and is nonzero as a ring
        element (so Π_many alone could not prove it)."""
        instance = _Instance(0)
        values = instance.ring.sub(instance.raw, instance.target)
        self.assertFalse(values[..., 0].any())
        for u in range(_M):
            self.assertTrue(values[u].any())


class EvalCompletenessTest(absltest.TestCase):
    def test_an_honest_proof_verifies(self) -> None:
        instance = _Instance(1)
        proof, _ = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_the_masked_aggregates_have_zero_constant_coefficients(self) -> None:
        """The verifier's first check, stated directly: h̃_j = 0 by
        construction on an honest run, because g contributes nothing to the
        constant coefficient and the aggregate's is zero."""
        instance = _Instance(2)
        proof, _ = instance.prove()
        self.assertEqual(proof.h.shape, (_LAM, len(_SPLIT_Q), _D))
        self.assertFalse(proof.h[..., 0].any())

    def test_the_aggregates_are_masked_away_from_the_constant(self) -> None:
        """The other side of the same coin: everything *but* the constant
        coefficient is garbage-masked, so two proofs of the same statement
        publish unrelated h. A leak here would be a zero-knowledge break,
        not a correctness one."""
        first, _ = _Instance(3).prove()
        second, _ = _Instance(4).prove()
        self.assertFalse(np.array_equal(first.h, second.h))

    def test_the_chain_is_byte_deterministic(self) -> None:
        first, _ = _Instance(5).prove()
        second, _ = _Instance(5).prove()
        np.testing.assert_array_equal(first.t_g, second.t_g)
        np.testing.assert_array_equal(first.h, second.h)
        np.testing.assert_array_equal(first.opening.c, second.opening.c)

    def test_distinct_transcripts_yield_distinct_proofs(self) -> None:
        instance = _Instance(6)
        p1, _ = instance.prove(tag=b"one")
        p2, _ = instance.prove(tag=b"two")
        self.assertFalse(np.array_equal(p1.h, p2.h))
        self.assertTrue(instance.verify(p1, tag=b"one"))
        self.assertFalse(instance.verify(p1, tag=b"two"))


class EvalSoundnessTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.instance = _Instance(7)
        cls.proof, _ = cls.instance.prove()

    def test_a_nonzero_constant_coefficient_is_rejected(self) -> None:
        """The statement's own negation: shift one target's constant
        coefficient so that F̃_u ≠ 0. The honest prover then cannot produce
        an h with a zero constant coefficient, so the proof must fail."""
        instance = _Instance(8)
        bad_target = instance.target.copy()
        bad_target[0, 0, 0] = (int(bad_target[0, 0, 0]) + 1) % _SPLIT_Q[0]
        proof, _ = instance.prove(target=bad_target)
        self.assertTrue(proof.h[..., 0].any())
        self.assertFalse(instance.verify(proof, target=bad_target))

    def test_a_tampered_masked_aggregate_is_rejected(self) -> None:
        """h is bound twice over — into the Π_many challenge and into the
        relation target — so moving a non-constant coefficient breaks the
        inner proof even though h̃ stays zero."""
        h = self.proof.h.copy()
        h[0, 0, 1] = (int(h[0, 0, 1]) + 1) % _SPLIT_Q[0]
        self.assertFalse(h[..., 0].any())
        bad = EvalProof(t_g=self.proof.t_g, h=h, opening=self.proof.opening)
        self.assertFalse(self.instance.verify(bad))

    def test_a_tampered_garbage_commitment_is_rejected(self) -> None:
        """t_g feeds γ, so tampering re-derives a different aggregation."""
        t_g = self.proof.t_g.copy()
        t_g[0, 0, 0] = (int(t_g[0, 0, 0]) + 1) % _SPLIT_Q[0]
        bad = EvalProof(t_g=t_g, h=self.proof.h, opening=self.proof.opening)
        self.assertFalse(self.instance.verify(bad))

    def test_a_wrong_linear_function_is_rejected(self) -> None:
        wrong = self.instance.fm.copy()
        wrong[0, 0, 0, 0] = (int(wrong[0, 0, 0, 0]) + 1) % _SPLIT_Q[0]
        self.assertFalse(self.instance.verify(self.proof, fm=wrong))

    def test_a_wrong_commitment_is_rejected(self) -> None:
        other = _uniform_stack(self.instance.ring, np.random.default_rng(99), _ROWS)
        self.assertFalse(self.instance.verify(self.proof, t_a=other))


class EvalSurfaceTest(absltest.TestCase):
    def test_a_non_extended_opening_is_refused(self) -> None:
        """The commonest way to misuse this seam: hand it an opening built
        over the plain scheme, whose BDLOP half has no room for g."""
        ring = _ring()
        plain = AbdlopCommitment(
            ring,
            rows=_ROWS,
            s1_cols=_M1,
            randomness_cols=_M2,
            messages=_ELL,
            beta1_inf=1,
            beta2_inf=1,
        )
        opening = AbdlopOpening(
            plain,
            s1_std=_STD,
            s2_std=_STD,
            rep1=_REP,
            rep2=_REP,
            challenge=_CHALLENGE,
        )
        with self.assertRaisesRegex(ValueError, "extended scheme"):
            AbdlopEval(opening, lam=_ELL + 1)

    def test_a_nonpositive_lam_is_refused(self) -> None:
        """λ = 0 would make the soundness error q1^0 = 1 — a proof of
        nothing — and leave γ with no rows to aggregate into."""
        opening = _eval(_ring()).opening
        with self.assertRaisesRegex(ValueError, "lam must be positive"):
            AbdlopEval(opening, lam=0)

    def test_mismatched_function_matrices_are_refused(self) -> None:
        instance = _Instance(9)
        with self.assertRaisesRegex(ValueError, "fm must lead with"):
            instance.prove(fm=instance.fm[:, :-1])


if __name__ == "__main__":
    absltest.main()
