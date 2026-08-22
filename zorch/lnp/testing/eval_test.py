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

from dataclasses import fields, replace

import numpy as np
from absl.testing import absltest
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.eval import AbdlopEval, EvalProof
from zorch.lnp.opening import AbdlopOpening
from zorch.lnp.testing import lnp_fixture
from zorch.lnp.testing.lnp_fixture import SPLIT_Q as _SPLIT_Q
from zorch.lnp.testing.lnp_fixture import D as _D
from zorch.lnp.testing.lnp_fixture import bump as _bump

# This suite's module shape: n rows, m1 committed / m2 randomness columns,
# ℓ messages, λ garbage terms, M linear functions. `_M1` must match the
# fixture's, which is what its masking standard deviation was derived for.
_ROWS, _M1, _M2 = 2, lnp_fixture.M1, 2
_ELL, _LAM, _M = 1, 2, 2


def _ring() -> HostSplitRing:
    return lnp_fixture.ring()


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return lnp_fixture.transcript(b"lnp-eval-test", tag)


def _scheme(ring: HostSplitRing, messages: int = _ELL + _LAM) -> AbdlopCommitment:
    """The extended scheme is the whole trick: the inner opening's BDLOP
    half carries ℓ + λ messages, because `m‖g` is what it opens."""
    return AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=_M1,
        randomness_cols=_M2,
        messages=messages,
        beta1_inf=1,
        beta2_inf=1,
    )


def _opening(ring: HostSplitRing, messages: int = _ELL + _LAM) -> AbdlopOpening:
    return AbdlopOpening(_scheme(ring, messages), **lnp_fixture.OPENING_PARAMS)  # type: ignore[arg-type]


def _eval(ring: HostSplitRing) -> AbdlopEval:
    return AbdlopEval(_opening(ring), lam=_LAM)


class _Instance:
    """One honest Π_eval statement: publics, witness, commitment, and M
    linear functions whose constant coefficients vanish while the functions
    themselves do not."""

    def __init__(self, seed: int) -> None:
        self.ring = _ring()
        self.protocol = _eval(self.ring)
        rng = np.random.default_rng(seed)
        self.a1 = self.ring.uniform_stack(rng, _ROWS, _M1)
        self.a2 = self.ring.uniform_stack(rng, _ROWS, _M2)
        self.b = self.ring.uniform_stack(rng, _ELL, _M2)
        self.bg = self.ring.uniform_stack(rng, _LAM, _M2)
        self.s1 = rng.integers(-1, 2, size=(_M1, _D)).astype(np.int64)
        self.s2 = rng.integers(-1, 2, size=(_M2, _D)).astype(np.int64)
        self.message = self.ring.uniform_stack(rng, _ELL)
        self.rng = rng

        # The commitment covers m‖g, but g is drawn per proof — the message
        # half is committed here and `prove` appends t_g.
        s1_ring = self.ring.from_signed_stack(self.s1)
        s2_ring = self.ring.from_signed_stack(self.s2)
        commitment = _scheme(self.ring, _ELL).commit(
            self.a1, self.a2, self.b, s1_ring, s2_ring, self.message
        )
        self.t_a, self.t_b = commitment.t_a, commitment.t_b

        # F_u(s1, m) = Fs1_u·s1 + Fm_u·m − target_u. Pick the matrices, then
        # solve for the target that makes each F_u a *nonzero* ring element
        # with a zero constant coefficient: subtract off only the constant
        # coefficient of the raw value.
        self.fs1 = self.ring.uniform_stack(rng, _M, _M1)
        self.fm = self.ring.uniform_stack(rng, _M, _ELL)
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
        self.assertFalse(instance.ring.constant_coeff(values).any())
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
        self.assertFalse(instance.ring.constant_coeff(proof.h).any())

    def test_the_aggregates_are_masked_away_from_the_constant(self) -> None:
        """The other side of the same coin: everything *but* the constant
        coefficient is garbage-masked, so two proofs of **one** statement
        publish unrelated h. A leak here would be a zero-knowledge break,
        not a correctness one — and it has to be the same statement twice,
        since two different ones would differ whether g masked anything or
        not. `_Instance.rng` is stateful, so the second run draws fresh
        garbage against identical publics and witness."""
        instance = _Instance(3)
        first, _ = instance.prove()
        second, _ = instance.prove()
        self.assertFalse(np.array_equal(first.h, second.h))
        self.assertTrue(instance.verify(first))
        self.assertTrue(instance.verify(second))

    def test_the_chain_is_byte_deterministic(self) -> None:
        first, _ = _Instance(5).prove()
        second, _ = _Instance(5).prove()
        np.testing.assert_array_equal(first.t_g, second.t_g)
        np.testing.assert_array_equal(first.h, second.h)
        np.testing.assert_array_equal(first.opening.c, second.opening.c)

    def test_distinct_transcripts_yield_distinct_proofs(self) -> None:
        """Two instances at one seed, so the publics, the witness and the
        garbage draw all match and the **tag** is the only difference —
        otherwise `h` would differ from the fresh `g` alone and say nothing
        about γ. A proof is then bound to the transcript it was made
        against, and to no other."""
        one, two = _Instance(6), _Instance(6)
        p1, _ = one.prove(tag=b"one")
        p2, _ = two.prove(tag=b"two")
        # The control: identical garbage, so h can only diverge through γ.
        np.testing.assert_array_equal(p1.t_g, p2.t_g)
        self.assertFalse(np.array_equal(p1.h, p2.h))
        self.assertTrue(one.verify(p1, tag=b"one"))
        self.assertFalse(one.verify(p1, tag=b"two"))


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
        bad_target = _bump(instance.target, 0, 0, 0)
        proof, _ = instance.prove(target=bad_target)
        self.assertTrue(instance.ring.constant_coeff(proof.h).any())
        self.assertFalse(instance.verify(proof, target=bad_target))

    def test_a_tampered_masked_aggregate_is_rejected(self) -> None:
        """h is bound twice over — into the Π_many challenge and into the
        relation target — so moving a non-constant coefficient breaks the
        inner proof even though h̃ stays zero."""
        h = _bump(self.proof.h, 0, 0, 1)
        self.assertFalse(self.instance.ring.constant_coeff(h).any())
        bad = replace(self.proof, h=h)
        self.assertFalse(self.instance.verify(bad))

    def test_a_tampered_garbage_commitment_is_rejected(self) -> None:
        """t_g feeds γ, so tampering re-derives a different aggregation."""
        t_g = _bump(self.proof.t_g, 0, 0, 0)
        bad = replace(self.proof, t_g=t_g)
        self.assertFalse(self.instance.verify(bad))

    def test_a_wrong_linear_function_is_rejected(self) -> None:
        wrong = _bump(self.instance.fm, 0, 0, 0, 0)
        self.assertFalse(self.instance.verify(self.proof, fm=wrong))

    def test_a_wrong_commitment_is_rejected(self) -> None:
        other = self.instance.ring.uniform_stack(np.random.default_rng(99), _ROWS)
        self.assertFalse(self.instance.verify(self.proof, t_a=other))


class EvalSurfaceTest(absltest.TestCase):
    def test_a_non_extended_opening_is_refused(self) -> None:
        """The commonest way to misuse this seam: hand it an opening built
        over the plain scheme, whose BDLOP half has no room for g."""
        with self.assertRaisesRegex(ValueError, "extended scheme"):
            AbdlopEval(_opening(_ring(), messages=_ELL), lam=_ELL + 1)

    def test_a_nonpositive_lam_is_refused(self) -> None:
        """λ = 0 would make the soundness error q1^0 = 1 — a proof of
        nothing — and leave γ with no rows to aggregate into."""
        with self.assertRaisesRegex(ValueError, "lam must be positive"):
            AbdlopEval(_opening(_ring()), lam=0)

    def test_mismatched_function_matrices_are_refused(self) -> None:
        instance = _Instance(9)
        with self.assertRaisesRegex(ValueError, "fm must be a ring stack"):
            instance.prove(fm=instance.fm[:, :-1])

    def test_a_function_matrix_with_the_wrong_ring_shape_is_refused(self) -> None:
        """The trailing `(limbs, d)` is gated too, not just the leading
        axes — a truncated RNS chain is a different-ring statement and has
        to fail here rather than inside a ring op."""
        instance = _Instance(12)
        with self.assertRaisesRegex(ValueError, "fs1 must be a ring stack"):
            instance.prove(fs1=instance.fs1[..., :-1, :])


class EvalWireContractTest(absltest.TestCase):
    """Malformed *proof* fields are a verdict, malformed *statement* fields
    are an exception — the one rule the proof surface answers by.

    A verifier reads the proof off the wire, so every value an adversary
    controls has to produce `False`; before this, a residue past its
    modulus escaped as a `ValueError` from whichever ring op reached it
    first, which makes a deployed verifier crash on a malformed message
    rather than reject it."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.instance = _Instance(11)
        cls.proof, _ = cls.instance.prove()

    def test_every_proof_field_is_gated(self) -> None:
        """Driven off the dataclass, not a hand-kept list: a field added to
        `EvalProof` without a gate fails here rather than at whatever a
        verifier does with an unchecked value.

        `None` stands in for every malformed value because it is what no
        field can legitimately be — including `opening`, the composite one
        a per-field habit forgets. That omission is what this catches: it
        reached an `AttributeError` out of `verify` instead of a verdict."""
        for field in fields(EvalProof):
            with self.subTest(field.name):
                bad = replace(self.proof, **{field.name: None})
                self.assertFalse(self.instance.verify(bad))

    def test_an_out_of_range_residue_on_the_wire_is_a_verdict(self) -> None:
        """The case the array contract exists for: `q ≤ v < 2^64` is a
        well-typed uint64 and a malformed residue."""
        for name in ("t_g", "h"):
            with self.subTest(name):
                bad = getattr(self.proof, name).copy()
                bad[0, 0, 0] = _SPLIT_Q[0]
                self.assertFalse(
                    self.instance.verify(replace(self.proof, **{name: bad}))
                )

    def test_a_misshaped_wire_stack_is_a_verdict(self) -> None:
        for name in ("t_g", "h"):
            with self.subTest(name):
                bad = getattr(self.proof, name)[:-1]
                self.assertFalse(
                    self.instance.verify(replace(self.proof, **{name: bad}))
                )

    def test_a_wrongly_typed_wire_stack_is_a_verdict(self) -> None:
        """`float64` cannot hold a residue, and `int64` is not the contract
        dtype however well its values fit."""
        for dtype in (np.float64, np.int64):
            with self.subTest(dtype.__name__):
                bad = self.proof.h.astype(dtype)
                self.assertFalse(self.instance.verify(replace(self.proof, h=bad)))

    def test_an_out_of_range_statement_still_raises(self) -> None:
        """The other half of the rule. `target` is the caller's, so a
        malformed one is that caller's bug — returning `False` would report
        a parameter mistake as a failed proof."""
        bad_target = self.instance.target.copy()
        bad_target[0, 0, 0] = _SPLIT_Q[0]
        with self.assertRaises(ValueError):
            self.instance.verify(self.proof, target=bad_target)

    def test_a_misshaped_statement_still_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "target must be a ring stack"):
            self.instance.verify(self.proof, target=self.instance.target[:-1])


if __name__ == "__main__":
    absltest.main()
