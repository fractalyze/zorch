# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π_eval^(2) (Fig. 8) — completeness, soundness, the wire contract, and
the lift layout the whole protocol is indexed against.

Two honest statements are built here, and they are built differently on
purpose. A *relation* vanishes as a ring element, so `r0` is solved as
`−(sᵀR2s + r1ᵀs)` the way `quadratic_test` does it. An *evaluation* only
has to vanish in its constant coefficient, so `e0` is solved as a constant
polynomial — which leaves `F(s)` itself nonzero. A suite that solved both
the same way would be proving `F(s) = 0` and could not tell this protocol
from Fig. 7.

The layout tests exist because the round-trip cannot see it. The garbage
`g` is appended to the *message*, and the σ-lift orbits the message stack
as a whole, so `g` lands inside every automorphism copy rather than after
the message's copies. Prover and verifier index it identically either way,
so a wrong layout would prove a statement about a permutation of the
witness and both sides would agree — hence `test_the_garbage_interleaves…`
and `test_an_embedded_function_is_the_same_function`, which pin the layout
against the ring rather than against the protocol.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from absl.testing import absltest
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.eval import AbdlopQuadraticEval, QuadraticEvalProof
from zorch.lnp.quadratic import (
    SIGMA_ORDER,
    AbdlopQuadratic,
    AbdlopQuadraticMany,
    evaluate,
    lift,
)
from zorch.lnp.testing import lnp_fixture

_ROWS = 2
_M1 = lnp_fixture.M1
_M2 = 3
_ELL = 1
_LAM = 2
_RELATIONS = 2
_EVALUATIONS = 3


def _scheme(
    ring: HostSplitRing, messages: int = _ELL + _LAM, s1_cols: int = _M1
) -> AbdlopCommitment:
    """The **extended** scheme by default: its BDLOP half carries `ℓ + λ`
    messages, because `m‖g` is what the inner protocol opens.

    `messages` is a parameter so the fixture can also build the *narrow*
    scheme the caller commits against — the same shape `eval_test` takes,
    and what lets the commitment come from `commit` rather than be
    re-spelled here."""
    return AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=s1_cols,
        randomness_cols=_M2,
        messages=messages,
        beta1_inf=1,
        beta2_inf=1,
    )


def _protocol(
    ring: HostSplitRing, s1_cols: int = _M1, s1_take: int | None = None
) -> AbdlopQuadraticEval:
    # `s1_std` is re-derived rather than taken from the fixture: the point's
    # T_1 = η·√(m1·d) is a bound on ‖s1‖, so a wider Ajtai half masked at the
    # narrow point would reject its way to `exhausted` instead of proving.
    std = lnp_fixture.GAMMA * lnp_fixture.ETA * float(np.sqrt(s1_cols * ring.d))
    masking = lnp_fixture.masking(_scheme(ring, s1_cols=s1_cols), s1_std=std)
    return AbdlopQuadraticEval(
        AbdlopQuadraticMany(AbdlopQuadratic(masking)), _LAM, s1_take=s1_take
    )


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return lnp_fixture.transcript(b"lnp-quadratic-eval-test", tag)


def _constant(ring: HostSplitRing, value: np.ndarray) -> np.ndarray:
    """The one-element stack holding `value` in its constant coefficient and
    nothing else.

    Written through the array layout the way `GarbageMasking.sample` zeroes
    that slot, and for the same reason: `constant_coeff` reads it, and the
    module convention has no constructor that writes it."""
    out = ring.zeros(1)
    out[0, :, 0] = value
    return out


class _Instance:
    """One honest Fig. 8 statement: publics, witness, commitment, `N`
    relations that vanish on the lifted witness, and `M` evaluations whose
    constant coefficients vanish on it."""

    def __init__(
        self,
        seed: int,
        relations: int = _RELATIONS,
        s1_cols: int = _M1,
        s1_take: int | None = None,
    ) -> None:
        ring = lnp_fixture.ring()
        self.ring = ring
        self.protocol = _protocol(ring, s1_cols=s1_cols, s1_take=s1_take)
        self.s1_take = self.protocol.s1_take
        rng = np.random.default_rng(seed)
        self.rng = rng
        width = self.protocol.width

        self.a1 = ring.uniform_stack(rng, _ROWS, s1_cols)
        self.a2 = ring.uniform_stack(rng, _ROWS, _M2)
        self.b = ring.uniform_stack(rng, _ELL, _M2)
        self.bg = ring.uniform_stack(rng, _LAM, _M2)
        self.b_quad = ring.uniform_stack(rng, _M2)

        # Ternary witness halves, the shape the fixture's std was derived
        # against (α = ‖s1‖ ≤ √(m1·d)).
        self.s1 = rng.integers(-1, 2, (s1_cols, ring.d)).astype(np.int64)
        self.s2 = rng.integers(-1, 2, (_M2, ring.d)).astype(np.int64)
        s1_ring = ring.from_signed_stack(self.s1)
        s2_ring = ring.from_signed_stack(self.s2)
        self.message = ring.uniform_stack(rng, _ELL)

        # The scheme's own commit over the *narrow* BDLOP half — the layer
        # appends its own garbage, so the caller commits to `m` alone.
        commitment = _scheme(ring, _ELL, s1_cols).commit(
            self.a1, self.a2, self.b, s1_ring, s2_ring, self.message
        )
        self.t_a, self.t_b = commitment.t_a, commitment.t_b

        # The lift the caller's two families are written against — `m`, not
        # `m‖g`; the protocol appends the garbage itself.
        self.s = lift(ring, s1_ring[: self.s1_take], self.message)
        self.r2 = ring.uniform_stack(rng, relations, width, width)
        self.r1 = ring.uniform_stack(rng, relations, width)
        # `N = 0` is a real statement, so the empty stack is spelled the
        # way the module convention spells one rather than skipped.
        self.r0 = (
            np.stack(
                [
                    ring.neg(
                        evaluate(ring, self.r2[j], self.r1[j], ring.zeros(1), self.s)
                    )
                    for j in range(relations)
                ]
            )
            if relations
            else ring.zeros(0, 1)
        )
        self.e2 = ring.uniform_stack(rng, _EVALUATIONS, width, width)
        self.e1 = ring.uniform_stack(rng, _EVALUATIONS, width)
        self.e0 = np.stack(
            [self._vanishing(self.e2[j], self.e1[j]) for j in range(_EVALUATIONS)]
        )

    def _vanishing(self, e2: np.ndarray, e1: np.ndarray) -> np.ndarray:
        """`e0` chosen so `F̃(s) = 0` while `F(s)` itself stays nonzero —
        a constant polynomial, not the whole value negated."""
        ring = self.ring
        value = evaluate(ring, e2, e1, ring.zeros(1), self.s)
        return ring.neg(_constant(ring, ring.constant_coeff(value)[0]))

    def statement(self) -> dict[str, np.ndarray]:
        return dict(
            a1=self.a1,
            a2=self.a2,
            b=self.b,
            bg=self.bg,
            b_quad=self.b_quad,
            r2=self.r2,
            r1=self.r1,
            r0=self.r0,
            e2=self.e2,
            e1=self.e1,
            e0=self.e0,
        )

    def prove(
        self, tag: bytes = b"", **overrides: np.ndarray
    ) -> tuple[QuadraticEvalProof, ByteTranscript]:
        args = self.statement()
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
        self, proof: QuadraticEvalProof, tag: bytes = b"", **overrides: np.ndarray
    ) -> bool:
        args = self.statement()
        args.update(t_a=self.t_a, t_b=self.t_b)
        args.update(overrides)
        ok, _ = self.protocol.verify(proof=proof, transcript=_transcript(tag), **args)
        return ok


class QuadraticEvalCompletenessTest(absltest.TestCase):
    def test_an_honest_proof_verifies(self) -> None:
        instance = _Instance(1)
        proof, _ = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_a_carved_ajtai_half_proves_a_statement_about_its_prefix(self) -> None:
        """Fig. 10 appends the binary-decomposition vector `x` to the Ajtai
        half and writes its functions against `s1` alone, so the statement
        covers a prefix of a half that is wider than it.

        The carve is what says which prefix. Taking the whole half — what
        every layer up to Fig. 9 did, and what the code spelled directly —
        puts the caller's functions in the wrong columns of the inner
        protocol's lift, so this round-trip is the gate on the index map
        rather than on the algebra."""
        instance = _Instance(60, s1_cols=_M1 + 1, s1_take=_M1)
        self.assertEqual(instance.protocol.s1_take, _M1)
        self.assertEqual(instance.protocol.width, SIGMA_ORDER * (_M1 + _ELL))
        proof, _ = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_the_carve_cannot_exceed_the_half_it_carves(self) -> None:
        ring = lnp_fixture.ring()
        with self.assertRaisesRegex(ValueError, r"^eval:"):
            _protocol(ring, s1_cols=_M1, s1_take=_M1 + 1)

    def test_the_evaluations_do_not_vanish_as_ring_elements(self) -> None:
        """What separates this protocol from Fig. 7. If the suite's `F_j`
        happened to be zero on the witness, every assertion here would still
        pass while proving only the layer below's statement."""
        instance = _Instance(2)
        ring = instance.ring
        for j in range(_EVALUATIONS):
            value = evaluate(
                ring, instance.e2[j], instance.e1[j], instance.e0[j], instance.s
            )
            self.assertFalse(ring.constant_coeff(value).any())
            self.assertTrue(value.any())

    def test_no_relations_is_the_fig5_generalization(self) -> None:
        """`N = 0` — evaluations alone, which is Fig. 5's statement with
        quadratic functions in place of linear ones."""
        instance = _Instance(3, relations=0)
        proof, _ = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_distinct_transcripts_give_distinct_proofs(self) -> None:
        instance = _Instance(4)
        first, _ = instance.prove(b"one")
        second, _ = instance.prove(b"two")
        self.assertFalse(np.array_equal(first.quadratic.c, second.quadratic.c))
        self.assertTrue(instance.verify(first, b"one"))
        self.assertTrue(instance.verify(second, b"two"))

    def test_the_caller_width_excludes_the_garbage_the_protocol_appends(self) -> None:
        instance = _Instance(5)
        self.assertEqual(instance.protocol.width, SIGMA_ORDER * (_M1 + _ELL))
        self.assertEqual(
            instance.protocol.many.width,
            SIGMA_ORDER * (_M1 + _ELL + _LAM),
        )
        self.assertEqual(instance.protocol.ell, _ELL)


class QuadraticEvalLayoutTest(absltest.TestCase):
    """Where `g` sits in the lifted witness — eq. 38's own claim, which no
    prove/verify round-trip can check."""

    def test_the_garbage_interleaves_with_each_automorphism_copy(self) -> None:
        """`lift(s1, m‖g)` groups by automorphism copy, `[m‖g, σ(m‖g)]` —
        not by vector, `[m, σ(m), g, σ(g)]`. So the caller's coordinates
        land on `_positions` and `g` on `_garbage_slots`."""
        instance = _Instance(6)
        ring = instance.ring
        protocol = instance.protocol
        rng = instance.rng
        s1_ring = ring.from_signed_stack(instance.s1)
        g = ring.uniform_stack(rng, _LAM)

        wide = lift(ring, s1_ring, np.concatenate([instance.message, g]))
        self.assertEqual(len(wide), protocol.many.width)
        np.testing.assert_array_equal(wide[protocol._positions], instance.s)
        np.testing.assert_array_equal(wide[protocol._garbage_slots], g)

    def test_an_embedded_function_is_the_same_function(self) -> None:
        """`_embed`'s whole contract: re-indexing a quadratic into the wider
        lift must not change what it evaluates to. Checked against the ring
        directly, so it holds independently of the protocol that uses it."""
        instance = _Instance(7)
        ring = instance.ring
        protocol = instance.protocol
        rng = instance.rng
        g = ring.uniform_stack(rng, _LAM)
        wide_s = lift(
            ring,
            ring.from_signed_stack(instance.s1),
            np.concatenate([instance.message, g]),
        )

        wide2, wide1, wide0 = protocol._embed(instance.e2, instance.e1, instance.e0)
        for j in range(_EVALUATIONS):
            narrow = evaluate(
                ring, instance.e2[j], instance.e1[j], instance.e0[j], instance.s
            )
            embedded = evaluate(ring, wide2[j], wide1[j], wide0[j], wide_s)
            np.testing.assert_array_equal(narrow, embedded)

    def test_the_garbage_slots_are_the_first_copy_only(self) -> None:
        """eq. 38 reads `x^{(g)}_{2,1,i}` — the first automorphism copy's
        `g_i`, not `σ(g_i)`. λ slots, all inside the first copy's block."""
        instance = _Instance(8)
        slots = instance.protocol._garbage_slots
        first_copy_end = SIGMA_ORDER * _M1 + _ELL + _LAM
        self.assertLen(slots, _LAM)
        self.assertGreaterEqual(int(slots.min()), SIGMA_ORDER * _M1 + _ELL)
        self.assertLess(int(slots.max()), first_copy_end)


class QuadraticEvalSoundnessTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.instance = _Instance(9)
        cls.proof, _ = cls.instance.prove()

    def test_the_honest_proof_is_the_baseline(self) -> None:
        self.assertTrue(self.instance.verify(self.proof))

    def test_a_nonvanishing_constant_coefficient_is_rejected(self) -> None:
        """The statement this layer exists to make. `e0 = 0` leaves `F̃(s)`
        generically nonzero, so the honest `h` fails the `h̃ = 0` check."""
        ring = self.instance.ring
        false_e0 = ring.zeros(_EVALUATIONS, 1)
        proof, _ = self.instance.prove(e0=false_e0)
        self.assertTrue(ring.constant_coeff(proof.h).any())
        self.assertFalse(self.instance.verify(proof, e0=false_e0))

    def test_a_false_relation_is_rejected(self) -> None:
        """One of the `N` carried relations made false — the Fig. 7 half of
        the statement, which this layer must not have dropped."""
        false_r0 = lnp_fixture.bump(self.instance.r0, 0, 0, 0, 0)
        proof, _ = self.instance.prove(r0=false_r0)
        self.assertFalse(self.instance.verify(proof, r0=false_r0))

    def test_a_tampered_aggregate_is_rejected(self) -> None:
        for index in ((0, 0, 0), (1, 0, 5)):
            with self.subTest(index=index):
                tampered = dataclasses.replace(
                    self.proof, h=lnp_fixture.bump(self.proof.h, *index)
                )
                self.assertFalse(self.instance.verify(tampered))

    def test_a_tampered_garbage_commitment_is_rejected(self) -> None:
        tampered = dataclasses.replace(
            self.proof, t_g=lnp_fixture.bump(self.proof.t_g, 0, 0, 0)
        )
        self.assertFalse(self.instance.verify(tampered))

    def test_a_verifier_holding_a_different_evaluation_rejects(self) -> None:
        self.assertFalse(
            self.instance.verify(
                self.proof, e2=lnp_fixture.bump(self.instance.e2, 0, 0, 0, 0, 0)
            )
        )

    def test_a_verifier_holding_a_different_relation_rejects(self) -> None:
        self.assertFalse(
            self.instance.verify(
                self.proof, r1=lnp_fixture.bump(self.instance.r1, 0, 0, 0, 0)
            )
        )

    def test_a_tampered_commitment_is_rejected(self) -> None:
        self.assertFalse(
            self.instance.verify(
                self.proof, t_a=lnp_fixture.bump(self.instance.t_a, 0, 0, 0)
            )
        )


class QuadraticEvalWireTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.instance = _Instance(10)
        cls.proof, _ = cls.instance.prove()

    def test_a_non_proof_is_a_verdict(self) -> None:
        self.assertFalse(self.instance.verify(object()))

    def test_every_proof_field_is_gated(self) -> None:
        """Drives off `dataclasses.fields`, so a field added to
        `QuadraticEvalProof` without a gate in `_is_well_formed` fails here
        rather than reaching an `AttributeError` inside `verify` — which is
        what the composite field did one layer down."""
        for field in dataclasses.fields(QuadraticEvalProof):
            tampered = dataclasses.replace(self.proof, **{field.name: None})
            self.assertFalse(self.instance.verify(tampered), field.name)

    def test_an_out_of_range_residue_is_a_verdict(self) -> None:
        """`wire.is_stack`'s range half: a residue at or above the modulus is
        refused as a verdict, not raised out of a ring op."""
        for name in ("t_g", "h"):
            with self.subTest(field=name):
                broken = getattr(self.proof, name).copy()
                broken[0, 0, 0] = lnp_fixture.SPLIT_Q[0]
                tampered = dataclasses.replace(self.proof, **{name: broken})
                self.assertFalse(self.instance.verify(tampered))

    def test_a_malformed_statement_raises(self) -> None:
        ring = self.instance.ring
        width = self.instance.protocol.width
        for name, bad in (
            ("r1", ring.zeros(_RELATIONS + 1, width)),
            ("r0", ring.zeros(_RELATIONS, 2)),
            ("e1", ring.zeros(_EVALUATIONS, width + 1)),
            ("e2", ring.zeros(_EVALUATIONS, width, width + 1)),
        ):
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, f"eval: {name}"):
                    self.instance.verify(self.proof, **{name: bad})

    def test_a_statement_with_no_evaluations_raises(self) -> None:
        ring = self.instance.ring
        width = self.instance.protocol.width
        with self.assertRaisesRegex(ValueError, "at least one evaluation"):
            self.instance.verify(
                self.proof,
                e2=ring.zeros(0, width, width),
                e1=ring.zeros(0, width),
                e0=ring.zeros(0, 1),
            )

    def test_the_scheme_must_carry_the_garbage(self) -> None:
        ring = lnp_fixture.ring()
        many = AbdlopQuadraticMany(AbdlopQuadratic(lnp_fixture.masking(_scheme(ring))))
        with self.assertRaisesRegex(ValueError, "extended scheme"):
            AbdlopQuadraticEval(many, _ELL + _LAM + 1)
        with self.assertRaisesRegex(ValueError, "lam must be positive"):
            AbdlopQuadraticEval(many, 0)


if __name__ == "__main__":
    absltest.main()
