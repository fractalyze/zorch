# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π^(2) (Fig. 6) and Π_many^(2) (Fig. 7) — completeness, soundness, and
the wire contract.

The honest statement is built the only way a quadratic one can be without
solving for a witness: pick `R2` and `r1` freely, then *define*
`r0 := −(sᵀR2s + r1ᵀs)` so that `f(s) = 0` holds by construction. A test
that instead picked `r0` freely would be proving a false statement and
could only ever assert rejection.

Two tests here exist because a prove/verify round-trip is a weak gate on a
protocol whose two sides share an implementation: it passes just as
happily when both sides drop the quadratic term, or lift through the wrong
automorphism. So the quadratic form is tampered with directly, and the
§2.3 identity is pinned against exact integer arithmetic rather than
against the protocol.
"""

from __future__ import annotations

from dataclasses import fields, replace
from unittest import mock

import numpy as np
from absl.testing import absltest
from lattice_frx import norms
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp import masking as masking_module
from zorch.lnp.quadratic import (
    SIGMA_ORDER,
    AbdlopQuadratic,
    AbdlopQuadraticMany,
    QuadraticProof,
    _dot,
)
from zorch.lnp.testing import lnp_fixture

_ROWS = 2
_M1 = lnp_fixture.M1
_M2 = 3
_ELL = 1


def _ring() -> HostSplitRing:
    return lnp_fixture.ring()


def _scheme(ring: HostSplitRing) -> AbdlopCommitment:
    return AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=_M1,
        randomness_cols=_M2,
        messages=_ELL,
        beta1_inf=1,
        beta2_inf=1,
    )


def _quadratic(ring: HostSplitRing, **overrides: object) -> AbdlopQuadratic:
    return AbdlopQuadratic(lnp_fixture.masking(_scheme(ring), **overrides))


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return lnp_fixture.transcript(b"lnp-quadratic-test", tag)


class _Instance:
    """One honest Π^(2) statement: publics, witness, commitment, and a
    quadratic function that vanishes on the lifted witness."""

    def __init__(self, seed: int) -> None:
        self.ring = _ring()
        self.quadratic = _quadratic(self.ring)
        # What `prove`/`verify` dispatch on — Fig. 6 here, Fig. 7 in
        # `_ManyInstance`. `self.quadratic` stays the inner protocol, which
        # is what carries the lifted width and the lift itself.
        self.protocol: AbdlopQuadratic | AbdlopQuadraticMany = self.quadratic
        # The scheme the protocol actually masks against, not a second one
        # built to the same parameters.
        self.scheme = self.quadratic.scheme
        rng = np.random.default_rng(seed)
        self.rng = rng
        ring = self.ring
        n = self.quadratic.width

        self.a1 = ring.uniform_stack(rng, _ROWS, _M1)
        self.a2 = ring.uniform_stack(rng, _ROWS, _M2)
        self.b = ring.uniform_stack(rng, _ELL, _M2)
        self.b_quad = ring.uniform_stack(rng, _M2)

        # Ternary witness halves, the shape the fixture's std was derived
        # against (α = ‖s1‖ ≤ √(m1·d)).
        self.s1 = rng.integers(-1, 2, (_M1, ring.d)).astype(np.int64)
        self.s2 = rng.integers(-1, 2, (_M2, ring.d)).astype(np.int64)
        s1_ring = ring.from_signed_stack(self.s1)
        s2_ring = ring.from_signed_stack(self.s2)
        self.message = ring.uniform_stack(rng, _ELL)

        commitment = self.scheme.commit(
            self.a1, self.a2, self.b, s1_ring, s2_ring, self.message
        )
        self.t_a, self.t_b = commitment.t_a, commitment.t_b

        # f(s) = sᵀR2s + r1ᵀs + r0 with r0 solved so the statement is true.
        self.r2 = ring.uniform_stack(rng, n, n)
        self.r1 = ring.uniform_stack(rng, n)
        self.lifted = self.quadratic._lift(s1_ring, self.message)
        self.r0 = ring.neg(self._vanishing_at(self.r2, self.r1))

    def _vanishing_at(self, square: np.ndarray, linear: np.ndarray) -> np.ndarray:
        """`sᵀ·square·s + linearᵀ·s` on the lifted witness — the value a
        relation's constant term must cancel for `f(s) = 0` to hold."""
        ring, s = self.ring, self.lifted
        return ring.add(_dot(ring, s, ring.matvec(square, s)), _dot(ring, linear, s))

    def _statement(self) -> dict[str, np.ndarray]:
        return dict(
            a1=self.a1,
            a2=self.a2,
            b=self.b,
            b_quad=self.b_quad,
            r2=self.r2,
            r1=self.r1,
            r0=self.r0,
        )

    def prove(
        self,
        tag: bytes = b"",
        protocol: AbdlopQuadratic | None = None,
        **overrides: np.ndarray,
    ) -> tuple[QuadraticProof, ByteTranscript]:
        args = self._statement()
        args.update(overrides)
        return (protocol or self.protocol).prove(
            s1=self.s1,
            s2=self.s2,
            message=self.message,
            rng=self.rng,
            transcript=_transcript(tag),
            **args,
        )

    def verify(
        self, proof: QuadraticProof, tag: bytes = b"", **overrides: np.ndarray
    ) -> bool:
        args = self._statement()
        args.update(t_a=self.t_a, t_b=self.t_b)
        args.update(overrides)
        ok, _ = self.protocol.verify(proof=proof, transcript=_transcript(tag), **args)
        return ok


class QuadraticCompletenessTest(absltest.TestCase):

    def test_an_honest_proof_verifies(self) -> None:
        instance = _Instance(1)
        proof, _ = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_the_lifted_width_is_k_times_the_message_and_witness(self) -> None:
        """§4's `n = k(m1 + ℓ)` — the statement is written against the
        witness *and* its automorphism images, not the witness alone."""
        instance = _Instance(2)
        self.assertEqual(instance.quadratic.width, SIGMA_ORDER * (_M1 + _ELL))
        self.assertEqual(instance.r2.shape[:2], (SIGMA_ORDER * (_M1 + _ELL),) * 2)

    def test_the_statement_really_is_quadratic(self) -> None:
        """A guard on the fixture, not on the protocol: if `R2` contributed
        nothing, every assertion below would still pass while proving only
        the linear layer's statement."""
        instance = _Instance(3)
        ring, s = instance.ring, instance.lifted
        quad = _dot(ring, s, ring.matvec(instance.r2, s))
        self.assertTrue(quad.any())

    def test_sigma_inner_product_is_the_squared_norm(self) -> None:
        """§2.3, the identity this whole layer exists to reach: the constant
        coefficient of `σ₋₁(a)·a` is `⟨a, a⟩ = ‖a‖²` over the integers.

        Pinned here rather than assumed because it is what makes a
        *quadratic* statement a *norm* statement — and because it is the one
        property a wrong choice of automorphism would break silently. The
        protocol's round-trip cannot see that: prover and verifier lift the
        same way, so they would agree on the wrong `σ` too."""
        ring = _ring()
        rng = np.random.default_rng(99)
        coeffs = rng.integers(-3, 4, (ring.d,)).astype(np.int64)
        a = ring.from_signed(coeffs)
        sigma_a = ring.galois(a, -1)
        product = ring.mul(sigma_a, a)
        want = norms.l2_squared(coeffs) % lnp_fixture.SPLIT_Q[0]
        self.assertEqual(int(ring.constant_coeff(product[None])[0, 0]), want)

    def test_distinct_transcripts_give_distinct_proofs(self) -> None:
        instance = _Instance(4)
        p1, _ = instance.prove(tag=b"one")
        p2, _ = instance.prove(tag=b"two")
        self.assertFalse(np.array_equal(p1.c, p2.c))
        self.assertTrue(instance.verify(p1, tag=b"one"))
        self.assertTrue(instance.verify(p2, tag=b"two"))
        # Binding to the transcript, not merely differing on it.
        self.assertFalse(instance.verify(p1, tag=b"two"))

    def test_an_exhausted_budget_raises(self) -> None:
        """fail_prob = 0.5 shrinks the budget to a handful of attempts;
        forcing every rejection then exhausts it — the loop must raise,
        not spin."""
        instance = _Instance(5)
        starved = _quadratic(instance.ring, fail_prob=0.5)
        with mock.patch.object(masking_module, "_rej1", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "prove"):
                instance.prove(protocol=starved)


class QuadraticSoundnessTest(absltest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.instance = _Instance(10)
        self.proof, _ = self.instance.prove()

    def test_the_honest_proof_is_the_baseline(self) -> None:
        self.assertTrue(self.instance.verify(self.proof))

    def test_a_false_statement_is_rejected(self) -> None:
        """`r0` moved by one breaks `f(s) = 0` — the statement the layer
        exists to prove, and the check that fails if `R2`'s contribution
        were dropped from either side."""
        bumped = lnp_fixture.bump(self.instance.r0, 0, 0, 0)
        self.assertFalse(self.instance.verify(self.proof, r0=bumped))

    def test_tampering_the_quadratic_form_is_rejected(self) -> None:
        """`R2` and `r1` must be load-bearing on *both* sides. A round-trip
        alone cannot show that: a layer that dropped the quadratic term from
        the prover and the verifier alike would still verify its own
        proofs, and would be proving the linear statement instead."""
        instance = self.instance
        self.assertFalse(
            instance.verify(self.proof, r2=lnp_fixture.bump(instance.r2, 0, 0, 0, 0))
        )
        self.assertFalse(
            instance.verify(self.proof, r1=lnp_fixture.bump(instance.r1, 0, 0, 0))
        )

    def test_a_tampered_garbage_commitment_is_rejected(self) -> None:
        """`t` is the one first-round message on the wire, so it is the one
        the verifier cannot recompute — and therefore the one a prover
        could try to move after seeing `c`."""
        tampered = replace(self.proof, t=lnp_fixture.bump(self.proof.t, 0, 0, 0))
        self.assertFalse(self.instance.verify(tampered))

    def test_tampered_responses_are_rejected(self) -> None:
        for field in ("c", "z1", "z2"):
            moved = getattr(self.proof, field).copy()
            moved[(0,) * moved.ndim] += 1
            tampered = replace(self.proof, **{field: moved})
            self.assertFalse(self.instance.verify(tampered), field)

    def test_an_over_norm_response_is_rejected(self) -> None:
        blown = self.proof.z1.copy()
        blown[0, 0] += int(10 * lnp_fixture.STD * self.instance.ring.d)
        tampered = replace(self.proof, z1=blown)
        self.assertFalse(self.instance.verify(tampered))

    def test_a_tampered_commitment_is_rejected(self) -> None:
        instance = self.instance
        self.assertFalse(
            instance.verify(self.proof, t_a=lnp_fixture.bump(instance.t_a, 0, 0, 0))
        )
        self.assertFalse(
            instance.verify(self.proof, t_b=lnp_fixture.bump(instance.t_b, 0, 0, 0))
        )


class QuadraticWireTest(absltest.TestCase):
    """The untrusted-proof rule: a malformed proof is a verdict, a malformed
    statement is the caller's bug. See `zorch/lnp/wire.py`."""

    def setUp(self) -> None:
        super().setUp()
        self.instance = _Instance(20)
        self.proof, _ = self.instance.prove()

    def test_every_proof_field_is_gated(self) -> None:
        """Drives off `dataclasses.fields`, so a field added to
        `QuadraticProof` without a gate in `_is_well_formed` fails here
        rather than reaching an AttributeError inside `verify`."""
        for field in fields(QuadraticProof):
            tampered = replace(self.proof, **{field.name: None})  # type: ignore[arg-type]
            self.assertFalse(self.instance.verify(tampered), field.name)

    def test_a_non_proof_is_a_verdict(self) -> None:
        self.assertFalse(self.instance.verify(object()))  # type: ignore[arg-type]

    def test_an_out_of_range_residue_in_t_is_a_verdict(self) -> None:
        """`wire.is_stack`'s range half: a residue at or above the modulus
        is refused as a verdict, not raised out of a ring op."""
        bad = self.proof.t.copy()
        bad[0, 0, 0] = lnp_fixture.SPLIT_Q[0]
        tampered = replace(self.proof, t=bad)
        self.assertFalse(self.instance.verify(tampered))

    def test_a_malformed_statement_raises(self) -> None:
        bad = np.zeros((2, 3))
        instance, proof = self.instance, self.proof
        for name, call in (
            ("r2", lambda: instance.verify(proof, r2=bad)),
            ("r1", lambda: instance.verify(proof, r1=bad)),
            ("r0", lambda: instance.verify(proof, r0=bad)),
            ("b_quad", lambda: instance.verify(proof, b_quad=bad)),
        ):
            with (
                self.subTest(name),
                self.assertRaisesRegex(ValueError, f"quadratic: {name}"),
            ):
                call()


class _ManyInstance(_Instance):
    """N honest quadratic relations over one commitment.

    The publics, witness and commitment are the single-relation fixture's —
    Fig. 7 changes only the statement — so this replaces `r2`/`r1`/`r0`
    with a relation-indexed stack and wraps the protocol. Each relation is
    built the same way the base one is, constant term solved so it holds on
    its own, because a statement that is only true in aggregate would pass
    Fig. 7 while failing what it claims."""

    def __init__(self, seed: int, relations: int = 3) -> None:
        super().__init__(seed)
        self.relations = relations
        ring, rng, n = self.ring, self.rng, self.quadratic.width
        squares, linears, constants = [], [], []
        for _ in range(relations):
            square = ring.uniform_stack(rng, n, n)
            linear = ring.uniform_stack(rng, n)
            squares.append(square)
            linears.append(linear)
            constants.append(ring.neg(self._vanishing_at(square, linear)))
        self.r2 = np.stack(squares)
        self.r1 = np.stack(linears)
        self.r0 = np.stack(constants)
        self.many = AbdlopQuadraticMany(self.quadratic)
        self.protocol = self.many


class QuadraticManyTest(absltest.TestCase):
    """Fig. 7: N relations aggregated into one, so the proof commits a
    single garbage term however many relations the statement carries."""

    def test_an_honest_proof_over_many_relations_verifies(self) -> None:
        instance = _ManyInstance(30)
        proof, _ = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_many_relations_cost_one_garbage_commitment(self) -> None:
        """The layer exists to make proof size independent of N, and it can
        because the aggregation challenge is Fiat-Shamir output rather than
        a message. Six relations must therefore produce the same
        single-element commitment that one does."""
        instance = _ManyInstance(31, relations=6)
        proof, _ = instance.prove()
        self.assertIsInstance(proof, QuadraticProof)
        self.assertEqual(proof.t.shape[0], 1)
        self.assertTrue(instance.verify(proof))

    def test_one_false_relation_among_many_is_rejected(self) -> None:
        """Aggregation must not let one bad relation hide behind the rest,
        and it must hold at every position — an implementation that dropped
        a row would still pass a check made only at the first."""
        instance = _ManyInstance(32)
        proof, _ = instance.prove()
        for j in range(instance.relations):
            bumped = lnp_fixture.bump(instance.r0, j, 0, 0, 0)
            self.assertFalse(instance.verify(proof, r0=bumped), j)

    def test_the_aggregation_challenge_spans_the_whole_ring(self) -> None:
        """§4.2 draws from R_q, not Z_q, and the `q1^{-d/2}` soundness bound
        rests on that: a scalar drawn into the constant coefficient would
        leave the other `d − 1` at zero and cost the degree factor."""
        instance = _ManyInstance(33)
        ring = instance.ring
        _, mu = instance.many._mu(_transcript(), instance.relations)
        self.assertEqual(mu.shape, (instance.relations, 1, ring.d))
        for j in range(instance.relations):
            self.assertGreater(int(np.count_nonzero(mu[j])), 1)

    def test_a_statement_with_no_relations_raises(self) -> None:
        instance = _ManyInstance(34)
        proof, _ = instance.prove()
        with self.assertRaisesRegex(ValueError, "at least one"):
            instance.verify(proof, r2=instance.r2[:0])


if __name__ == "__main__":
    absltest.main()
