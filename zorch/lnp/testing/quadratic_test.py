# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π^(2) (Fig. 6) — completeness, soundness, and the wire contract.

The honest statement is built the only way a quadratic one can be without
solving for a witness: pick `R2` and `r1` freely, then *define*
`r0 := −(sᵀR2s + r1ᵀs)` so that `f(s) = 0` holds by construction. A test
that instead picked `r0` freely would be proving a false statement and
could only ever assert rejection.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp import masking as masking_module
from zorch.lnp.quadratic import SIGMA_ORDER, AbdlopQuadratic, QuadraticProof
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
        self.scheme = _scheme(self.ring)
        self.protocol = _quadratic(self.ring)
        rng = np.random.default_rng(seed)
        self.rng = rng
        ring = self.ring
        n = self.protocol.width
        self.assert_width = n

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

        self.t_a = ring.add(
            ring.matvec(self.a1, s1_ring), ring.matvec(self.a2, s2_ring)
        )
        self.t_b = ring.add(ring.matvec(self.b, s2_ring), self.message)

        # f(s) = sᵀR2s + r1ᵀs + r0 with r0 solved so the statement is true.
        self.r2 = ring.uniform_stack(rng, n, n)
        self.r1 = ring.uniform_stack(rng, n)
        s = self.protocol._lift(s1_ring, self.message)
        quad = ring.matvec(s[None, :], ring.matvec(self.r2, s))
        linear = ring.matvec(self.r1[None, :], s)
        self.r0 = ring.neg(ring.add(quad, linear))

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
        self.assertEqual(instance.assert_width, SIGMA_ORDER * (_M1 + _ELL))
        self.assertEqual(instance.r2.shape[:2], (SIGMA_ORDER * (_M1 + _ELL),) * 2)

    def test_the_statement_really_is_quadratic(self) -> None:
        """A guard on the fixture, not on the protocol: if `R2` contributed
        nothing, every assertion below would still pass while proving only
        the linear layer's statement."""
        instance = _Instance(3)
        ring = instance.ring
        s = instance.protocol._lift(
            ring.from_signed_stack(instance.s1), instance.message
        )
        quad = ring.matvec(s[None, :], ring.matvec(instance.r2, s))
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
        sigma_a = ring.galois(a, 2 * ring.d - 1)
        product = ring.mul(sigma_a, a)
        want = int((coeffs.astype(object) ** 2).sum()) % lnp_fixture.SPLIT_Q[0]
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
        with absltest.mock.patch.object(masking_module, "_rej1", return_value=False):
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
        import dataclasses

        tampered = dataclasses.replace(
            self.proof, t=lnp_fixture.bump(self.proof.t, 0, 0, 0)
        )
        self.assertFalse(self.instance.verify(tampered))

    def test_tampered_responses_are_rejected(self) -> None:
        import dataclasses

        for field in ("c", "z1", "z2"):
            moved = getattr(self.proof, field).copy()
            moved[(0,) * moved.ndim] += 1
            tampered = dataclasses.replace(self.proof, **{field: moved})
            self.assertFalse(self.instance.verify(tampered), field)

    def test_an_over_norm_response_is_rejected(self) -> None:
        import dataclasses

        blown = self.proof.z1.copy()
        blown[0, 0] += int(10 * lnp_fixture.STD * self.instance.ring.d)
        tampered = dataclasses.replace(self.proof, z1=blown)
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
        import dataclasses

        for field in dataclasses.fields(QuadraticProof):
            tampered = dataclasses.replace(self.proof, **{field.name: None})  # type: ignore[arg-type]
            self.assertFalse(self.instance.verify(tampered), field.name)

    def test_a_non_proof_is_a_verdict(self) -> None:
        self.assertFalse(self.instance.verify(object()))  # type: ignore[arg-type]

    def test_an_out_of_range_residue_in_t_is_a_verdict(self) -> None:
        """`wire.is_stack`'s range half: a residue at or above the modulus
        is refused as a verdict, not raised out of a ring op."""
        import dataclasses

        bad = self.proof.t.copy()
        bad[0, 0, 0] = lnp_fixture.SPLIT_Q[0]
        tampered = dataclasses.replace(self.proof, t=bad)
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


if __name__ == "__main__":
    absltest.main()
