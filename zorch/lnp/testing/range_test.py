# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The approximate range proof (Fig. 9) — completeness, soundness, the
statement it hands Π_eval^(2), and the message layout it is indexed
against.

The layout tests carry the weight here, for the reason they do one layer
down: **a prove/verify round-trip cannot see a wrong layout.** Both sides
build `F_i`, `G_j` and `f` from the same `_witness_positions`,
`_mask_positions` and `_sign_position`, so a proof over a permutation of
the witness verifies exactly as happily as the honest one. The suite
therefore pins those positions against the *ring* — against `lift` of a
known stack — and pins each function against what it is supposed to
evaluate to, rather than against the fact that a proof went through.

The functions are pinned three ways because they fail three ways:

- `F_i` must vanish in its constant coefficient and *not* as a ring
  element. A suite that only checked the former would not notice a
  statement that had become a relation.
- `G_j` must vanish exactly when the sign is a constant polynomial. Their
  whole job is to license `F_i`'s constant coefficient factoring through
  `b`, and nothing else in the protocol notices if they are dropped.
- `f` must vanish as a ring element and only for `b = ±1`.

One more is pinned for a subtler reason: `⃗z` reaching the transcript. Both
sides build the statement from one method, so removing that absorb is
symmetric and leaves every completeness and soundness test green — the
binding has to be asserted directly or not at all.

Every structural claim below was verified by mutation — each of them, made
alone, turns this suite red.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from absl.testing import absltest
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp import masking as masking_module
from zorch.lnp.eval import AbdlopQuadraticEval
from zorch.lnp.masking import BimodalMasking, L2Bound, LinfBound
from zorch.lnp.quadratic import (
    AbdlopQuadratic,
    AbdlopQuadraticMany,
    Publics,
    evaluate,
    lift,
    lift_slots,
)
from zorch.lnp.range import (
    AffineImage,
    ApproximateRange,
    RangeProof,
    _joint_budget,
)
from zorch.lnp.testing import lnp_fixture

_ROWS = 2
_M1 = lnp_fixture.M1
_M2 = 3
_ELL = 1
_LAM = 2
# The witness whose norm is the statement: `(s1, m)`, ternary.
_WITNESS_COLS = _M1 + _ELL


def _scheme(
    ring: HostSplitRing, mask_cols: int, s1_cols: int = _M1, legs: int = 1
) -> AbdlopCommitment:
    """The **twice-extended** scheme: its BDLOP half carries `ℓ + legs·(256/d
    + 1) + λ` messages, because `m‖y…‖b…‖g` is what the innermost protocol
    opens — every leg's mask and sign, and the garbage of the layer below."""
    return AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=s1_cols,
        randomness_cols=_M2,
        messages=_ELL + legs * (mask_cols + 1) + _LAM,
        beta1_inf=1,
        beta2_inf=1,
    )


def _maskings(
    ring: HostSplitRing,
    legs: int,
    bounds: Sequence[L2Bound | LinfBound] | None = None,
    witness_cols: int = _WITNESS_COLS,
) -> tuple[BimodalMasking, ...]:
    """One bimodal point per leg, each carrying the gate it is verified
    under.

    `bounds` is the count when it is given — `legs` only means "that many
    default-gated legs" — so the number lives in one place. `bounds`
    defaults to Fig. 9's ℓ2 gate throughout; Fig. 10's own pairing is one of
    each, which is what it exists to spell.

    Nothing here sizes the Gaussian sampler for the composition:
    `ApproximateRange` re-resolves each leg through `for_attempts`, so the
    joint-budget rule has one owner and this fixture is not a second
    spelling of it."""
    if bounds is None:
        bounds = (L2Bound(),) * legs
    return tuple(
        lnp_fixture.bimodal(ring, witness_cols, bound=bound) for bound in bounds
    )


def _protocol(
    ring: HostSplitRing,
    maskings: Sequence[BimodalMasking],
    s1_cols: int = _M1,
    s1_take: int | None = None,
    images: Sequence[AffineImage | None] = (),
) -> ApproximateRange:
    scheme = _scheme(ring, maskings[0].mask_cols, s1_cols, len(maskings))
    # Re-derived for the same reason `quadratic_eval_test` re-derives it: the
    # point's T_1 bounds ‖s1‖ over `_M1` columns, and a wider Ajtai half
    # masked there rejects its way to `exhausted`.
    inner = lnp_fixture.masking(scheme, s1_std=lnp_fixture.s1_std(ring, s1_cols))
    evaluation = AbdlopQuadraticEval(
        AbdlopQuadraticMany(AbdlopQuadratic(inner)), _LAM, s1_take=s1_take
    )
    return ApproximateRange(evaluation, *maskings, images=images)


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return lnp_fixture.transcript(b"lnp-range-test", tag)


class _Instance:
    """One honest Fig. 9 statement: publics, a short ternary witness, its
    commitment, and the protocol over both."""

    def __init__(
        self,
        seed: int,
        s1_cols: int = _M1,
        s1_take: int | None = None,
        legs: int = 1,
        bounds: Sequence[L2Bound | LinfBound] | None = None,
        images: Sequence[AffineImage | None] = (),
        witness_cols: int = _WITNESS_COLS,
    ) -> None:
        ring = lnp_fixture.ring()
        self.ring = ring
        self.maskings = _maskings(ring, legs, bounds, witness_cols)
        legs = len(self.maskings)
        self.masking = self.maskings[0]
        self.protocol = _protocol(ring, self.maskings, s1_cols, s1_take, images)
        # Fig. 9 is the one-leg composition, and every layout assertion below
        # is about the leg rather than about the layer that schedules it.
        self.leg = self.protocol.legs[0]
        self.scheme = self.protocol.scheme
        mask_cols = self.masking.mask_cols
        rng = np.random.default_rng(seed)
        self.rng = rng

        self.a1 = ring.uniform_stack(rng, _ROWS, s1_cols)
        self.a2 = ring.uniform_stack(rng, _ROWS, _M2)
        self.b = ring.uniform_stack(rng, _ELL, _M2)
        # Every leg's mask rows, then every leg's sign row — the composing
        # layer's order, which is Fig. 10's `(m, y^(d), y^(e), b^(d), b^(e))`.
        self.b_mask = ring.uniform_stack(rng, mask_cols * legs, _M2)
        self.b_sign = ring.uniform_stack(rng, legs, _M2)
        self.bg = ring.uniform_stack(rng, _LAM, _M2)
        self.b_quad = ring.uniform_stack(rng, _M2)
        # The whole BDLOP matrix in the order the message is concatenated
        # in — `m`, this layer's mask and sign, the garbage below. Each
        # layer carves its own rows back out; see `Publics`.
        self.publics = Publics(
            a1=self.a1,
            a2=self.a2,
            blocks=np.concatenate([self.b, self.b_mask, self.b_sign, self.bg]),
            b_quad=self.b_quad,
        )

        # Both halves ternary, and the *message* too — unlike every other
        # suite here, where `m` is uniform. `m` is half of the vector whose
        # norm this protocol bounds, so a uniform one is not a witness.
        self.s1 = rng.integers(-1, 2, (s1_cols, ring.d)).astype(np.int64)
        self.s2 = rng.integers(-1, 2, (_M2, ring.d)).astype(np.int64)
        self.message = rng.integers(-1, 2, (_ELL, ring.d)).astype(np.int64)

        s1_ring = ring.from_signed_stack(self.s1)
        s2_ring = ring.from_signed_stack(self.s2)
        self.t_a = ring.add(
            ring.matvec(self.a1, s1_ring), ring.matvec(self.a2, s2_ring)
        )
        self.t_b = ring.add(
            ring.matvec(self.b, s2_ring), ring.from_signed_stack(self.message)
        )

    def statement(self) -> dict[str, Any]:
        """The publics both sides take, by name — the shape every sibling
        suite uses, so a test that varies one states only that."""
        return dict(publics=self.publics)

    def prove(self, tag: bytes = b"", **overrides: Any) -> RangeProof:
        args = self.statement() | dict(s1=self.s1, s2=self.s2, message=self.message)
        proof, _ = self.protocol.prove(
            **(args | overrides), rng=self.rng, transcript=_transcript(tag)
        )
        return proof

    def verify(
        self,
        proof: RangeProof,
        tag: bytes = b"",
        protocol: ApproximateRange | None = None,
        **overrides: Any,
    ) -> bool:
        args = self.statement() | dict(t_a=self.t_a, t_b=self.t_b)
        ok, _ = (protocol or self.protocol).verify(
            **(args | overrides), proof=proof, transcript=_transcript(tag)
        )
        return ok

    def challenge(
        self, t_mask: np.ndarray | None = None, t_sign: np.ndarray | None = None
    ) -> np.ndarray:
        """`R` off a chosen pair of first-round commitments, defaulting to
        zero stacks — the derivation the statement tests index against.

        Absorb then squeeze, the two steps the leg keeps apart so Fig. 10
        can bind every commitment before drawing any challenge."""
        ring = self.ring
        leg = self.leg
        observed = leg.observe(
            _transcript(),
            ring.zeros(self.masking.mask_cols) if t_mask is None else t_mask,
            ring.zeros(1) if t_sign is None else t_sign,
        )
        _, projection = leg.challenge(observed)
        return projection

    def caller_family(self, seed: int) -> tuple[Any, Any]:
        """One relation and one evaluation of the caller's own, both true on
        this witness — the `(f_i, F_i)` Fig. 10 takes alongside its legs.

        Supported on the witness columns alone. The mask and sign columns
        hold values the prover draws per attempt, so a constant term solved
        against them here would be solved against a `y` that no run will
        use — and a caller could not have written such a function down
        either, having never seen the mask.

        The relation vanishes as a ring element and the evaluation only in
        its constant coefficient, which is the distinction the two families
        exist for."""
        ring = self.ring
        rng = np.random.default_rng(seed)
        width = self.protocol.evaluation.width
        positions = self.leg._witness_positions
        s = self.lifted()

        def supported() -> tuple[np.ndarray, np.ndarray]:
            quadratic = ring.zeros(1, width, width)
            quadratic[0, positions[:, None], positions[None, :]] = ring.uniform_stack(
                rng, len(positions), len(positions)
            )
            linear = ring.zeros(1, width)
            linear[0, positions] = ring.uniform_stack(rng, len(positions))
            return quadratic, linear

        r2, r1 = supported()
        r0 = np.stack([ring.neg(evaluate(ring, r2[0], r1[0], ring.zeros(1), s))])

        e2, e1 = supported()
        value = evaluate(ring, e2[0], e1[0], ring.zeros(1), s)
        e0 = np.stack([lnp_fixture.vanishing_constant(ring, value)])
        return (r2, r1, r0), (e2, e1, e0)

    def tamper(self, proof: RangeProof, **field: np.ndarray) -> RangeProof:
        """A copy of `proof` with one field of its only leg replaced — the
        `dataclasses.replace` idiom, one level deeper now that the wire
        carries a leg per projection."""
        return dataclasses.replace(
            proof, legs=(dataclasses.replace(proof.legs[0], **field),)
        )

    def lifted(self, sign: int = 1, y: np.ndarray | None = None) -> np.ndarray:
        """The eval layer's lift of `(s1, m‖y‖b)` at a chosen sign and mask.

        `y` defaults to zero because the layout tests are about *positions*,
        and a zero there makes a misread one visible. It is a parameter
        rather than a constant so the statement tests use this one
        construction too — the message layout is the thing they pin, and a
        second hand-written copy of it could drift from this one unseen."""
        ring = self.ring
        sign_row = np.zeros((1, ring.d), dtype=np.int64)
        sign_row[0, 0] = sign
        if y is None:
            y = np.zeros((self.masking.mask_cols, ring.d), dtype=np.int64)
        return lift(
            ring,
            # Carved to what the eval layer's statement covers, since this is
            # that layer's lift and the positions index into it.
            ring.from_signed_stack(self.s1[: self.protocol.evaluation.s1_take]),
            ring.from_signed_stack(np.concatenate([self.message, y, sign_row])),
        )


class RangeCompletenessTest(absltest.TestCase):
    def test_an_honest_proof_verifies(self) -> None:
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                instance = _Instance(seed)
                self.assertTrue(instance.verify(instance.prove()))

    def test_a_carved_ajtai_half_bounds_only_the_prefix(self) -> None:
        """The Ajtai carve reaches this layer too.

        `ApproximateRange` derives its positions from the *eval layer's*
        lift, so when that layer covers a prefix of a wider Ajtai half —
        which is what Fig. 10 does, committing to `(s1, x)` — this layer's
        `s1_span`, witness positions and chunk count have to follow it.
        Reading the scheme's own width instead puts the mask and sign
        positions past the end of the lift the statement is written
        against."""
        instance = _Instance(50, s1_cols=_M1 + 1, s1_take=_M1)
        self.assertEqual(instance.protocol.evaluation.s1_take, _M1)
        proof = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_the_revealed_projection_is_within_the_norm_bound(self) -> None:
        instance = _Instance(4)
        z = instance.prove().legs[0].z
        self.assertEqual(z.shape, (instance.masking.mask_cols, instance.ring.d))
        self.assertTrue(instance.masking.within_bounds(z))

    def test_a_proof_does_not_verify_under_a_different_transcript(self) -> None:
        instance = _Instance(5)
        self.assertFalse(instance.verify(instance.prove(b"one"), b"two"))


class RangeSoundnessTest(absltest.TestCase):
    def test_a_tampered_projection_is_rejected(self) -> None:
        instance = _Instance(6)
        proof = instance.prove()
        z = proof.legs[0].z.copy()
        z[0, 0] += 1
        self.assertFalse(instance.verify(instance.tamper(proof, z=z)))

    def test_a_tampered_mask_commitment_is_rejected(self) -> None:
        instance = _Instance(7)
        proof = instance.prove()
        bumped = lnp_fixture.bump(proof.legs[0].t_mask, 0, 0, 0)
        self.assertFalse(instance.verify(instance.tamper(proof, t_mask=bumped)))

    def test_a_tampered_sign_commitment_is_rejected(self) -> None:
        instance = _Instance(8)
        proof = instance.prove()
        bumped = lnp_fixture.bump(proof.legs[0].t_sign, 0, 0, 0)
        self.assertFalse(instance.verify(instance.tamper(proof, t_sign=bumped)))

    def test_a_tampered_inner_proof_is_rejected(self) -> None:
        instance = _Instance(9)
        proof = instance.prove()
        inner = proof.evaluation
        tampered = dataclasses.replace(inner, h=lnp_fixture.bump(inner.h, 0, 0, 1))
        self.assertFalse(
            instance.verify(dataclasses.replace(proof, evaluation=tampered))
        )

    def test_a_proof_against_a_different_commitment_is_rejected(self) -> None:
        instance = _Instance(10)
        proof = instance.prove()
        self.assertFalse(
            instance.verify(proof, t_a=lnp_fixture.bump(instance.t_a, 0, 0, 0))
        )
        self.assertFalse(
            instance.verify(proof, t_b=lnp_fixture.bump(instance.t_b, 0, 0, 0))
        )

    def test_the_norm_gate_is_consulted(self) -> None:
        """An honest proof against a verifier whose bound is too tight.

        The only difference is the gate's `accept_t`, which touches
        nothing the transcript sees — so the inner proof still verifies and
        the verdict is the norm gate's alone. Tampering `z` upward would not
        pin this: that changes the statement and the inner proof rejects it
        first."""
        instance = _Instance(11)
        proof = instance.prove()
        strict = _protocol(
            instance.ring,
            [
                lnp_fixture.bimodal(
                    instance.ring, _WITNESS_COLS, bound=L2Bound(accept_t=1e-6)
                )
            ],
        )
        self.assertFalse(instance.verify(proof, protocol=strict))


class RangeStatementTest(absltest.TestCase):
    """What this layer hands Π_eval^(2), checked against the ring rather
    than against a proof going through."""

    def test_the_projection_functions_vanish_in_their_constant_coefficient(
        self,
    ) -> None:
        instance = _Instance(12)
        ring = instance.ring
        protocol = instance.protocol
        sign, y = instance.masking.draw(np.random.default_rng(99))
        projection = instance.challenge()
        witness = np.concatenate([instance.s1, instance.message])
        z = sign * (projection @ witness.reshape(-1)) + y.reshape(-1)

        s = instance.lifted(sign, y)
        e2, e1, e0 = instance.leg._evaluations(projection, z.reshape(y.shape))
        values = np.concatenate(
            [evaluate(ring, e2[i], e1[i], e0[i], s) for i in range(len(e2))]
        )
        self.assertFalse(ring.constant_coeff(values).any())
        # ...and they are evaluations, not relations: a suite that solved
        # them to zero as ring elements could not tell Fig. 8 from Fig. 7.
        self.assertTrue(values.any())

    def test_the_coefficient_functions_detect_a_non_integer_sign(self) -> None:
        """`G_j` is what licenses `F_i`'s constant coefficient factoring
        through `b`. Nothing else in the protocol notices a `b` that is not
        a constant polynomial, so this is pinned directly."""
        instance = _Instance(13)
        ring = instance.ring
        protocol = instance.protocol
        zeros = np.zeros((instance.masking.mask_cols, ring.d), dtype=np.int64)
        _, e1, _ = instance.leg._evaluations(instance.challenge(), zeros)
        coefficient = e1[instance.masking.projection :]
        empty = ring.zeros(protocol.evaluation.width, protocol.evaluation.width)

        def coefficients_of(s: np.ndarray) -> np.ndarray:
            values = np.concatenate(
                [evaluate(ring, empty, row, ring.zeros(1), s) for row in coefficient]
            )
            return ring.constant_coeff(values)

        for sign in (1, -1):
            with self.subTest(sign=sign):
                self.assertFalse(coefficients_of(instance.lifted(sign)).any())

        # A sign carrying an `X^1` term is exactly what `G_1` is there for.
        crooked = instance.lifted(1)
        crooked[instance.leg._sign_position] = ring.add(
            crooked[instance.leg._sign_position],
            ring.from_signed([0, 1] + [0] * (ring.d - 2)),
        )
        self.assertTrue(coefficients_of(crooked).any())

    def test_the_revealed_projection_is_bound_to_the_transcript(self) -> None:
        """`⃗z` reaches the transcript on its own, not only through `e0`.

        A round-trip cannot see this. The statement already carries `⃗z` into
        the inner proof's constants, and both sides build that statement from
        the same method — so dropping the absorb entirely leaves every
        completeness and soundness test green. Pinned directly because the
        paper's message ordering puts `⃗z` on the wire before Π runs, and a
        later change that stopped routing `⃗z` through `e0` would otherwise
        lose the binding silently."""
        instance = _Instance(24)
        ring = instance.ring
        projection = instance.challenge()
        z = np.zeros((instance.masking.mask_cols, ring.d), dtype=np.int64)
        moved = z.copy()
        moved[0, 0] = 1
        base, _ = instance.protocol._statement(_transcript(), [projection], [z])
        other, _ = instance.protocol._statement(_transcript(), [projection], [moved])
        self.assertNotEqual(base.sample_scalar(16)[1], other.sample_scalar(16)[1])

    def test_the_sign_relation_vanishes_only_for_a_sign(self) -> None:
        instance = _Instance(14)
        ring = instance.ring
        protocol = instance.protocol
        r2, r1, r0 = instance.leg._relation()
        for sign in (1, -1):
            with self.subTest(sign=sign):
                value = evaluate(ring, r2[0], r1[0], r0[0], instance.lifted(sign))
                self.assertFalse(value.any())
        # `b = 2` satisfies integrality but not `b² = 1`.
        two = instance.lifted(1)
        two[instance.leg._sign_position] = ring.from_signed([2] + [0] * (ring.d - 1))
        self.assertTrue(evaluate(ring, r2[0], r1[0], r0[0], two).any())


class CallerFamilyTest(absltest.TestCase):
    """A caller's own `(f_i, F_i)` riding along with the range statement.

    Fig. 10's whole shape is "these relations *and* these norm bounds, over
    one commitment", so the range layer has to carry a caller's families
    down to the single `Pi_eval^(2)` rather than making the caller run a
    second proof against the same witness. `AbdlopQuadraticEval` already
    allows `N = 0` relations, which is why the legs' own statement did not
    need this before.
    """

    def test_a_caller_family_proves_beside_the_range_statement(self) -> None:
        instance = _Instance(36)
        relations, evaluations = instance.caller_family(80)
        proof = instance.prove(relations=relations, evaluations=evaluations)
        self.assertTrue(
            instance.verify(proof, relations=relations, evaluations=evaluations)
        )

    def test_either_family_may_be_given_alone(self) -> None:
        """`N = 0` relations is legal one layer down and `M >= 1` is not, so
        the two halves are not symmetric — but the legs always contribute an
        evaluation, so a caller supplying only relations is fine too."""
        instance = _Instance(37)
        relations, evaluations = instance.caller_family(81)
        for name, extra in (
            ("relations only", dict(relations=relations)),
            ("evaluations only", dict(evaluations=evaluations)),
        ):
            with self.subTest(name):
                proof = instance.prove(**extra)
                self.assertTrue(instance.verify(proof, **extra))

    def test_a_false_caller_relation_is_rejected(self) -> None:
        """The caller's relations are proved, not carried: `f(s) != 0` has
        to fail even though every range obligation still holds.

        Rejected by the verifier, not refused by the prover. Fig. 6 never
        evaluates `f(s)` — it commits the cross terms and answers a
        challenge — so a false relation costs the prover nothing and shows
        up as the verifier's recomputed `v` differing by `c^2 f(s)`, which
        is exactly what the replayed challenge catches."""
        instance = _Instance(38)
        (r2, r1, r0), _ = instance.caller_family(82)
        false = (r2, r1, lnp_fixture.bump(r0, 0, 0, 0, 0))
        proof = instance.prove(relations=false)
        self.assertFalse(instance.verify(proof, relations=false))

    def test_a_false_caller_evaluation_is_rejected(self) -> None:
        """`F(s)` with a nonzero constant coefficient. Its aggregate lands
        in `h`, whose `h~ = 0` check is the verifier's."""
        instance = _Instance(39)
        _, (e2, e1, e0) = instance.caller_family(83)
        false = lnp_fixture.bump(e0, 0, 0, 0, 0)
        proof = instance.prove(evaluations=(e2, e1, false))
        self.assertFalse(instance.verify(proof, evaluations=(e2, e1, false)))

    def test_the_caller_family_is_part_of_the_statement(self) -> None:
        """A verifier given different families checks a different claim, so
        the inner proof does not replay. Pinned both ways round: dropping
        them and swapping them each fail."""
        instance = _Instance(40)
        relations, evaluations = instance.caller_family(84)
        proof = instance.prove(relations=relations, evaluations=evaluations)
        self.assertFalse(instance.verify(proof))
        self.assertFalse(
            instance.verify(
                proof,
                relations=relations,
                evaluations=instance.caller_family(85)[1],
            )
        )

    def test_the_caller_family_comes_first(self) -> None:
        """Fig. 10 indexes `f_i` and `F_i` from one and appends the legs'
        obligations, and the two sides only agree because both build the
        statement through `_families`. Pinned against the block itself: the
        caller's relation is row 0, and the leg's `b^2 - 1` follows it."""
        instance = _Instance(41)
        relations, evaluations = instance.caller_family(86)
        # One leg's `(relations, evaluations)` pair, in the per-leg sequence
        # `_families` now takes — it stacks caller and legs in one pass.
        merged = instance.protocol._families(
            [
                (
                    instance.leg._sign_relation,
                    instance.leg._evaluations(
                        instance.challenge(),
                        np.zeros(
                            (instance.masking.mask_cols, instance.ring.d), np.int64
                        ),
                    ),
                )
            ],
            relations,
            evaluations,
        )
        r2, r1, r0, e2, e1, e0 = merged
        np.testing.assert_array_equal(r2[0], relations[0][0])
        np.testing.assert_array_equal(r2[1], instance.leg._sign_relation[0][0])
        np.testing.assert_array_equal(e2[0], evaluations[0][0])
        self.assertLen(r2, 2)
        self.assertLen(e2, 1 + instance.masking.projection + instance.ring.d - 1)


class RangeLayoutTest(absltest.TestCase):
    """The positions, pinned against `lift` of a known stack.

    A round-trip cannot see these: prover and verifier index the lift
    identically, so a wrong position proves a statement about a permutation
    of the witness and both sides agree on it."""

    def test_the_witness_positions_select_the_identity_copies_of_s1_and_m(
        self,
    ) -> None:
        instance = _Instance(15)
        ring = instance.ring
        s = instance.lifted()
        want = np.concatenate(
            [
                ring.from_signed_stack(instance.s1),
                ring.from_signed_stack(instance.message),
            ]
        )
        got = s[instance.leg._witness_positions]
        np.testing.assert_array_equal(got, want)

    def test_the_positions_follow_a_carved_ajtai_half(self) -> None:
        """The carve moves every position after `s1`, not just `s1`'s own.

        `lift` orbits each half as a whole, so a narrower Ajtai half shrinks
        `s1_span` and shifts the message copies — and with them the mask and
        the sign — left by twice the columns dropped. Pinned against `lift`
        rather than a round-trip for this suite's standing reason: both
        sides index identically, so a wrong carve proves a statement about a
        permutation and both agree on it."""
        instance = _Instance(17, s1_cols=_M1 + 1, s1_take=_M1)
        ring = instance.ring
        protocol = instance.protocol
        _, y = instance.masking.draw(np.random.default_rng(4))
        s = instance.lifted(-1, y)

        sign_row = np.zeros((1, ring.d), dtype=np.int64)
        sign_row[0, 0] = -1
        np.testing.assert_array_equal(
            s[instance.leg._witness_positions],
            np.concatenate(
                [
                    ring.from_signed_stack(instance.s1[:_M1]),
                    ring.from_signed_stack(instance.message),
                ]
            ),
        )
        np.testing.assert_array_equal(
            s[instance.leg._mask_positions], ring.from_signed_stack(y)
        )
        np.testing.assert_array_equal(
            s[instance.leg._sign_position], ring.from_signed_stack(sign_row)[0]
        )

    def test_the_mask_and_sign_interleave_into_the_first_automorphism_copy(
        self,
    ) -> None:
        """`lift` orbits the message stack as a whole, so `y` and `b` land
        inside each copy rather than after the message's copies. An append
        layout would put them past `σ(m)`, which is a different vector."""
        instance = _Instance(16)
        ring = instance.ring
        protocol = instance.protocol
        _, y = instance.masking.draw(np.random.default_rng(3))
        s = instance.lifted(-1, y)
        sign_row = np.zeros((1, ring.d), dtype=np.int64)
        sign_row[0, 0] = -1
        np.testing.assert_array_equal(
            s[instance.leg._mask_positions], ring.from_signed_stack(y)
        )
        np.testing.assert_array_equal(
            s[instance.leg._sign_position], ring.from_signed_stack(sign_row)[0]
        )

    def test_the_layer_carves_its_share_off_the_extended_scheme(self) -> None:
        instance = _Instance(17)
        protocol = instance.protocol
        self.assertEqual(protocol.ell, _ELL)
        self.assertEqual(protocol.evaluation.ell, _ELL + instance.masking.mask_cols + 1)
        self.assertEqual(instance.scheme.messages, protocol.evaluation.ell + _LAM)

    def test_a_scheme_without_room_for_the_mask_is_refused(self) -> None:
        ring = lnp_fixture.ring()
        masking = lnp_fixture.bimodal(ring, _WITNESS_COLS)
        scheme = AbdlopCommitment(
            ring,
            rows=_ROWS,
            s1_cols=_M1,
            randomness_cols=_M2,
            messages=_ELL + _LAM,
            beta1_inf=1,
            beta2_inf=1,
        )
        evaluation = AbdlopQuadraticEval(
            AbdlopQuadraticMany(AbdlopQuadratic(lnp_fixture.masking(scheme))), _LAM
        )
        with self.assertRaisesRegex(ValueError, "extended scheme"):
            ApproximateRange(evaluation, masking)

    def test_a_range_proof_with_no_leg_is_refused(self) -> None:
        """`*maskings` makes the empty composition spellable, and a proof of
        nothing would otherwise reach `AbdlopQuadraticEval` with an empty
        evaluation family and fail there."""
        instance = _Instance(26)
        with self.assertRaisesRegex(ValueError, "at least one projection leg"):
            ApproximateRange(instance.protocol.evaluation)


class ProjectionLegTest(absltest.TestCase):
    """The round, exercised without the proof it used to be fused to.

    Fig. 9 fuses them and Fig. 10 cannot: it gates *both* legs' Rej0 and
    then runs one Π_eval^(2), so no leg can own the inner call. These pin
    that the round stands alone — a leg commits, takes its challenge and
    responds — and that its verdict is a value rather than control flow.
    """

    def test_the_projection_round_runs_without_the_inner_proof(self) -> None:
        instance = _Instance(27)
        ring, leg = instance.ring, instance.leg
        rng = np.random.default_rng(5)
        s2_ring = ring.from_signed_stack(instance.s2)

        draw = leg.draw(leg.randomness(instance.publics, s2_ring), rng)
        observed = leg.observe(_transcript(), draw.t_mask, draw.t_sign)
        _, projection = leg.challenge(observed)
        flat = np.concatenate([instance.s1, instance.message]).reshape(-1)
        z = leg.respond(draw, projection, flat, rng)

        # Not a rejection at this parameter point (Rej0 accepts ~99.7% of
        # the time), and the response is what the figure says it is.
        self.assertIsNotNone(z)
        assert z is not None
        np.testing.assert_array_equal(
            z.reshape(-1), draw.sign * (projection @ flat) + draw.y.reshape(-1)
        )
        self.assertTrue(leg.masking.within_bounds(z))

    def test_the_commitments_are_the_mask_and_sign_over_their_own_rows(
        self,
    ) -> None:
        """`t_mask = B2·s2 + y` and `t_sign = b1·s2 + b`, against the rows
        the leg was told are its own — the arithmetic that goes wrong when a
        leg reads the wrong slice of the assembled matrix."""
        instance = _Instance(28)
        ring, leg = instance.ring, instance.leg
        s2_ring = ring.from_signed_stack(instance.s2)
        draw = leg.draw(
            leg.randomness(instance.publics, s2_ring), np.random.default_rng(6)
        )
        np.testing.assert_array_equal(
            draw.t_mask,
            ring.add(ring.matvec(instance.b_mask, s2_ring), draw.y_ring),
        )
        np.testing.assert_array_equal(
            draw.t_sign,
            ring.add(ring.matvec(instance.b_sign, s2_ring), draw.sign_ring),
        )

    def test_a_rejected_response_is_a_value_not_a_raise(self) -> None:
        """The composing layer abandons an attempt when *any* leg rejects,
        so a leg reports its verdict rather than retrying or raising. Forced
        with a centre far outside the mask's range, where Rej0 cannot
        accept."""
        instance = _Instance(29)
        leg = instance.leg
        rng = np.random.default_rng(7)
        s2_ring = instance.ring.from_signed_stack(instance.s2)
        draw = leg.draw(leg.randomness(instance.publics, s2_ring), rng)
        _, projection = leg.challenge(
            leg.observe(_transcript(), draw.t_mask, draw.t_sign)
        )
        huge = np.full(projection.shape[1], 10**6, dtype=np.int64)
        self.assertIsNone(leg.respond(draw, projection, huge, rng))


class TwoLegTest(absltest.TestCase):
    """Fig. 10's *shape* over one commitment: two projection legs, one
    Π_eval^(2).

    Both legs bound the same `(s1, m)` here, which is redundant as a
    statement and exactly right as a structural fixture — it is the
    composition Fig. 10 needs, with the ℓ∞ gate, the binary/exact-ℓ2
    machinery and the φ/Ψ packing that make the two legs actually *differ*
    still to come. What is pinned is that neither leg can see the other's
    columns and that one inner proof covers both.
    """

    def test_two_legs_share_no_column(self) -> None:
        instance = _Instance(30, legs=2)
        first, second = instance.protocol.legs
        occupied = [
            set(leg._mask_positions.tolist()) | {int(leg._sign_position)}
            for leg in (first, second)
        ]
        self.assertEqual(len(occupied[0]), instance.masking.mask_cols + 1)
        self.assertFalse(occupied[0] & occupied[1])
        # ...and neither reaches into the caller's `m`, which they share.
        witness = set(first._witness_positions.tolist())
        self.assertFalse(witness & (occupied[0] | occupied[1]))
        np.testing.assert_array_equal(
            first._witness_positions, second._witness_positions
        )

    def test_the_legs_read_their_own_rows_of_the_assembled_matrix(self) -> None:
        """The slot a leg was told indexes `blocks` and the lift alike. A
        round-trip cannot see this: both sides carve identically, so a leg
        reading its sibling's rows commits to the wrong thing and agrees
        with itself about it."""
        instance = _Instance(31, legs=2)
        mask_cols = instance.masking.mask_cols
        for index, leg in enumerate(instance.protocol.legs):
            with self.subTest(leg=index):
                np.testing.assert_array_equal(
                    leg._mask_rows(instance.publics),
                    instance.b_mask[index * mask_cols : (index + 1) * mask_cols],
                )
                np.testing.assert_array_equal(
                    leg._sign_rows(instance.publics),
                    instance.b_sign[index : index + 1],
                )

    def test_the_legs_share_the_slots_the_message_is_built_in(self) -> None:
        """Fig. 10's message is `(m, y^(d), y^(e), b^(d), b^(e))` — every
        mask, then every sign — and the slots have to be that same order or
        the commitment opens to a permuted message."""
        instance = _Instance(32, legs=2)
        first, second = instance.protocol.legs
        mask_cols = instance.masking.mask_cols
        self.assertEqual(first.mask_slot, _ELL)
        self.assertEqual(second.mask_slot, _ELL + mask_cols)
        self.assertEqual(first.sign_slot, _ELL + 2 * mask_cols)
        self.assertEqual(second.sign_slot, _ELL + 2 * mask_cols + 1)

    def test_a_projection_is_bound_to_every_leg(self) -> None:
        """Fig. 10 sends *all* four first-round commitments before *any* `R`,
        so leg 0's challenge has to move when leg 1's mask commitment does.

        No round-trip can see this: a composition that bound only each leg's
        own pair would have both sides derive the same wrong `R` and verify
        happily. It is the entire reason the leg keeps `observe` and
        `challenge` as two steps instead of the one `_challenge` Fig. 9
        needed, so it is asserted against the derivation directly."""
        instance = _Instance(50, legs=2)
        ring = instance.ring
        mask, sign = ring.zeros(instance.masking.mask_cols), ring.zeros(1)
        # Only the *second* leg's mask commitment differs between the two.
        _, base = instance.protocol._round(_transcript(), [(mask, sign), (mask, sign)])
        _, moved = instance.protocol._round(
            _transcript(),
            [(mask, sign), (lnp_fixture.bump(mask, 0, 0, 0), sign)],
        )
        self.assertFalse(np.array_equal(base[0], moved[0]))

    def test_two_legs_prove_and_verify_under_one_inner_proof(self) -> None:
        instance = _Instance(33, legs=2)
        proof = instance.prove()
        self.assertLen(proof.legs, 2)
        # One Π_eval^(2), not two: the wire carries a single inner proof
        # however many legs contributed statements to it.
        self.assertTrue(instance.verify(proof))

    def test_tampering_either_leg_is_rejected(self) -> None:
        """Each leg's `⃗z` reaches the shared statement, so neither can be
        moved without the single inner proof failing to replay."""
        instance = _Instance(34, legs=2)
        proof = instance.prove()
        for index in range(2):
            with self.subTest(leg=index):
                legs = list(proof.legs)
                z = legs[index].z.copy()
                z[0, 0] += 1
                legs[index] = dataclasses.replace(legs[index], z=z)
                self.assertFalse(
                    instance.verify(dataclasses.replace(proof, legs=tuple(legs)))
                )

    def test_one_leg_rejecting_abandons_the_whole_attempt(self) -> None:
        """Fig. 10 continues only if *every* Rej0 accepts.

        A leg that kept its accepted `⃗z` while a sibling redrew would be
        revealing a projection under a challenge the redrawn transcript no
        longer produces. Rej0 accepts ~99.7% of the time at this parameter
        point, so no round-trip ever reaches the mixed verdict — one leg is
        stubbed to reject once, and what is asserted is that the *other*
        leg's accepted response is thrown away with it."""
        instance = _Instance(35, legs=2)
        first, second = instance.protocol.legs
        accepted: list[np.ndarray] = []
        honest_first, honest_second = first.respond, second.respond
        pending_rejection = [True]

        def record(*args: Any) -> np.ndarray | None:
            z = honest_first(*args)
            if z is not None:
                accepted.append(z)
            return z

        def reject_once(*args: Any) -> np.ndarray | None:
            z = honest_second(*args)
            return None if pending_rejection and pending_rejection.pop() else z

        first.respond = record  # type: ignore[assignment]
        second.respond = reject_once  # type: ignore[assignment]
        proof = instance.prove()

        self.assertTrue(instance.verify(proof))
        # Twice, because the rejected attempt cost it its first response...
        self.assertLen(accepted, 2)
        # ...and that response is nowhere on the wire.
        self.assertFalse(any(np.array_equal(accepted[0], leg.z) for leg in proof.legs))

    def test_the_composition_sizes_each_leg_for_the_joint_budget(self) -> None:
        """Two legs reject together often enough to need more attempts than
        either's own rate implies, and a leg's Gaussian sampler picks its
        tier from that number. The caller states a parameter point and the
        composition sizes it — nothing is refused, and no caller has to
        derive the joint budget to build a leg."""
        ring = lnp_fixture.ring()
        alone = lnp_fixture.bimodal(ring, _WITNESS_COLS)
        joint = _joint_budget([alone, alone])
        self.assertGreater(joint, alone.attempts)

        instance = _Instance(48, legs=2)
        self.assertEqual(instance.protocol.attempts, joint)
        for leg in instance.protocol.legs:
            self.assertEqual(leg.masking.attempts, joint)
        # ...and the point the caller handed in is left as it was.
        self.assertEqual(instance.maskings[0].attempts, alone.attempts)

    def test_a_single_leg_reuses_its_masking_untouched(self) -> None:
        """Fig. 9's budget already is the joint one, so `for_attempts` has
        nothing to re-resolve and hands back the same object."""
        instance = _Instance(49)
        self.assertIs(instance.protocol.legs[0].masking, instance.maskings[0])

    def test_a_shorter_loop_than_the_point_implies_is_refused(self) -> None:
        """`for_attempts` only ever sizes *up*. Down would mean a loop that
        gives up before `fail_prob`, which is the guarantee the number is
        derived to make."""
        masking = lnp_fixture.bimodal(lnp_fixture.ring(), _WITNESS_COLS)
        with self.assertRaisesRegex(ValueError, "is below the"):
            masking.for_attempts(masking.attempts - 1)


class RangeChallengeTest(absltest.TestCase):
    def test_the_projection_matrix_is_centred_binomial(self) -> None:
        """`Bin_1`: `−1` and `+1` a quarter each, `0` the other half."""
        instance = _Instance(18)
        ring = instance.ring
        projection = instance.challenge()
        self.assertEqual(
            projection.shape,
            (instance.masking.projection, (_M1 + _ELL) * ring.d),
        )
        self.assertEqual(set(np.unique(projection).tolist()), {-1, 0, 1})
        share = np.mean(projection == 0)
        self.assertAlmostEqual(share, 0.5, delta=0.02)

    def test_the_projection_matrix_is_bound_to_both_commitments(self) -> None:
        instance = _Instance(19)
        ring = instance.ring
        zeros_mask = ring.zeros(instance.masking.mask_cols)
        zeros_sign = ring.zeros(1)
        base = instance.challenge()
        moved_mask = instance.challenge(t_mask=lnp_fixture.bump(zeros_mask, 0, 0, 0))
        moved_sign = instance.challenge(t_sign=lnp_fixture.bump(zeros_sign, 0, 0, 0))
        self.assertFalse(np.array_equal(base, moved_mask))
        self.assertFalse(np.array_equal(base, moved_sign))


class RangeWireTest(absltest.TestCase):
    def test_every_proof_field_is_gated(self) -> None:
        """Proof fields are a verdict, never a raise — `zorch/lnp/wire.py`.

        Two levels now: the leg owns its three, the composition owns the
        sequence holding them. The count case is the one a `strict=True` zip
        would otherwise turn into a `ValueError` out of `verify`."""
        instance = _Instance(20)
        proof = instance.prove()
        ring = instance.ring
        leg = proof.legs[0]
        malformed: tuple[tuple[str, Any], ...] = (
            ("t_mask", ring.zeros(instance.masking.mask_cols + 1)),
            ("t_sign", ring.zeros(2)),
            ("z", np.zeros((instance.masking.mask_cols + 1, ring.d), np.int64)),
            ("z", leg.z.astype(np.float64)),
        )
        for field, value in malformed:
            with self.subTest(field=field):
                self.assertFalse(
                    instance.verify(instance.tamper(proof, **{field: value}))
                )

        for name, replacement in (
            ("evaluation", dict(evaluation=None)),
            ("legs is not a tuple", dict(legs=None)),
            ("legs is the wrong length", dict(legs=(leg, leg))),
            ("a leg is not a LegMessage", dict(legs=(object(),))),
        ):
            with self.subTest(name):
                self.assertFalse(
                    instance.verify(dataclasses.replace(proof, **replacement))
                )

    def test_a_foreign_proof_object_is_gated(self) -> None:
        instance = _Instance(21)
        self.assertFalse(instance.verify(object()))  # type: ignore[arg-type]

    def test_a_malformed_statement_raises(self) -> None:
        """Statement fields are the caller's bug and raise, so a parameter
        mistake cannot become a silently always-false verifier.

        A row short of the assembled matrix rather than a wrong `b_mask`:
        the legs' rows are carved out of `blocks` now, so the whole matrix
        is what a caller can get wrong and `Publics` is what gates it."""
        instance = _Instance(22)
        proof = instance.prove()
        wrong = instance.publics.blocks[:-1]
        with self.assertRaisesRegex(ValueError, "publics: blocks"):
            instance.verify(
                proof, publics=dataclasses.replace(instance.publics, blocks=wrong)
            )

    def test_a_malformed_witness_raises(self) -> None:
        instance = _Instance(23)
        wrong = np.zeros((_ELL + 1, instance.ring.d), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "range.prove: message"):
            instance.prove(message=wrong)


class NormGateTest(absltest.TestCase):
    """Fig. 10's two verifier gates.

    Its legs are identical up to this one thing: the same `D_s^{256/d}`
    draw, the same `s = γ·√337·α` derived from an **ℓ2** bound on the
    projected vector, the same Rej0 — and then one is accepted in ℓ∞ at
    `14·s` and the other in ℓ2 at `t·√256·s`. So the gate is a parameter,
    and these pin that it is the right one and that it bites.
    """

    def _masking(self, **overrides: Any) -> BimodalMasking:
        return lnp_fixture.bimodal(lnp_fixture.ring(), _WITNESS_COLS, **overrides)

    def test_each_gate_is_the_expression_the_paper_writes(self) -> None:
        """`t·√256·s` and `14·s`, floored.

        Pinned as numbers because the gate is the only place the range
        statement's slack is enforced: one that silently widened would still
        pass every completeness and soundness round-trip in this file, since
        an honest reveal sits far inside either bound."""
        l2 = self._masking()
        linf = self._masking(bound=LinfBound())
        std = l2.mask_std
        self.assertEqual(l2._limit, math.floor((1.64**2) * l2.projection * std**2))
        self.assertEqual(linf._limit, math.floor(14 * std))

    def test_neither_gate_implies_the_other(self) -> None:
        """Which is why a leg is told its gate rather than inferring one.

        At `t = 1.64` and 256 rows the ℓ2 bound is `26.24·s` and the ℓ∞ one
        `14·s`, so a single coordinate at `20·s` clears ℓ2 and fails ℓ∞,
        while a vector flat at `13·s` clears ℓ∞ and fails ℓ2 by a factor of
        the dimension."""
        l2 = self._masking()
        linf = self._masking(bound=LinfBound())
        std = l2.mask_std
        shape = (l2.mask_cols, lnp_fixture.D)

        spike = np.zeros(shape, dtype=np.int64)
        spike[0, 0] = round(20 * std)
        self.assertTrue(l2.within_bounds(spike))
        self.assertFalse(linf.within_bounds(spike))

        flat = np.full(shape, round(13 * std), dtype=np.int64)
        self.assertTrue(linf.within_bounds(flat))
        self.assertFalse(l2.within_bounds(flat))

    def test_an_honest_reveal_clears_the_linf_tail(self) -> None:
        """Completeness of the ℓ∞ gate, and the reason it is a tail bound at
        all: Rej0's *accepted* output is `D_s^256` centred at zero (Lemma
        2.14-3), not the bimodal mixture the prover drew from, so `14·s` is
        a tail an honest run clears rather than a slack it might spend."""
        linf = self._masking(bound=LinfBound())
        rng = np.random.default_rng(11)
        for attempt in range(20):
            with self.subTest(attempt=attempt):
                self.assertTrue(linf.within_bounds(linf.draw(rng)[1]))

    def test_the_proven_bound_is_not_the_gate(self) -> None:
        """`resolve` gives the gate on `⃗z`; `proven_norm` gives what Lemma
        2.9 then concludes about the *projected vector*, `‖⃗s‖ ≤
        2·√(256/26)·t·s`. Two different numbers, and §5.2's wraparound
        conditions want the second.

        Pinned as a value because those conditions are slack by orders of
        magnitude at any sane point, so a wrong number still satisfies them
        — which is how this accessor's first caller came to pass the
        projection *dimension*, a count, and see a green test."""
        masking = self._masking()
        self.assertEqual(
            masking.proven_norm(),
            math.ceil(
                2.0 * math.sqrt(masking.projection / 26.0) * 1.64 * masking.mask_std
            ),
        )
        # It is neither the gate on `⃗z` nor the dimension — the two values
        # the formula is most easily confused with.
        self.assertNotEqual(masking.proven_norm(), masking._limit)
        self.assertNotEqual(masking.proven_norm(), masking.projection)

    def test_the_linf_leg_refuses_to_state_an_l2_bound(self) -> None:
        """§5.2 does draw a conclusion for the ℓ∞ leg — `‖e⃗‖_∞ ≤ 24·s` —
        but in the other norm. Returning it where an ℓ2 bound is expected
        would satisfy a wraparound condition nothing established."""
        masking = self._masking(bound=LinfBound())
        with self.assertRaisesRegex(ValueError, "not the ℓ2 one"):
            masking.proven_norm()

    def test_accept_t_is_gated_where_it_means_something(self) -> None:
        """It moved off the masking with this slice: `t` is Prop. 5.1's ℓ2
        coefficient, so a masking that carried one its gate ignored would be
        a parameter nobody could act on."""
        with self.assertRaisesRegex(ValueError, "accept_t must be positive"):
            L2Bound(accept_t=0.0)


class LinfLegTest(absltest.TestCase):
    """A leg verified in ℓ∞, and Fig. 10's own pairing of one gate of each.

    The ℓ∞ leg proves Equation (52), `‖D_i s − u_i‖_∞ ≤ β_i^(d)` — the
    approximate bound §5.2 says is proved "with the technique from Figure 9
    with the ℓ∞-norm instead". Everything below is that: Fig. 9's machinery,
    a different acceptance norm.
    """

    def test_an_linf_gated_leg_proves_and_verifies(self) -> None:
        instance = _Instance(42, bounds=[LinfBound()])
        proof = instance.prove()
        self.assertTrue(instance.verify(proof))
        self.assertTrue(instance.masking.within_bounds(proof.legs[0].z))

    def test_the_gate_never_touches_the_prover(self) -> None:
        """`bound` is the verifier's alone: the draw, Rej0, the transcript
        and therefore the whole wire are identical under either. That is
        what lets Fig. 10 give its two legs different gates without giving
        them different provers — and it means an ℓ2-proved proof verifies
        under an ℓ∞ verifier at the same `s`, and back."""
        l2 = _Instance(43)
        linf = _Instance(43, bounds=[LinfBound()])
        proof = l2.prove()

        np.testing.assert_array_equal(proof.legs[0].z, linf.prove().legs[0].z)
        self.assertTrue(l2.verify(proof, protocol=linf.protocol))
        self.assertTrue(linf.verify(linf.prove(), protocol=l2.protocol))

    def test_the_linf_gate_is_consulted(self) -> None:
        """An honest proof against a verifier whose ℓ∞ bound is too tight.

        The gate has no slack knob of its own — 14 is a tail bound, not a
        coefficient — so the too-tight verifier is one built at a smaller
        `s`. Nothing the transcript sees changes, so the inner proof still
        verifies and the verdict is the gate's alone."""
        instance = _Instance(44, bounds=[LinfBound()])
        proof = instance.prove()
        strict = _protocol(
            instance.ring,
            [
                lnp_fixture.bimodal(
                    instance.ring,
                    _WITNESS_COLS,
                    bound=LinfBound(),
                    mask_std=instance.masking.mask_std / 1000.0,
                )
            ],
        )
        self.assertFalse(instance.verify(proof, protocol=strict))

    def test_figure_10_pairs_one_gate_of_each_norm(self) -> None:
        """The composition Fig. 10 actually runs: the approximate-ℓ∞ leg and
        the exact-ℓ2 leg over one commitment and one `Pi_eval^(2)`, each
        accepted under its own norm."""
        instance = _Instance(45, legs=2, bounds=[LinfBound(), L2Bound()])
        first, second = instance.protocol.legs
        self.assertIsInstance(first.masking.bound, LinfBound)
        self.assertIsInstance(second.masking.bound, L2Bound)

        proof = instance.prove()
        self.assertTrue(instance.verify(proof))
        for leg, message in zip(instance.protocol.legs, proof.legs):
            self.assertTrue(leg.masking.within_bounds(message.z))

    def test_each_leg_is_gated_in_its_own_norm(self) -> None:
        """Not both in whichever one the first leg carries. A reveal that
        only the ℓ2 leg's bound admits has to be refused when it lands on
        the ℓ∞ leg, and the composition checks each leg against its own."""
        instance = _Instance(46, legs=2, bounds=[LinfBound(), L2Bound()])
        linf_leg, l2_leg = instance.protocol.legs
        spike = np.zeros((instance.masking.mask_cols, lnp_fixture.D), np.int64)
        spike[0, 0] = round(20 * instance.masking.mask_std)
        self.assertFalse(linf_leg.masking.within_bounds(spike))
        self.assertTrue(l2_leg.masking.within_bounds(spike))

    def test_the_composition_consults_every_leg_and_not_just_the_first(
        self,
    ) -> None:
        """Through `verify`, which is where it matters.

        The test above asserts on the legs directly and a composition that
        gated all of them by `legs[0]` would still pass it — as would every
        honest round-trip here, since an honest reveal clears both norms
        comfortably. So the verdict has to turn on a leg *other than the
        first*: only the second leg's bound is made impossibly tight, and
        nothing the transcript sees changes, so a rejection can only be that
        leg's gate being consulted."""
        instance = _Instance(47, legs=2, bounds=[LinfBound(), L2Bound()])
        proof = instance.prove()
        self.assertTrue(instance.verify(proof))

        tight = lnp_fixture.bimodal(
            instance.ring, _WITNESS_COLS, bound=L2Bound(accept_t=1e-6)
        )
        strict = _protocol(instance.ring, [instance.maskings[0], tight])
        self.assertFalse(instance.verify(proof, protocol=strict))


class BimodalMaskingTest(absltest.TestCase):
    def test_the_projection_must_tile_the_ring_degree(self) -> None:
        ring = lnp_fixture.ring()
        with self.assertRaisesRegex(ValueError, "multiple of the ring degree"):
            BimodalMasking(ring, mask_std=1.0, rep0=1.5, projection=ring.d + 1)

    def test_the_parameters_are_gated(self) -> None:
        ring = lnp_fixture.ring()
        for kwargs, pattern in (
            (dict(projection=0), "projection must be positive"),
            (dict(mask_std=0.0), "mask_std must be positive"),
            (dict(rep0=1.0), "rep0 must exceed 1"),
            (dict(fail_prob=1.0), "fail_prob must be in"),
        ):
            with self.subTest(**kwargs):
                params: dict[str, Any] = (
                    dict(projection=ring.d, mask_std=1.0, rep0=1.5) | kwargs
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    BimodalMasking(ring, **params)

    def test_a_rate_far_from_its_lemma_value_is_refused(self) -> None:
        ring = lnp_fixture.ring()
        with self.assertRaisesRegex(ValueError, "Lemma 2.14-3"):
            BimodalMasking(ring, mask_std=1.0, rep0=1e9, projection=ring.d)

    def test_a_composition_cannot_resize_past_the_safety_limit(self) -> None:
        """`for_attempts` applies the constructor's `_MAX_ATTEMPTS` gate.

        Composition is what makes the limit reachable from legs that each
        pass it: the joint budget is `∏ 1/rep0_i`, so three legs at
        `rep0 = 100` each need ~8.8e3 attempts alone and ~8.9e7 together.
        Without the gate `ApproximateRange` schedules that loop instead of
        refusing the parameter point, which is a hang — and `_MAX_ATTEMPTS`
        exists precisely to turn "a rate far from its Lemma 2.14-3 value"
        into an error rather than an unbounded run."""
        ring = lnp_fixture.ring()
        legs = [
            BimodalMasking(ring, mask_std=1000.0, rep0=100.0, projection=ring.d)
            for _ in range(3)
        ]
        # Each is individually acceptable...
        for leg in legs:
            self.assertLessEqual(leg.attempts, masking_module._MAX_ATTEMPTS)
        # ...and together they are not.
        joint = _joint_budget(legs)
        self.assertGreater(joint, masking_module._MAX_ATTEMPTS)
        with self.assertRaisesRegex(ValueError, "Lemma 2.14-3"):
            legs[0].for_attempts(joint)

    def test_the_gate_accepts_near_its_repetition_rate(self) -> None:
        """Rej0's whole claim: at `s = γ·T` the bimodal gate accepts about
        `1/M` of the time, and `M ≈ 1.003` is what makes Fig. 9 cheap."""
        ring = lnp_fixture.ring()
        masking = lnp_fixture.bimodal(ring, _WITNESS_COLS)
        rng = np.random.default_rng(2)
        centre = rng.integers(-40, 41, masking.projection).astype(np.int64)
        accepted = sum(
            masking.accepts(rng, masking.draw(rng)[1].reshape(-1) + centre, centre)
            for _ in range(200)
        )
        self.assertGreater(accepted, 190)

    def test_an_exhausted_budget_names_the_rate(self) -> None:
        """The budget the composition gives up on is the *joint* one, so the
        message is `ApproximateRange`'s and not a single masking's — one leg
        cannot name a number that depends on every leg's rate."""
        instance = _Instance(25)
        with self.assertRaisesRegex(RuntimeError, "Lemma 2.14-3"):
            raise instance.protocol._exhausted()


def _rotation_image(
    ring: HostSplitRing,
    width: int,
    columns: Sequence[int],
    exponents: Sequence[int] | None = None,
    offset: np.ndarray | None = None,
) -> AffineImage:
    """An `E` with one monomial per row: row `i` reads lift column
    `columns[i]`, rotated by `X^{exponents[i]}`.

    Norm-preserving by construction, which is what makes it usable as a
    round-trip witness at all — multiplication by `X^k` is a negacyclic
    rotation, so `‖E⃗s‖ = ‖⃗s‖` exactly and the leg's Gaussian stays sized
    for the witness the bound was derived from. A *uniform* `E` would be a
    perfectly correct statement about a vector no masking in this suite is
    parameterised for, and would reject its way to `exhausted` rather than
    tell anyone the image was the problem."""
    rows = len(columns)
    if exponents is None:
        exponents = [0] * rows
    matrix = ring.zeros(rows, width)
    monomials = ring.from_signed_stack(np.eye(ring.d, dtype=np.int64))
    for row, (column, exponent) in enumerate(zip(columns, exponents, strict=True)):
        matrix[row, column] = monomials[exponent]
    return AffineImage(
        matrix=matrix, offset=ring.zeros(rows) if offset is None else offset
    )


def _identity_columns(s1_take: int, ell: int) -> list[int]:
    """Where `(s1, m)` sit in the narrow lift `(s1, σ(s1), m, σ(m))` an
    affine image is written over — the columns an imageless leg selects."""
    slots = lift_slots(s1_take, ell)
    return list(np.concatenate([slots.s1, slots.message]))


class AffineImageTest(absltest.TestCase):
    """Fig. 10's `(D_i, u_i)` and `(E_i, v_i)`: a leg bounds an affine image
    of the lift (eq. 52, 53) instead of the witness itself.

    The weight is on the two structural tests, for this suite's usual
    reason — both sides build the statement from one method, so a wrong
    contraction verifies as happily as a right one. `E = I` is pinned
    against the scatter it must reproduce, and `project` against the image
    computed by hand.
    """

    def test_an_identity_image_rebuilds_the_selection_statement(self) -> None:
        """`matmul(σ₋₁(R), I)` lands exactly where `_witness_positions`
        scattered `σ₋₁(R)` — block for block, over the whole statement.

        The claim the general path rests on: the contraction and the
        selection are one map at `E = I`. Nothing downstream can see this
        difference, because a verifier contracts the same wrong way a prover
        does."""
        plain = _Instance(40)
        ring = plain.ring
        take = plain.protocol.evaluation.s1_take
        ell = plain.protocol.ell
        width = len(plain.leg._image_positions)
        identity = _rotation_image(ring, width, _identity_columns(take, ell))
        spelled = _Instance(40, images=[identity])

        # Same leg, same draw, same challenge — only how `E` reaches the
        # quadratic form differs.
        self.assertEqual(plain.leg._chunks, spelled.leg._chunks)
        sign, y = plain.masking.draw(np.random.default_rng(7))
        projection = plain.challenge()
        witness = np.concatenate([plain.s1[:take], plain.message])
        z = (sign * (projection @ witness.reshape(-1)) + y.reshape(-1)).reshape(y.shape)

        for block, (want, got) in enumerate(
            zip(
                plain.leg._evaluations(projection, z),
                spelled.leg._evaluations(projection, z),
                strict=True,
            )
        ):
            with self.subTest(block=block):
                np.testing.assert_array_equal(got, want)

    def test_the_projected_vector_is_the_image_of_the_lift(self) -> None:
        """`project` is `E⃗s − ⃗v` read back on `(−q/2, q/2]`, and it is what
        `respond` contracts `R` against.

        Pinned against the image computed directly rather than against a
        proof: `respond` would happily bound any vector handed to it."""
        instance = _Instance(41)
        ring = instance.ring
        take = instance.protocol.evaluation.s1_take
        ell = instance.protocol.ell
        width = 2 * (take + ell)
        offset = ring.from_signed_stack(
            np.random.default_rng(5).integers(-1, 2, (2, ring.d)).astype(np.int64)
        )
        # Reads one σ column on purpose: a statement about `σ(s1)` is as
        # writable as one about `s1`, and an image that could only reach the
        # identity copies would be the narrow case wearing a general name.
        slots = lift_slots(take, ell)
        columns = [int(slots.sigma_s1[0]), int(slots.message[0])]
        image = _rotation_image(ring, width, columns, [3, 0], offset)
        leg = _Instance(41, images=[image]).leg

        narrow = lift(
            ring,
            ring.from_signed_stack(instance.s1[:take]),
            ring.from_signed_stack(instance.message),
        )
        want = np.concatenate(
            [
                ring.to_balanced_limb0(row)
                for row in ring.sub(ring.matvec(image.matrix, narrow), offset)
            ]
        )
        np.testing.assert_array_equal(leg.project(narrow), want)

    def test_an_honest_image_proof_verifies(self) -> None:
        """A genuine `E` — a rotation of a permutation of the lift, reading
        both automorphism copies — proves and verifies end to end."""
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                probe = _Instance(seed)
                ring = probe.ring
                take = probe.protocol.evaluation.s1_take
                ell = probe.protocol.ell
                slots = lift_slots(take, ell)
                columns = [
                    int(slots.message[0]),
                    int(slots.s1[0]),
                    int(slots.sigma_s1[0]),
                ]
                image = _rotation_image(ring, 2 * (take + ell), columns, [0, 5, 1])
                instance = _Instance(seed, images=[image])
                self.assertTrue(instance.verify(instance.prove()))

    def test_an_offset_is_part_of_the_statement(self) -> None:
        """`⃗v` moves what is bounded, so a verifier holding a different one
        checks a different claim and the inner proof stops replaying.

        The masking is widened because `E⃗s − ⃗v` is genuinely longer than
        the witness — a ternary offset against a ternary image reaches
        coefficients of 2 — and a gate sized for `⃗s` alone would reject an
        honest prover."""
        ring = lnp_fixture.ring()
        take, ell = _M1, _ELL
        columns = _identity_columns(take, ell)
        offset = ring.from_signed_stack(
            np.random.default_rng(11)
            .integers(-1, 2, (len(columns), ring.d))
            .astype(np.int64)
        )
        image = _rotation_image(ring, 2 * (take + ell), columns, None, offset)
        wide = 4 * _WITNESS_COLS
        instance = _Instance(45, images=[image], witness_cols=wide)
        proof = instance.prove()
        self.assertTrue(instance.verify(proof))

        # The same proof against the same `E` and a different `⃗v`.
        other = _rotation_image(
            ring, 2 * (take + ell), columns, None, ring.zeros(len(columns))
        )
        self.assertFalse(
            instance.verify(
                proof,
                protocol=_Instance(45, images=[other], witness_cols=wide).protocol,
            )
        )

    def test_an_image_sizes_the_challenge_matrix(self) -> None:
        """`R` is squeezed to the *image's* width, not the witness's — the
        projected vector is `E⃗s − ⃗v`, and a leg that kept the witness's
        width would draw a matrix its own verifier could not replay."""
        ring = lnp_fixture.ring()
        take, ell = _M1, _ELL
        columns = _identity_columns(take, ell)[:1]
        image = _rotation_image(ring, 2 * (take + ell), columns)
        instance = _Instance(46, images=[image])
        self.assertEqual(instance.leg._chunks, 1)
        self.assertEqual(
            instance.challenge().shape, (instance.masking.projection, ring.d)
        )

    def test_an_image_may_not_reach_the_mask_or_the_sign(self) -> None:
        """The columns an image is written over are the caller's own halves,
        which is Fig. 10's `2(m1+ℓ)`. The mask and the sign hold values the
        prover redraws per attempt, so no statement can be a function of
        them — and here that is a width, not a check."""
        instance = _Instance(47)
        self.assertEqual(
            len(instance.leg._image_positions),
            2 * (instance.protocol.evaluation.s1_take + instance.protocol.ell),
        )
        self.assertLess(len(instance.leg._image_positions), instance.leg.width)

    def test_projecting_without_an_image_is_refused(self) -> None:
        """A leg bounding the witness takes the caller's own integers, so
        there is nothing here to compute — and handing back the witness
        instead would leave a caller that thought it had passed an image
        proving Fig. 9's statement under Fig. 10's name."""
        instance = _Instance(51)
        with self.assertRaisesRegex(ValueError, "no image"):
            instance.leg.project(instance.lifted())

    def test_a_misshapen_image_is_refused(self) -> None:
        ring = lnp_fixture.ring()
        take, ell = _M1, _ELL
        width = 2 * (take + ell)
        columns = _identity_columns(take, ell)
        cases = {
            "one ring column per position": AffineImage(
                matrix=ring.zeros(2, width + 1), offset=ring.zeros(2)
            ),
            "at least one": AffineImage(
                matrix=ring.zeros(0, width), offset=ring.zeros(0)
            ),
            "one ring element per row": AffineImage(
                matrix=ring.zeros(2, width), offset=ring.zeros(3)
            ),
        }
        for message, image in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _Instance(48, images=[image])

    def test_one_image_per_leg_or_none(self) -> None:
        ring = lnp_fixture.ring()
        image = _rotation_image(ring, 2 * (_M1 + _ELL), _identity_columns(_M1, _ELL))
        with self.assertRaisesRegex(ValueError, "affine image"):
            _Instance(49, legs=2, images=[image])

    def test_a_leg_without_an_image_bounds_the_witness_beside_one_that_does(
        self,
    ) -> None:
        """Fig. 10's own shape: `D` and `E` are different maps on the same
        commitment, and `None` is the leg that keeps Fig. 9's case."""
        ring = lnp_fixture.ring()
        image = _rotation_image(
            ring, 2 * (_M1 + _ELL), _identity_columns(_M1, _ELL), [2, 2, 2]
        )
        instance = _Instance(50, legs=2, images=[None, image])
        self.assertIsNone(instance.protocol.legs[0].image)
        self.assertIsNotNone(instance.protocol.legs[1].image)
        self.assertTrue(instance.verify(instance.prove()))


if __name__ == "__main__":
    absltest.main()
