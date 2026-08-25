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
from typing import Any

import numpy as np
from absl.testing import absltest
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.eval import AbdlopQuadraticEval
from zorch.lnp.masking import BimodalMasking
from zorch.lnp.quadratic import (
    AbdlopQuadratic,
    AbdlopQuadraticMany,
    Publics,
    evaluate,
    lift,
)
from zorch.lnp.range import ApproximateRange, RangeProof
from zorch.lnp.testing import lnp_fixture

_ROWS = 2
_M1 = lnp_fixture.M1
_M2 = 3
_ELL = 1
_LAM = 2
# The witness whose norm is the statement: `(s1, m)`, ternary.
_WITNESS_COLS = _M1 + _ELL


def _scheme(
    ring: HostSplitRing, mask_cols: int, s1_cols: int = _M1
) -> AbdlopCommitment:
    """The **twice-extended** scheme: its BDLOP half carries `ℓ + 256/d + 1
    + λ` messages, because `m‖y‖b‖g` is what the innermost protocol opens —
    this layer's mask and sign, and the garbage of the layer below."""
    return AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=s1_cols,
        randomness_cols=_M2,
        messages=_ELL + mask_cols + 1 + _LAM,
        beta1_inf=1,
        beta2_inf=1,
    )


def _protocol(
    ring: HostSplitRing,
    masking: BimodalMasking,
    s1_cols: int = _M1,
    s1_take: int | None = None,
) -> ApproximateRange:
    scheme = _scheme(ring, masking.mask_cols, s1_cols)
    # Re-derived for the same reason `quadratic_eval_test` re-derives it: the
    # point's T_1 bounds ‖s1‖ over `_M1` columns, and a wider Ajtai half
    # masked there rejects its way to `exhausted`.
    std = lnp_fixture.GAMMA * lnp_fixture.ETA * float(np.sqrt(s1_cols * ring.d))
    inner = lnp_fixture.masking(scheme, s1_std=std)
    evaluation = AbdlopQuadraticEval(
        AbdlopQuadraticMany(AbdlopQuadratic(inner)), _LAM, s1_take=s1_take
    )
    return ApproximateRange(evaluation, masking)


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return lnp_fixture.transcript(b"lnp-range-test", tag)


class _Instance:
    """One honest Fig. 9 statement: publics, a short ternary witness, its
    commitment, and the protocol over both."""

    def __init__(
        self, seed: int, s1_cols: int = _M1, s1_take: int | None = None
    ) -> None:
        ring = lnp_fixture.ring()
        self.ring = ring
        self.masking = lnp_fixture.bimodal(ring, _WITNESS_COLS)
        self.protocol = _protocol(ring, self.masking, s1_cols, s1_take)
        self.scheme = self.protocol.scheme
        mask_cols = self.masking.mask_cols
        rng = np.random.default_rng(seed)
        self.rng = rng

        self.a1 = ring.uniform_stack(rng, _ROWS, s1_cols)
        self.a2 = ring.uniform_stack(rng, _ROWS, _M2)
        self.b = ring.uniform_stack(rng, _ELL, _M2)
        self.b_mask = ring.uniform_stack(rng, mask_cols, _M2)
        self.b_sign = ring.uniform_stack(rng, 1, _M2)
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
        zero stacks — the derivation the statement tests index against."""
        ring = self.ring
        _, projection = self.protocol._challenge(
            _transcript(),
            ring.zeros(self.masking.mask_cols) if t_mask is None else t_mask,
            ring.zeros(1) if t_sign is None else t_sign,
        )
        return projection

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
        proof = instance.prove()
        self.assertEqual(proof.z.shape, (instance.masking.mask_cols, instance.ring.d))
        self.assertTrue(instance.masking.within_bounds(proof.z))

    def test_a_proof_does_not_verify_under_a_different_transcript(self) -> None:
        instance = _Instance(5)
        self.assertFalse(instance.verify(instance.prove(b"one"), b"two"))


class RangeSoundnessTest(absltest.TestCase):
    def test_a_tampered_projection_is_rejected(self) -> None:
        instance = _Instance(6)
        proof = instance.prove()
        z = proof.z.copy()
        z[0, 0] += 1
        self.assertFalse(instance.verify(dataclasses.replace(proof, z=z)))

    def test_a_tampered_mask_commitment_is_rejected(self) -> None:
        instance = _Instance(7)
        proof = instance.prove()
        self.assertFalse(
            instance.verify(
                dataclasses.replace(
                    proof, t_mask=lnp_fixture.bump(proof.t_mask, 0, 0, 0)
                )
            )
        )

    def test_a_tampered_sign_commitment_is_rejected(self) -> None:
        instance = _Instance(8)
        proof = instance.prove()
        self.assertFalse(
            instance.verify(
                dataclasses.replace(
                    proof, t_sign=lnp_fixture.bump(proof.t_sign, 0, 0, 0)
                )
            )
        )

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

        The only difference is `accept_t`, which touches nothing the
        transcript sees — so the inner proof still verifies and the verdict
        is the norm gate's alone. Tampering `z` upward would not pin this:
        that changes the statement and the inner proof rejects it first."""
        instance = _Instance(11)
        proof = instance.prove()
        strict = _protocol(
            instance.ring,
            lnp_fixture.bimodal(instance.ring, _WITNESS_COLS, accept_t=1e-6),
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
        e2, e1, e0 = protocol._evaluations(projection, z.reshape(y.shape))
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
        _, e1, _ = protocol._evaluations(instance.challenge(), zeros)
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
        crooked[protocol._sign_position] = ring.add(
            crooked[protocol._sign_position],
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
        base, _ = instance.protocol._statement(_transcript(), projection, z)
        other, _ = instance.protocol._statement(_transcript(), projection, moved)
        self.assertNotEqual(base.sample_scalar(16)[1], other.sample_scalar(16)[1])

    def test_the_sign_relation_vanishes_only_for_a_sign(self) -> None:
        instance = _Instance(14)
        ring = instance.ring
        protocol = instance.protocol
        r2, r1, r0 = protocol._relation()
        for sign in (1, -1):
            with self.subTest(sign=sign):
                value = evaluate(ring, r2[0], r1[0], r0[0], instance.lifted(sign))
                self.assertFalse(value.any())
        # `b = 2` satisfies integrality but not `b² = 1`.
        two = instance.lifted(1)
        two[protocol._sign_position] = ring.from_signed([2] + [0] * (ring.d - 1))
        self.assertTrue(evaluate(ring, r2[0], r1[0], r0[0], two).any())


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
        got = s[instance.protocol._witness_positions]
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
            s[protocol._witness_positions],
            np.concatenate(
                [
                    ring.from_signed_stack(instance.s1[:_M1]),
                    ring.from_signed_stack(instance.message),
                ]
            ),
        )
        np.testing.assert_array_equal(
            s[protocol._mask_positions], ring.from_signed_stack(y)
        )
        np.testing.assert_array_equal(
            s[protocol._sign_position], ring.from_signed_stack(sign_row)[0]
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
            s[protocol._mask_positions], ring.from_signed_stack(y)
        )
        np.testing.assert_array_equal(
            s[protocol._sign_position], ring.from_signed_stack(sign_row)[0]
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
        """Proof fields are a verdict, never a raise — `zorch/lnp/wire.py`."""
        instance = _Instance(20)
        proof = instance.prove()
        ring = instance.ring
        malformed: tuple[tuple[str, Any], ...] = (
            ("t_mask", ring.zeros(instance.masking.mask_cols + 1)),
            ("t_sign", ring.zeros(2)),
            ("z", np.zeros((instance.masking.mask_cols + 1, ring.d), np.int64)),
            ("z", proof.z.astype(np.float64)),
            ("evaluation", None),
        )
        for field, value in malformed:
            with self.subTest(field=field):
                self.assertFalse(
                    instance.verify(dataclasses.replace(proof, **{field: value}))
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
            (dict(accept_t=0.0), "accept_t must be positive"),
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
        ring = lnp_fixture.ring()
        masking = lnp_fixture.bimodal(ring, _WITNESS_COLS)
        with self.assertRaisesRegex(RuntimeError, "Lemma 2.14-3"):
            raise masking.exhausted("range.prove")


if __name__ == "__main__":
    absltest.main()
