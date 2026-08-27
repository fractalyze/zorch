# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The exact ℓ2 bound (Fig. 10, eq. 53 at `E = I`) — the two evaluations it
builds, and the end-to-end proof they buy.

The weight here is on the *algebra*, for the reason it is one layer down: a
prove/verify round-trip cannot see a wrong statement. Both sides build `G`
and `I` from the same object, so a builder that encoded the wrong inner
product produces a proof that verifies happily and means something else.
Each function is therefore pinned against what it is supposed to evaluate
to, over the ring, at witnesses chosen to make each failure visible:

- `G` must vanish exactly when the digits are binary — the whole content of
  Lemma 5.2 — and *not* vanish as a ring element, or it would be a relation
  and this layer would be claiming something stronger and false.
- `I` must vanish exactly when `‖s‖² + ⟨p⃗, x⃗⟩ = β²`, which is the identity
  that makes the bound tight. A test that only checked the honest witness
  could not tell it from `I ≡ 0`.

And one test exists because the two together are the point: binarity alone
admits `⟨p⃗, x⃗⟩` reading back as anything, and the norm identity alone admits
digits that are not digits. Dropping either leaves the other's proof green.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.eval import AbdlopQuadraticEval
from zorch.lnp.exact import ExactL2, binarity
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
# The Ajtai half carries the witness *and* the one column the binary
# decomposition needs, which is what §5.2 appends to `s1`.
_DIGIT_COLS = 1
_S1_COLS = _M1 + _DIGIT_COLS
# The vector whose norm is the statement: `(s1, m)`, without the digits.
_WITNESS_COLS = _M1 + _ELL


def _scheme(ring: HostSplitRing, mask_cols: int) -> AbdlopCommitment:
    return AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=_S1_COLS,
        randomness_cols=_M2,
        messages=_ELL + mask_cols + 1 + _LAM,
        beta1_inf=1,
        beta2_inf=1,
    )


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return lnp_fixture.transcript(b"lnp-exact-test", tag)


class _Instance:
    """One honest exact-ℓ2 statement: a ternary `(s1, m)`, the digits of its
    slack under `β`, and the protocol stack over both.

    `β` is the ternary bound `√(witness_cols·d)` rounded up, so the witness
    really does satisfy it and the slack really is nonnegative — the bound
    is tight in the sense that matters here, that nothing below rounds it
    up by a factor of 189."""

    def __init__(self, seed: int) -> None:
        ring = lnp_fixture.ring()
        self.ring = ring
        self.masking = lnp_fixture.bimodal(ring, _WITNESS_COLS + _DIGIT_COLS)
        scheme = _scheme(ring, self.masking.mask_cols)
        self.evaluation = AbdlopQuadraticEval(
            AbdlopQuadraticMany(
                AbdlopQuadratic(
                    lnp_fixture.masking(
                        scheme, s1_std=lnp_fixture.s1_std(ring, _S1_COLS)
                    )
                )
            ),
            _LAM,
        )
        self.protocol = ApproximateRange(self.evaluation, self.masking)
        self.bound = lnp_fixture.ternary_beta(ring, _WITNESS_COLS)
        self.exact = ExactL2(self.evaluation, _M1, self.protocol.ell, self.bound)

        rng = np.random.default_rng(seed)
        self.rng = rng
        self.witness = rng.integers(-1, 2, (_M1, ring.d)).astype(np.int64)
        self.message = rng.integers(-1, 2, (_ELL, ring.d)).astype(np.int64)
        self.digits = self.exact.decompose(np.concatenate([self.witness, self.message]))
        # The Ajtai half is `(s1, x)`: footnote 14 — the digits have to be
        # committed with `s1`, not appended later.
        self.s1 = np.concatenate([self.witness, self.digits])
        self.s2 = rng.integers(-1, 2, (_M2, ring.d)).astype(np.int64)

        self.a1 = ring.uniform_stack(rng, _ROWS, _S1_COLS)
        self.a2 = ring.uniform_stack(rng, _ROWS, _M2)
        self.b = ring.uniform_stack(rng, _ELL, _M2)
        b_mask = ring.uniform_stack(rng, self.masking.mask_cols, _M2)
        b_sign = ring.uniform_stack(rng, 1, _M2)
        bg = ring.uniform_stack(rng, _LAM, _M2)
        self.publics = Publics(
            a1=self.a1,
            a2=self.a2,
            blocks=np.concatenate([self.b, b_mask, b_sign, bg]),
            b_quad=ring.uniform_stack(rng, _M2),
        )

        s1_ring = ring.from_signed_stack(self.s1)
        s2_ring = ring.from_signed_stack(self.s2)
        self.t_a = ring.add(
            ring.matvec(self.a1, s1_ring), ring.matvec(self.a2, s2_ring)
        )
        self.t_b = ring.add(
            ring.matvec(self.b, s2_ring), ring.from_signed_stack(self.message)
        )

    def lifted(self, digits: np.ndarray | None = None) -> np.ndarray:
        """The eval layer's lift of `(s1‖x, m‖y‖b)` at a zero mask and a
        `+1` sign, with the digits overridable.

        The mask and sign are zero because these tests are about the two
        exact-ℓ2 functions, which touch neither — a nonzero there would only
        make a misread position harder to see."""
        ring = self.ring
        if digits is None:
            digits = self.digits
        mask = np.zeros((self.masking.mask_cols, ring.d), dtype=np.int64)
        sign = np.zeros((1, ring.d), dtype=np.int64)
        sign[0, 0] = 1
        return lift(
            ring,
            ring.from_signed_stack(np.concatenate([self.witness, digits])),
            ring.from_signed_stack(np.concatenate([self.message, mask, sign])),
        )

    def values(self, digits: np.ndarray | None = None) -> np.ndarray:
        """`(G(x), I(s, x))` evaluated on the lift, as a two-element stack."""
        ring = self.ring
        e2, e1, e0 = self.exact.evaluations()
        s = self.lifted(digits)
        return np.concatenate(
            [evaluate(ring, e2[j], e1[j], e0[j], s) for j in range(2)]
        )


class ExactDecompositionTest(absltest.TestCase):
    def test_the_digits_read_back_as_the_slack(self) -> None:
        """`⟨p⃗, x⃗⟩ = β² − ‖s‖²`, which is the whole content of the
        decomposition and the thing `I` is written against."""
        instance = _Instance(1)
        witness = np.concatenate([instance.witness, instance.message])
        norm_sq = int((witness.astype(object) ** 2).sum())
        digits = instance.digits[0, : instance.exact.digits]
        read_back = int(sum(int(b) << i for i, b in enumerate(digits)))
        self.assertEqual(read_back, instance.bound**2 - norm_sq)

    def test_the_digits_are_binary(self) -> None:
        instance = _Instance(2)
        self.assertEqual(set(np.unique(instance.digits).tolist()) - {0}, {1})

    def test_an_over_bound_witness_is_refused(self) -> None:
        """There is no exact proof of a false statement to build, so this
        raises where a protocol layer would return a verdict — the caller
        handed in a witness that does not satisfy its own claim."""
        instance = _Instance(3)
        # Not an all-ones ternary witness: that is exactly the worst case
        # `β = ⌈√(witness_cols·d)⌉` was derived from, so it sits *inside* the
        # bound. Twos are genuinely past it.
        over = np.full((_M1 + _ELL, instance.ring.d), 2, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "past the bound"):
            instance.exact.decompose(over)

    def test_a_bound_too_wide_for_one_ring_element_is_refused(self) -> None:
        instance = _Instance(4)
        with self.assertRaisesRegex(ValueError, "past the ring degree"):
            ExactL2(instance.evaluation, _M1, instance.protocol.ell, 1 << 40)


class ExactStatementTest(absltest.TestCase):
    """`G` and `I` against the ring, not against a proof going through."""

    def test_both_functions_vanish_in_their_constant_coefficient(self) -> None:
        instance = _Instance(5)
        values = instance.values()
        self.assertFalse(instance.ring.constant_coeff(values).any())
        # ...and not as ring elements: claiming these as relations would be
        # a stronger statement, and a false one.
        self.assertTrue(values.any())

    def test_the_binarity_function_detects_a_non_binary_digit(self) -> None:
        """Lemma 5.2 is what licenses reading `x⃗` as a number at all. A `2`
        in the digits keeps `⟨p⃗, x⃗⟩` meaningful-looking while making the
        vector not a binary representation."""
        instance = _Instance(6)
        crooked = instance.digits.copy()
        crooked[0, 0] = 2
        binarity = instance.ring.constant_coeff(instance.values(crooked))
        self.assertTrue(binarity[0].any())

    def test_the_norm_identity_detects_a_wrong_slack(self) -> None:
        """Flipping a digit changes what `⟨p⃗, x⃗⟩` reads back as, so the
        identity `‖s‖² + ⟨p⃗, x⃗⟩ = β²` stops holding — while `0 ↔ 1` keeps
        the digits binary, so `G` still vanishes. That is the point of
        having both: they fail independently."""
        instance = _Instance(7)
        flipped = instance.digits.copy()
        flipped[0, 0] ^= 1
        values = instance.ring.constant_coeff(instance.values(flipped))
        self.assertFalse(values[0].any())
        self.assertTrue(values[1].any())

    def test_digits_past_the_radix_are_free_and_harmless(self) -> None:
        """`p⃗ = (1, 2, …, 2^{digits-1}, 0, …, 0)` is zero-padded to the ring
        degree, so a digit above the top used bit contributes nothing to
        `⟨p⃗, x⃗⟩` and neither function moves.

        That is an unused degree of freedom rather than a gap: such a digit
        is still forced binary by `G`, still counted by the range leg's norm
        bound, and `⟨p⃗, x⃗⟩` still pins `‖s‖²` exactly — a prover who sets
        one has changed nothing it could profit from. Pinned because the
        alternative reading — that `I` failed to notice a tampered digit —
        is the one a reader reaches for first."""
        instance = _Instance(14)
        padded = instance.digits.copy()
        padded[0, instance.exact.digits] = 1
        values = instance.ring.constant_coeff(instance.values(padded))
        self.assertFalse(values.any())

    def test_the_norm_identity_moves_with_the_witness(self) -> None:
        """A guard on the fixture as much as the builder: if `I` did not
        read `⟨s, s⟩` at all, every assertion above would still pass while
        proving a statement about the digits alone."""
        instance = _Instance(8)
        instance.witness = instance.witness.copy()
        instance.witness[0, 0] += 1
        moved = instance.ring.constant_coeff(instance.values())
        self.assertTrue(moved[1].any())


class BinarityTest(absltest.TestCase):
    """Eq. 54 on its own — `E_bin s − v_bin ∈ {0,1}` at `E_bin` a selection.

    Lemma 5.2 is the whole content: `⟨v⃗, v⃗ − 1⃗⟩ = 0` forces binary, because
    every term `v_t(v_t − 1)` is nonnegative over ℤ and vanishes only at 0
    and 1. Pinned against the ring rather than a round-trip, for this file's
    standing reason — both sides build the function from one object, so a
    builder that named the wrong columns proves something else and verifies.
    """

    def _value(
        self,
        instance: _Instance,
        family: tuple[np.ndarray, np.ndarray, np.ndarray],
        s1_extra: np.ndarray | None = None,
    ) -> np.ndarray:
        """The one evaluation's value on the lift, with the Ajtai half's
        binary column overridable."""
        ring = instance.ring
        e2, e1, e0 = family
        s = instance.lifted(digits=s1_extra)
        return evaluate(ring, e2[0], e1[0], e0[0], s)

    def test_it_vanishes_on_binary_columns(self) -> None:
        """The digits are binary by construction, so a statement naming
        their column vanishes in its constant coefficient — and not as a
        ring element, or it would be a relation claiming something
        stronger."""
        instance = _Instance(20)
        family = binarity(instance.evaluation, s1_columns=[_M1])
        value = self._value(instance, family)
        self.assertFalse(instance.ring.constant_coeff(value).any())
        self.assertTrue(value.any())

    def test_it_detects_a_non_binary_coefficient(self) -> None:
        """`-1` is the half worth naming: it is the case a `v² = v`-shaped
        check would miss, and the one Lemma 5.2's nonnegativity argument
        exists to cover — `(-1)(-2) = 2 > 0`, so the sum cannot cancel it
        against another term."""
        instance = _Instance(21)
        family = binarity(instance.evaluation, s1_columns=[_M1])
        for planted in (2, -1):
            with self.subTest(planted=planted):
                crooked = instance.digits.copy()
                crooked[0, 0] = planted
                value = self._value(instance, family, s1_extra=crooked)
                self.assertTrue(instance.ring.constant_coeff(value).any())

    def test_it_names_the_columns_it_is_about(self) -> None:
        """A statement over the *witness* column does not vanish — the
        witness is ternary, not binary. That the same builder gives opposite
        verdicts on two columns of one lift is what says it reads the
        column it was told rather than a fixed one."""
        instance = _Instance(23)
        digits = binarity(instance.evaluation, s1_columns=[_M1])
        witness = binarity(instance.evaluation, s1_columns=[0])
        self.assertFalse(
            instance.ring.constant_coeff(self._value(instance, digits)).any()
        )
        self.assertTrue(
            instance.ring.constant_coeff(self._value(instance, witness)).any()
        )

    def test_a_column_outside_the_statement_is_refused(self) -> None:
        instance = _Instance(24)
        with self.assertRaisesRegex(ValueError, "binary s1 column outside"):
            binarity(instance.evaluation, s1_columns=[instance.evaluation.s1_take])
        with self.assertRaisesRegex(ValueError, "binary message column outside"):
            binarity(instance.evaluation, message_columns=[instance.evaluation.ell])

    def test_an_empty_claim_is_refused(self) -> None:
        """`⟨(), () − 1⃗⟩ = 0` holds for every witness, so it is an
        evaluation that proves nothing while looking like a proof."""
        instance = _Instance(25)
        with self.assertRaisesRegex(ValueError, "needs a column"):
            binarity(instance.evaluation)


class ExactWraparoundTest(absltest.TestCase):
    def test_each_wraparound_condition_is_checked(self) -> None:
        """Both statements are proved mod q, and an integer identity that
        wrapped is not one. Thm 5.3's conditions are checked rather than
        assumed because a violation reads exactly like a valid proof.

        Each is tripped on its own, which needs choosing `B` between the
        thresholds: at this parameter point Lemma 2.9's precondition binds at
        `q/(41·c) ≈ 4.1e5` and the binarity check at `≈ 6.6e4`, so a `B` in
        between clears the first and fails the second. A single huge `B`
        would violate both and only ever prove the earliest check runs."""
        instance = _Instance(9)
        instance.exact.require_no_wraparound(16)
        with self.assertRaisesRegex(ValueError, "the binarity check"):
            instance.exact.require_no_wraparound(100_000)
        with self.assertRaisesRegex(ValueError, "Lemma 2.9"):
            instance.exact.require_no_wraparound(500_000)

    def test_binary_columns_widen_the_projection_lemma_29_bounds(self) -> None:
        """`c` is the projected vector's width in *integers*, so eq. 54's
        binary columns count in it — they are part of `x'` and therefore part
        of what the range leg has to bound.

        Leaving them out silently loosens Lemma 2.9's precondition, which is
        the one condition whose failure means the projection establishes
        nothing at all rather than merely establishing it modulo q."""
        instance = _Instance(15)
        # 60_000 clears every condition at `binary_cols = 0` (Lemma 2.9 binds
        # at q/(41·256) ≈ 4.1e5 there). Enough binary columns pull `c` up
        # until it does not — 30 is well past a realistic point, which is the
        # point: the formula has to move with `c` at all.
        instance.exact.require_no_wraparound(60_000)
        with self.assertRaisesRegex(ValueError, "Lemma 2.9"):
            instance.exact.require_no_wraparound(60_000, binary_cols=30)

    def test_the_honest_parameter_point_has_room(self) -> None:
        """The conditions are not tight at any sane point — `q` is ~2^32 and
        the projection of a ternary witness is tiny — so the guard should
        never fire on the suite's own numbers.

        `B` is the ℓ2 bound Prop. 5.1 actually proves about `(s ‖ x⃗)`, not
        `masking.projection`, which is the projection *dimension*: 256 rows,
        a count. Passing the count happens to satisfy the conditions too, so
        the distinction is invisible here and worth spelling — the number
        this gate reads is a norm."""
        instance = _Instance(10)
        instance.exact.require_no_wraparound(instance.masking.proven_norm())


class ExactRoundTripTest(absltest.TestCase):
    """The exact statement riding the range proof, which is what makes it
    hold over ℤ rather than only mod q."""

    def _prove(self, instance: _Instance) -> RangeProof:
        proof, _ = instance.protocol.prove(
            instance.publics,
            s1=instance.s1,
            s2=instance.s2,
            message=instance.message,
            rng=instance.rng,
            transcript=_transcript(),
            evaluations=instance.exact.evaluations(),
        )
        return proof

    def _verify(self, instance: _Instance, proof: RangeProof, **overrides: Any) -> bool:
        args: dict[str, Any] = dict(evaluations=instance.exact.evaluations())
        args.update(overrides)
        ok, _ = instance.protocol.verify(
            instance.publics,
            t_a=instance.t_a,
            t_b=instance.t_b,
            proof=proof,
            transcript=_transcript(),
            **args,
        )
        return ok

    def test_an_honest_exact_proof_verifies(self) -> None:
        instance = _Instance(11)
        self.assertTrue(self._verify(instance, self._prove(instance)))

    def test_the_range_leg_still_bounds_the_committed_vector(self) -> None:
        """The approximate proof is not decoration here — it is what rules
        out a wraparound in the two exact identities, so it has to be about
        the vector that includes the digits."""
        instance = _Instance(12)
        proof = self._prove(instance)
        self.assertTrue(instance.masking.within_bounds(proof.legs[0].z))

    def test_a_proof_of_a_wrong_slack_is_rejected(self) -> None:
        """The digits are committed, so a prover that decomposes a different
        number cannot make `I` vanish — this is the exactness, end to end."""
        instance = _Instance(13)
        instance.digits = instance.digits.copy()
        instance.digits[0, 0] ^= 1
        instance.s1 = np.concatenate([instance.witness, instance.digits])
        ring = instance.ring
        s1_ring = ring.from_signed_stack(instance.s1)
        s2_ring = ring.from_signed_stack(instance.s2)
        instance.t_a = ring.add(
            ring.matvec(instance.a1, s1_ring), ring.matvec(instance.a2, s2_ring)
        )
        self.assertFalse(self._verify(instance, self._prove(instance)))


if __name__ == "__main__":
    absltest.main()
