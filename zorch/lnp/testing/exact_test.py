# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The exact ℓ2 bound (Fig. 10, eq. 53) — the two evaluations it builds, and
the end-to-end proof they buy, at `E = I` and at a general affine image.

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

`ExactAffineImageTest` carries the same weight for the general case. Its
load-bearing test is that an explicitly spelled `E = I, ⃗v = 0` rebuilds the
diagonal blocks the direct branch writes — the four terms of the expansion
have to collapse, and no proof going through would show it if they did not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from absl.testing import absltest
from lattice_frx import primes
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.eval import AbdlopQuadraticEval
from zorch.lnp.exact import ExactL2, binarity
from zorch.lnp.masking import LinfBound
from zorch.lnp.quadratic import (
    SIGMA_ORDER,
    AbdlopQuadratic,
    AbdlopQuadraticMany,
    AffineImage,
    Publics,
    evaluate,
    lift,
)
from zorch.lnp.range import ApproximateRange, ProjectionLeg, RangeProof
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

    def __init__(
        self,
        seed: int,
        image: AffineImage | None = None,
        range_image: AffineImage | None = None,
        bound: int | None = None,
        ring: HostSplitRing | None = None,
    ) -> None:
        ring = lnp_fixture.ring() if ring is None else ring
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
        self.protocol = ApproximateRange(
            self.evaluation,
            self.masking,
            images=[range_image] if range_image is not None else (),
        )
        # `is None`, not falsy: `bound=0` is a value a test may want to
        # hand `ExactL2` to watch it refuse a non-positive bound, and `or`
        # would quietly hand it the default instead.
        self.bound = (
            lnp_fixture.ternary_beta(ring, _WITNESS_COLS) if bound is None else bound
        )
        self.image = image
        self.exact = ExactL2(self.evaluation, _M1, self.protocol.ell, self.bound, image)

        rng = np.random.default_rng(seed)
        self.rng = rng
        self.witness = rng.integers(-1, 2, (_M1, ring.d)).astype(np.int64)
        self.message = rng.integers(-1, 2, (_ELL, ring.d)).astype(np.int64)
        # What the bound is about: `(s1, m)` itself, or its affine image.
        # `AffineImage.apply` is the same call `range.ProjectionLeg.project`
        # makes, which is what keeps the two halves of Fig. 10's eq. 53 —
        # the leg's projection and this norm identity — about one vector.
        self.bounded = np.concatenate([self.witness, self.message])
        if image is not None:
            self.bounded = image.apply(
                ring,
                lift(
                    ring,
                    ring.from_signed_stack(self.witness),
                    ring.from_signed_stack(self.message),
                ),
            ).reshape(image.rows, ring.d)
        self.digits = self.exact.decompose(self.bounded)
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

    def test_a_message_column_count_outside_the_statement_is_refused(self) -> None:
        """`message_cols` is a count, and both bad directions are silent
        rather than loud: a negative one slices `slots.message[:-1]`, which
        proves a *prefix* while `c` under-counts the projected width by the
        columns it dropped, and an over-count clamps in the slice while
        inflating `c`. Neither raises on its own."""
        instance = _Instance(5)
        for bad in (-1, instance.evaluation.ell + 1):
            with self.subTest(message_cols=bad):
                with self.assertRaisesRegex(ValueError, "column message half"):
                    ExactL2(instance.evaluation, _M1, bad, instance.exact.bound)


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

    def test_it_covers_the_message_half_too(self) -> None:
        """`message_columns` was the untested half of the signature: every
        production caller names Ajtai columns, and the only test that passed
        a message column was the out-of-range rejection — so the branch that
        indexes `slots.message` had never built a family anyone evaluated.

        Both verdicts on one lift, as above: the column is made binary and
        the statement vanishes, then a `2` is planted in it and it does not.
        The digits are untouched throughout, so a family that silently read
        the Ajtai half instead would vanish in both halves of this test.
        """
        instance = _Instance(30)
        instance.message = np.zeros_like(instance.message)
        instance.message[0, :4] = 1
        family = binarity(instance.evaluation, message_columns=[0])
        self.assertFalse(
            instance.ring.constant_coeff(self._value(instance, family)).any()
        )
        instance.message[0, 0] = 2
        self.assertTrue(
            instance.ring.constant_coeff(self._value(instance, family)).any()
        )

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


def _leg_at_bound(
    instance: _Instance, target: int, image: AffineImage | None = None
) -> ProjectionLeg:
    """A range leg over `instance`'s evaluation whose `proven_norm()` is
    `target`, so a test can put `B` where the condition it wants to trip
    lives.

    Scaled off the instance's own masking rather than by inverting the formula:
    `L2Bound.proven_norm` is linear in `mask_std` and nothing else, so one
    division lands on `target` up to its ceiling — and the constant chain
    `2·√(256/26)·t` stays spelled in exactly one place, `masking.py`. (It is
    also pinned as a value by the range suite; a third copy here would be the
    one nobody updates.)

    Going through a real leg rather than passing a number is the point of the
    API under test: `B` and `c` now describe a proof that exists."""
    reference = instance.masking
    masking = lnp_fixture.bimodal(
        instance.ring,
        _WITNESS_COLS + _DIGIT_COLS,
        mask_std=reference.mask_std * target / reference.proven_norm(),
    )
    protocol = ApproximateRange(
        instance.evaluation, masking, images=[image] if image is not None else ()
    )
    return protocol.legs[0]


def _leg_image(instance: _Instance, columns: Sequence[int]) -> AffineImage:
    """An image over the *leg's* lift `(s1‖x, m)` — wider than the one
    `ExactL2`'s own `E` is written over, which is the gap `range_image`
    composes across."""
    ell = instance.protocol.ell
    return lnp_fixture.rotation_image(
        instance.ring, SIGMA_ORDER * (_S1_COLS + ell), list(columns)
    )


def _leg_columns(instance: _Instance, extra_cols: int = 0) -> list[int]:
    """The lift positions an honest leg bounds, plus `extra_cols` more — the
    shape a statement carrying `E_bin` columns rides on."""
    columns = lnp_fixture.identity_columns(_S1_COLS, instance.protocol.ell)
    return columns + [columns[0]] * extra_cols


def _probe_image(
    ring: HostSplitRing, ell: int, offset: np.ndarray | None = None
) -> AffineImage:
    """The suite's standard non-trivial `E`: one rotation per lift position
    of `(s1, m)`, written over `ExactL2`'s own narrow lift.

    Norm-preserving, so an honest witness still decomposes under the fixture
    bound — a uniform `E` would be a correct statement about a vector no
    masking here is parameterised for. `offset` is `⃗v`, zero when omitted.

    The rotations are held here rather than passed in: two tests comparing
    the same statement through different paths have to be about one `E`, and
    a per-test spelling is how they stop being."""
    return lnp_fixture.rotation_image(
        ring,
        SIGMA_ORDER * (_M1 + ell),
        lnp_fixture.identity_columns(_M1, ell),
        [1, 6, 2],
        offset,
    )


def _ternary_offset(ring: HostSplitRing) -> np.ndarray:
    """A ternary `⃗v` for the probe image — non-zero, so a statement that
    drops it is visible.

    One row per `(s1, m)` column, which is what that image selects. Against
    a ternary image this reaches coefficients of 2, so every caller also
    widens the bound it hands `ExactL2`."""
    return ring.from_signed_stack(
        np.random.default_rng(3).integers(-1, 2, (_WITNESS_COLS, ring.d))
    )


def _imaged(
    seed: int, offset: np.ndarray | None = None, bound: int | None = None
) -> tuple[_Instance, AffineImage]:
    """An instance carrying the suite's standard `E`, and that `E`.

    `_scheme` gives the caller exactly `_ELL` message columns, and that is
    what `ApproximateRange` hands back as its own `ell` — so the image is
    built without standing up a probe instance to ask for it."""
    ring = lnp_fixture.ring()
    image = _probe_image(ring, _ELL, offset)
    return _Instance(seed, image=image, bound=bound, ring=ring), image


class ExactRangeImageTest(absltest.TestCase):
    """Eq. 61's `e⃗^(e) = [E⃗s − ⃗v ; x⃗]` — the image the range leg has to
    carry.

    A round-trip cannot see this wrong: a leg built from a mis-composed
    image bounds some other vector perfectly well and its proof verifies.
    So the composition is pinned against what it must evaluate to, over the
    ring, on an honest witness."""

    def _assert_composes(self, instance: _Instance, image: AffineImage) -> AffineImage:
        """The composed image, applied to the leg's lift `(s1‖x, m)`,
        evaluates to `E⃗s − ⃗v` on `E`'s rows and `x⃗` on the rest.

        Hands the composition back, so a caller with further claims about it
        does not build a second one."""
        ring = instance.ring
        composed = instance.exact.range_image()
        assert composed is not None
        self.assertEqual(composed.rows, image.rows + _DIGIT_COLS)
        wide = lift(
            ring,
            ring.from_signed_stack(instance.s1),
            ring.from_signed_stack(instance.message),
        )
        # `apply` hands back flat balanced coefficients, as `_Instance` and
        # `ProjectionLeg.respond` both read them.
        got = composed.apply(ring, wide).reshape(composed.rows, ring.d)
        np.testing.assert_array_equal(got[: image.rows], instance.bounded)
        np.testing.assert_array_equal(got[image.rows :], instance.digits)
        return composed

    def test_the_witness_case_needs_no_image(self) -> None:
        """At `E = I` the imageless leg already bounds `(s1‖x, m)`, whose
        coefficients are the composed vector's in a different order — so the
        norm it proves is the one eq. 61 needs, and spelling an identity out
        would cost a contraction to say the same thing."""
        self.assertIsNone(_Instance(70).exact.range_image())

    def test_the_composition_selects_the_image_then_the_digits(self) -> None:
        """The load-bearing claim: applied to the leg's lift `(s1‖x, m)`,
        the composed image evaluates to `E⃗s − ⃗v` on its first rows and to
        `x⃗` on the rows appended after them.

        `E` is written over `(s1, σ(s1), m, σ(m))` *without* the digits and
        the leg's lift has them, so every column of `E` shifts. Reading the
        digits out of the same object is what ties the two halves of eq. 53
        to one vector."""
        instance, image = _imaged(71)
        self._assert_composes(instance, image)

    def test_the_composition_carries_the_offset(self) -> None:
        """`⃗v` is half of `E⃗s − ⃗v`, and the composition has to carry it —
        onto the image's own rows and nowhere else.

        Pinned with a non-zero offset because the fixture's default is zero,
        and against zero a dropped `⃗v` is invisible: the leg would then
        bound `E⃗s` while eq. 66's inner product is written about
        `E⃗s − ⃗v`, and both proofs verify. The digit rows select `x⃗`
        outright, so their offset must stay zero."""
        ring = lnp_fixture.ring()
        offset = _ternary_offset(ring)
        # A ternary offset against a ternary image reaches coefficients of
        # 2, so the bound is widened or `decompose` refuses the witness.
        instance, image = _imaged(
            73, offset, bound=2 * lnp_fixture.ternary_beta(ring, _WITNESS_COLS)
        )
        composed = self._assert_composes(instance, image)
        np.testing.assert_array_equal(composed.offset[: image.rows], offset)
        np.testing.assert_array_equal(
            composed.offset[image.rows :], ring.zeros(_DIGIT_COLS)
        )

    def test_the_composed_image_spans_the_legs_lift(self) -> None:
        """`E` is written over `2(m1+ℓ)` and the leg's lift is
        `2(s1_take+ℓ)`. The composition is what crosses that gap, and a leg
        refuses an image of either other width — which is why passing
        `ExactL2`'s own `E` to a leg is a shape error rather than a silent
        proof of something else."""
        instance, _ = _imaged(72)
        ell = instance.protocol.ell
        composed = instance.exact.range_image()
        assert composed is not None and instance.exact.image is not None
        self.assertEqual(composed.matrix.shape[1], SIGMA_ORDER * (_S1_COLS + ell))
        self.assertEqual(
            instance.exact.image.matrix.shape[1], SIGMA_ORDER * (_M1 + ell)
        )


class ExactWraparoundTest(absltest.TestCase):
    def test_each_wraparound_condition_is_checked(self) -> None:
        """Both statements are proved mod q, and an integer identity that
        wrapped is not one. Thm 5.3's conditions are checked rather than
        assumed because a violation reads exactly like a valid proof.

        Each is tripped on its own, which needs putting `B` between the
        thresholds: at this parameter point Lemma 2.9's precondition binds at
        `q/(41·c) ≈ 4.1e5` and the binarity check at `≈ 6.6e4`, so a `B` in
        between clears the first and fails the second. A single huge `B`
        would violate both and only ever prove the earliest check runs."""
        instance = _Instance(9)
        instance.exact.require_no_wraparound(_leg_at_bound(instance, 16))
        with self.assertRaisesRegex(ValueError, "the binarity check"):
            instance.exact.require_no_wraparound(_leg_at_bound(instance, 100_000))
        with self.assertRaisesRegex(ValueError, "Lemma 2.9"):
            instance.exact.require_no_wraparound(_leg_at_bound(instance, 500_000))

    def test_binary_columns_widen_the_projection_lemma_29_bounds(self) -> None:
        """`c` is the projected vector's width in *integers*, so eq. 54's
        binary columns count in it — they are part of `x'` and therefore part
        of what the range leg has to bound.

        Leaving them out silently loosens Lemma 2.9's precondition, which is
        the one condition whose failure means the projection establishes
        nothing at all rather than merely establishing it modulo q. Read off
        the `41·c` the message reports, since the leg must genuinely bound
        those columns for the call to get that far at all."""
        instance = _Instance(15)
        ring = instance.ring
        for extra in (0, 30):
            with self.subTest(binary_cols=extra):
                image = _leg_image(instance, _leg_columns(instance, extra))
                leg = _leg_at_bound(instance, 500_000, image)
                with self.assertRaisesRegex(
                    ValueError, rf"Lemma 2\.9.*/{41 * image.rows * ring.d}\b"
                ):
                    instance.exact.require_no_wraparound(leg, binary_cols=extra)

    def test_a_negative_binary_width_is_refused_before_the_bounds(self) -> None:
        """The one argument still passed by hand, and every way it can be
        wrong fails *open*: a negative count shrinks `span` and drops a ring
        element from the expected width, buying a gate that passes rather
        than one that raises. So it is rejected before any bound is computed.

        `B` is no longer among the arguments — it is read off the leg — which
        retires the matching negative-bound case: `proven_norm` is a ceiling
        over positive quantities and has no negative value to return."""
        instance = _Instance(16)
        leg = _leg_at_bound(instance, 16)
        with self.assertRaisesRegex(ValueError, "binary-column width"):
            instance.exact.require_no_wraparound(leg, binary_cols=-1)

    def test_a_leg_bounding_a_different_vector_is_refused(self) -> None:
        """The leg has to bound eq. 61's `(E⃗s − ⃗v ‖ x⃗)`, and a leg that
        bounds anything else prices Theorem 5.3's conditions for a vector
        the proof never covered — with both proofs verifying.

        `ExactL2`'s own `E` cannot be handed to a leg at all: it is written
        over `2(m1+ℓ)` and the leg's lift is `2(s1_take+ℓ)`, so
        `require_image` refuses it on shape. What survives that is an image
        over the *right* lift with the wrong rows — a caller writing the
        leg's `E` by hand and leaving off the rows that select `x⃗`. Only
        the width agreement sees it."""
        probe, image = _imaged(66)
        exact = probe.exact
        composed = exact.range_image()
        assert composed is not None
        self.assertEqual(composed.rows, image.rows + _DIGIT_COLS)
        exact.require_no_wraparound(_leg_at_bound(probe, 16, composed))

        short = _leg_image(probe, _leg_columns(probe)[:-_DIGIT_COLS])
        with self.assertRaisesRegex(ValueError, "eq. 61's composition"):
            exact.require_no_wraparound(_leg_at_bound(probe, 16, short))

    def test_a_leg_of_the_right_width_but_another_statement_is_refused(self) -> None:
        """A width alone cannot see this: an `E` over the leg's own lift with
        the right row count, about the wrong columns.

        The leg then proves a perfectly good bound on some other vector, the
        wraparound conditions are priced against it, and both proofs verify.
        Only comparing the image against `range_image()` catches it — which
        is why the check is on the statement and not on its size."""
        probe, _ = _imaged(74)
        exact = probe.exact
        required = exact.range_image()
        assert required is not None

        # Same lift, same rows, no rotations — a selection where the
        # composition has `[1, 6, 2]`.
        wrong_columns = _leg_image(probe, _leg_columns(probe))
        self.assertEqual(wrong_columns.rows, required.rows)
        with self.assertRaisesRegex(ValueError, "not eq. 61's composition"):
            exact.require_no_wraparound(_leg_at_bound(probe, 16, wrong_columns))

        # Same `E`, shifted `⃗v` — the half a rotation check would miss.
        shifted = AffineImage(
            matrix=required.matrix,
            offset=probe.ring.add(required.offset, probe.ring.one()),
        )
        with self.assertRaisesRegex(ValueError, "not eq. 61's composition"):
            exact.require_no_wraparound(_leg_at_bound(probe, 16, shifted))

    def test_undeclared_extra_rows_are_refused(self) -> None:
        """The width agreement's own job, now that the image check owns the
        leading rows: a leg may bound *more* than the composition, but only
        the `binary_cols` it was told about.

        This leg opens with `range_image()` exactly — so the image check
        passes — and then bounds one row nobody declared. Left unchecked,
        `span` and `c` would both be priced for a narrower vector than the
        proof actually covers."""
        probe, _ = _imaged(76)
        exact = probe.exact
        required = exact.range_image()
        assert required is not None
        padded = AffineImage(
            matrix=np.concatenate([required.matrix, required.matrix[:1]]),
            offset=np.concatenate([required.offset, required.offset[:1]]),
        )
        leg = _leg_at_bound(probe, 16, padded)
        with self.assertRaisesRegex(ValueError, "eq. 61's composition over"):
            exact.require_no_wraparound(leg)
        # Declared, it is the same leg and the conditions apply to it.
        exact.require_no_wraparound(leg, binary_cols=1)

    def test_an_imageless_leg_cannot_carry_an_imaged_statement(self) -> None:
        """The widths coincide here — the composition is `E`'s rows plus the
        digit row, and an imageless leg bounds `(s1‖x, m)`, which at this
        parameter point is the same count.

        So the width agreement passes and the statement is still wrong: the
        leg bounds the witness while eq. 66's inner product is written about
        `E⃗s − ⃗v`."""
        probe, _ = _imaged(75)
        exact = probe.exact
        required = exact.range_image()
        assert required is not None
        imageless = _leg_at_bound(probe, 16)
        self.assertEqual(imageless.bounded_width(), required.rows * probe.ring.d)
        with self.assertRaisesRegex(ValueError, "not eq. 61's composition"):
            exact.require_no_wraparound(imageless)

    def test_an_ell_infinity_leg_cannot_carry_an_exact_statement(self) -> None:
        """Fig. 10 runs two legs and only the ℓ2 one carries an exact
        statement: Theorem 5.3 states all three conditions over `B^(e)`
        alone, because they keep an integer identity proved mod `q` from
        wrapping and the ℓ∞ leg's eq. 52 is not one.

        Refused rather than converted. `‖·‖₂ ≤ √n·‖·‖_∞` over the 256-row
        projection costs `16·14 = 224` against the ℓ2 leg's `2√(256/26)·t`,
        some `21.8×`, while this parameter point clears Lemma 2.9 by `9.7×`
        — so the conversion would refuse honest configurations to check a
        condition they do not owe."""
        instance = _Instance(67)
        masking = lnp_fixture.bimodal(
            instance.ring, _WITNESS_COLS + _DIGIT_COLS, bound=LinfBound()
        )
        leg = ApproximateRange(instance.evaluation, masking).legs[0]
        with self.assertRaisesRegex(ValueError, "ℓ∞ leg"):
            instance.exact.require_no_wraparound(leg)

    def test_a_leg_over_a_different_ring_is_refused(self) -> None:
        """A bound proved over another modulus prices nothing here, and the
        conditions are all comparisons against *this* ring's `q`."""
        instance = _Instance(68)
        # Any `q ≡ 5 (mod 8)` will do — what the guard reads is object
        # identity, not the value.
        other_q = next(
            q
            for q in primes.find_nearest_split_primes(32.0, 2)
            if q not in lnp_fixture.SPLIT_Q
        )
        other = _Instance(69, ring=HostSplitRing((other_q,), lnp_fixture.D))
        with self.assertRaisesRegex(ValueError, "one ring"):
            instance.exact.require_no_wraparound(other.protocol.legs[0])

    def test_the_honest_parameter_point_has_room(self) -> None:
        """The conditions are not tight at any sane point — `q` is ~2^32 and
        the projection of a ternary witness is tiny — so the guard should
        never fire on the suite's own numbers.

        This is the production call shape: the leg the proof is actually
        built on, and nothing passed by hand. `B` is the ℓ2 bound Lemma 2.9
        concludes about `(s ‖ x⃗)` — 42323 here, against a `q/(41·c)` of
        409200, so 9.7× of room."""
        instance = _Instance(10)
        instance.exact.require_no_wraparound(instance.protocol.legs[0])


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


class ExactAffineImageTest(absltest.TestCase):
    """Eq. 53 at a general `E`: `I` is written about `E⃗s − ⃗v` rather than
    about `(s1, m)`.

    Same discipline as the rest of this suite — a round-trip cannot see a
    wrong statement, so the expansion of `T(E⃗s − ⃗v, E⃗s − ⃗v)` is pinned
    against the `E = I` blocks it must reproduce, and against what it is
    supposed to evaluate to.
    """

    def test_an_identity_image_rebuilds_the_e_equals_i_statement(self) -> None:
        """The four terms of the expansion collapse back to the diagonal
        scatter when `E = I` and `⃗v = 0`, block for block.

        The claim everything else here rests on: `σ₋₁(E)ᵀ·E` reduces to
        `e2[σ(c), c] = 1` on the witness columns, and the two linear terms
        and the constant all vanish. Nothing downstream can see a wrong
        expansion — both sides build `I` from this one object."""
        plain = _Instance(60)
        ring = plain.ring
        columns = lnp_fixture.identity_columns(_M1, plain.protocol.ell)
        identity = lnp_fixture.rotation_image(
            ring, SIGMA_ORDER * (_M1 + plain.protocol.ell), columns
        )
        spelled = _Instance(60, image=identity)

        for block, (want, got) in enumerate(
            zip(plain.exact.evaluations(), spelled.exact.evaluations(), strict=True)
        ):
            with self.subTest(block=block):
                np.testing.assert_array_equal(got, want)

    def test_the_norm_identity_is_about_the_image(self) -> None:
        """`I` vanishes for the digits of `β² − ‖E⃗s − ⃗v‖²` and not for the
        digits of `β² − ‖(s1, m)‖²`.

        A rotation that is *not* norm-preserving on the selected subset —
        it drops a column and repeats another — so the two slacks genuinely
        differ and a builder still reading the witness fails here."""
        ring = lnp_fixture.ring()
        probe = _Instance(61)
        ell = probe.protocol.ell
        columns = lnp_fixture.identity_columns(_M1, ell)
        # Read the first witness column twice and the message not at all.
        image = lnp_fixture.rotation_image(
            ring,
            SIGMA_ORDER * (_M1 + ell),
            [columns[0], columns[0], columns[1]],
            [0, 7, 3],
        )
        instance = _Instance(61, image=image)
        self.assertFalse(ring.constant_coeff(instance.values()).any())

        witness_digits = probe.digits
        self.assertFalse(np.array_equal(witness_digits, instance.digits))
        values = ring.constant_coeff(instance.values(witness_digits))
        self.assertTrue(values[1].any())

    def test_an_offset_moves_the_statement(self) -> None:
        """`⃗v` is part of what is bounded, so the digits that satisfy `I`
        under one offset do not satisfy it under another.

        The bound is widened because a ternary offset against a ternary
        image reaches coefficients of 2, so `‖E⃗s − ⃗v‖` genuinely exceeds
        the witness bound and `decompose` would refuse an honest witness."""
        ring = lnp_fixture.ring()
        columns = lnp_fixture.identity_columns(_M1, _ELL)
        offset = _ternary_offset(ring)
        wide = 2 * lnp_fixture.ternary_beta(ring, _WITNESS_COLS)
        shifted = _Instance(
            62,
            image=lnp_fixture.rotation_image(
                ring, SIGMA_ORDER * (_M1 + _ELL), columns, None, offset
            ),
            bound=wide,
        )
        self.assertFalse(ring.constant_coeff(shifted.values()).any())

        plain = _Instance(62, bound=wide)
        self.assertFalse(np.array_equal(plain.digits, shifted.digits))
        values = ring.constant_coeff(shifted.values(plain.digits))
        self.assertTrue(values[1].any())

    def test_an_honest_image_proof_verifies(self) -> None:
        """End to end: the range leg bounds eq. 61's `(E⃗s − ⃗v ‖ x⃗)` and the
        two exact functions ride along in the same inner proof.

        The leg carries the *composed* image, which is what makes the
        wraparound premise be about the vector `I` is written for."""
        probe, image = _imaged(63)
        instance = _Instance(63, image=image, range_image=probe.exact.range_image())
        proof, _ = instance.protocol.prove(
            instance.publics,
            s1=instance.s1,
            s2=instance.s2,
            message=instance.message,
            rng=instance.rng,
            transcript=_transcript(),
            evaluations=instance.exact.evaluations(),
        )
        ok, _ = instance.protocol.verify(
            instance.publics,
            t_a=instance.t_a,
            t_b=instance.t_b,
            proof=proof,
            transcript=_transcript(),
            evaluations=instance.exact.evaluations(),
        )
        self.assertTrue(ok)

    def test_the_wraparound_width_counts_the_image_rows(self) -> None:
        """Lemma 2.9's `c` is the width of what is *projected*. With an
        image that is `E`'s own row count plus the digit rows `range_image`
        appends, which need not match either half of the lift `E` was
        written over.

        Pinned through the reported `41·c` rather than by bracketing a bound
        between the two widths: at this point `B² < q` binds long before
        Lemma 2.9 does, so no `B` exists that Lemma 2.9 accepts for one
        width and refuses for the other. The number in the message is the
        `c` actually used, and it comes from the leg."""
        probe = _Instance(64)
        ring, ell = probe.ring, probe.protocol.ell
        columns = lnp_fixture.identity_columns(_M1, ell)
        # Built directly rather than through the fixture: a three-times-wider
        # image has three times the norm, which `decompose` would refuse
        # before the gate is ever asked.
        rows = 3 * len(columns)
        wide = ExactL2(
            probe.evaluation,
            _M1,
            ell,
            probe.bound,
            lnp_fixture.rotation_image(
                ring, SIGMA_ORDER * (_M1 + ell), columns * 3, [0] * rows
            ),
        )
        # Past every condition, so Lemma 2.9 — checked first — is the one
        # that speaks.
        for exact, width in (
            (wide, rows + _DIGIT_COLS),
            (probe.exact, _S1_COLS + ell),
        ):
            with self.subTest(width=width):
                leg = _leg_at_bound(probe, 500_000, exact.range_image())
                with self.assertRaisesRegex(
                    ValueError, rf"Lemma 2\.9.*/{41 * width * ring.d}\b"
                ):
                    exact.require_no_wraparound(leg)

    def test_a_misshapen_image_is_refused(self) -> None:
        ring = lnp_fixture.ring()
        instance = _Instance(65)
        width = SIGMA_ORDER * (_M1 + instance.protocol.ell)
        for message, image in {
            "one ring column per position": AffineImage(
                matrix=ring.zeros(2, width + 1), offset=ring.zeros(2)
            ),
            "one ring element per row": AffineImage(
                matrix=ring.zeros(2, width), offset=ring.zeros(3)
            ),
        }.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    ExactL2(
                        instance.evaluation,
                        _M1,
                        instance.protocol.ell,
                        instance.bound,
                        image,
                    )


if __name__ == "__main__":
    absltest.main()
