# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Akita parameter point — the derived shapes, and the gates on the choices
that produce them.

The arithmetic here is what both sides of an opening index against, so it is
checked against the substrate it claims to follow (`gadget`'s digit interval,
lattice-frx's NTT-friendly walk) rather than against restated constants.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized
from lattice_frx import gadget

from zorch.pcs.akita.config import AkitaConfig, Decomposition, SisProfile

_D = 64
_Q = (34359753217, 34359754753)
# The narrowest outer decomposition this chain admits: what the outer tier
# shortens is a ring coefficient modulo `Q`, so the digit count follows `Q`.
_OUTER = Decomposition.covering(8, (_Q[0] * _Q[1]) // 2)


class SisProfileTest(absltest.TestCase):
    def test_nearest_walks_ntt_friendly_primes(self) -> None:
        profile = SisProfile.nearest(_D, 36, 3)
        self.assertLen(profile.moduli, 3)
        for q in profile.moduli:
            self.assertEqual((q - 1) % (2 * _D), 0)

    def test_modulus_is_the_chain_product(self) -> None:
        self.assertEqual(SisProfile(_D, _Q).modulus, _Q[0] * _Q[1])

    def test_ring_is_built_at_the_declared_point(self) -> None:
        ring = SisProfile(_D, _Q).ring
        self.assertEqual(ring.d, _D)
        self.assertEqual(ring.q_moduli, _Q)

    def test_refuses_a_non_power_of_two_degree(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            SisProfile(48, _Q)

    def test_refuses_a_modulus_the_degree_cannot_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "NTT-friendly"):
            SisProfile(_D, (_Q[0] + 2,))

    def test_refuses_an_empty_chain(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            SisProfile(_D, ())


class DecompositionTest(parameterized.TestCase):
    @parameterized.parameters((1, 1), (4, 3), (8, 5), (16, 2))
    def test_representable_endpoints_are_exactly_attainable(
        self, log_base: int, num_digits: int
    ) -> None:
        decomposition = Decomposition(log_base, num_digits)
        low, high = decomposition.representable
        for value in (low, high):
            self.assertEqual(
                gadget.recompose(
                    gadget.decompose(value, log_base, num_digits), log_base
                ),
                value,
            )
        for value in (low - 1, high + 1):
            with self.assertRaises(ValueError):
                gadget.decompose(value, log_base, num_digits)

    def test_beta_bounds_every_digit(self) -> None:
        decomposition = Decomposition(8, 4)
        low, high = decomposition.representable
        for value in (low, high, 0, high // 3):
            digits = gadget.decompose(value, 8, 4)
            self.assertLessEqual(max(abs(d) for d in digits), decomposition.beta_inf)

    @parameterized.parameters(1, 255, 2**31 - 1, 2**63 - 1)
    def test_covering_is_the_minimal_digit_count(self, magnitude: int) -> None:
        covering = Decomposition.covering(8, magnitude)
        low, high = covering.representable
        self.assertLessEqual(low, -magnitude)
        self.assertLessEqual(magnitude, high)
        if covering.num_digits > 1:
            _, shorter_high = Decomposition(8, covering.num_digits - 1).representable
            self.assertLess(shorter_high, magnitude)

    def test_covering_refuses_a_base_that_reaches_no_positive_value(self) -> None:
        # `{-1, 0}` digits cover only non-positive values, so the search for a
        # digit count has no answer to converge on.
        with self.assertRaisesRegex(ValueError, "base-2"):
            Decomposition.covering(1, 1)
        self.assertEqual(Decomposition.covering(1, 0), Decomposition(1, 1))

    def test_refuses_a_degenerate_base(self) -> None:
        with self.assertRaisesRegex(ValueError, "log_base"):
            Decomposition(0, 4)

    def test_refuses_an_empty_digit_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_digits"):
            Decomposition(8, 0)


class AkitaConfigTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = AkitaConfig(
            profile=SisProfile(_D, _Q),
            inner_decomposition=Decomposition(8, 4),
            outer_decomposition=_OUTER,
            inner_rows=2,
            outer_rows=3,
            message_lens=(_D, _D + 5, 1),
        )

    def test_each_message_is_padded_to_whole_ring_elements(self) -> None:
        self.assertEqual(self.config.blocks_per_message, (1, 2, 1))
        self.assertEqual(self.config.blocks, 4)

    def test_inner_width_is_one_column_per_digit_plane(self) -> None:
        self.assertEqual(self.config.inner_cols, 4)

    def test_images_are_one_module_vector_per_block(self) -> None:
        self.assertEqual(self.config.images, 4 * 2)

    def test_outer_width_is_one_column_per_digit_and_image(self) -> None:
        self.assertEqual(self.config.outer_cols, _OUTER.num_digits * 8)

    def test_witness_columns_are_block_major_and_cover_the_module(self) -> None:
        columns = [
            self.config.column(block, digit)
            for block in range(self.config.blocks)
            for digit in range(self.config.inner_cols)
        ]
        self.assertEqual(columns, list(range(self.config.blocks * 4)))

    def test_column_refuses_an_index_outside_the_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "digit"):
            self.config.column(0, 4)
        with self.assertRaisesRegex(ValueError, "block"):
            self.config.column(4, 0)

    def test_betas_are_the_two_digit_bounds(self) -> None:
        self.assertEqual(
            self.config.inner_beta_inf, self.config.inner_decomposition.beta_inf
        )
        self.assertEqual(self.config.outer_beta_inf, _OUTER.beta_inf)

    def test_refuses_an_empty_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one message"):
            AkitaConfig(SisProfile(_D, _Q), Decomposition(8, 4), _OUTER, 2, 3, ())

    def test_refuses_a_module_with_no_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "inner_rows"):
            AkitaConfig(SisProfile(_D, _Q), Decomposition(8, 4), _OUTER, 0, 3, (_D,))
        with self.assertRaisesRegex(ValueError, "outer_rows"):
            AkitaConfig(SisProfile(_D, _Q), Decomposition(8, 4), _OUTER, 2, 0, (_D,))

    def test_refuses_an_outer_decomposition_too_narrow_for_the_ring(self) -> None:
        # The inner tier's own exactness is `gadget`'s to refuse at commit time
        # — the field it decomposes is not part of this parameter point — but
        # `Q` is, so a digit count that cannot represent a ring coefficient is
        # caught where it was written down.
        with self.assertRaisesRegex(ValueError, "outer_decomposition reaches"):
            AkitaConfig(
                SisProfile(_D, _Q),
                Decomposition(8, 4),
                Decomposition(_OUTER.log_base, _OUTER.num_digits - 1),
                2,
                3,
                (_D,),
            )


if __name__ == "__main__":
    absltest.main()
