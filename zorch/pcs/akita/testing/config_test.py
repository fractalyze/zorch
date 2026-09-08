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
            decomposition=Decomposition(8, 4),
            rows=2,
            message_lens=(_D, _D + 5, 1),
        )

    def test_each_message_is_padded_to_whole_ring_elements(self) -> None:
        self.assertEqual(self.config.blocks_per_message, (1, 2, 1))
        self.assertEqual(self.config.blocks, 4)

    def test_module_width_is_one_column_per_digit_and_block(self) -> None:
        self.assertEqual(self.config.cols, 4 * 4)

    def test_columns_are_digit_major_and_cover_the_module(self) -> None:
        columns = [
            self.config.column(digit, block)
            for digit in range(self.config.decomposition.num_digits)
            for block in range(self.config.blocks)
        ]
        self.assertEqual(columns, list(range(self.config.cols)))

    def test_column_refuses_an_index_outside_the_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "digit"):
            self.config.column(4, 0)
        with self.assertRaisesRegex(ValueError, "block"):
            self.config.column(0, 4)

    def test_beta_is_the_digit_bound(self) -> None:
        self.assertEqual(self.config.beta_inf, self.config.decomposition.beta_inf)

    def test_refuses_an_empty_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one message"):
            AkitaConfig(SisProfile(_D, _Q), Decomposition(8, 4), 2, ())

    def test_refuses_a_module_with_no_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows"):
            AkitaConfig(SisProfile(_D, _Q), Decomposition(8, 4), 0, (_D,))


if __name__ == "__main__":
    absltest.main()
