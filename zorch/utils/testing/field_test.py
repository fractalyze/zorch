# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import frx

frx.config.update("jax_enable_x64", True)  # binary_field_ghash is uint64-backed

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
import zk_dtypes  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402
from frx import Array, lax  # noqa: E402

from zorch.utils.field import (
    base_field,
    embed_base,
    is_binary_field,
    join_coeffs,
    naturals,
    split_coeffs,
)

KB = zk_dtypes.koalabear_mont
KX = zk_dtypes.koalabearx4_mont
GH = zk_dtypes.binary_field_ghash


class BaseFieldTest(absltest.TestCase):
    def test_prime_field_is_itself(self) -> None:
        self.assertEqual(base_field(KB), KB)

    def test_extension_gives_its_prime_subfield(self) -> None:
        self.assertEqual(base_field(KX), KB)


class NaturalsTest(absltest.TestCase):
    def test_values_and_base_dtype(self) -> None:
        # [0..n−1] in the base field, whether asked for the base or an extension.
        for dt in (KB, KX):
            got = naturals(5, dt)
            self.assertEqual(got.dtype, fnp.dtype(KB))
            self.assertEqual([int(v) for v in got], [0, 1, 2, 3, 4])

    def test_matches_per_element_embedding(self) -> None:
        # The base nodes an extension caller promotes must equal embedding each
        # integer into the extension directly (the pattern this replaces).
        got = naturals(6, KX).astype(KX)
        want = fnp.stack([fnp.array(i, KX) for i in range(6)])
        self.assertTrue(bool(fnp.all(got == want)))


class IsBinaryFieldTest(absltest.TestCase):
    def test_binary_is_true(self) -> None:
        self.assertTrue(is_binary_field(GH))

    def test_prime_and_extension_are_false(self) -> None:
        self.assertFalse(is_binary_field(KB))
        self.assertFalse(is_binary_field(KX))


def _gh(n: int, seed: int = 0) -> Array:
    """`n` GF(2^128) elements from random uint32 limbs (the ghash list constructor
    is vacuous on the CPU PJRT path; a limb bitcast is not)."""
    raw = np.random.default_rng(seed).integers(0, 1 << 32, size=(n, 4), dtype=np.uint32)
    return lax.bitcast_convert_type(fnp.asarray(raw), GH)


def _u64(a: Array) -> np.ndarray:
    """The packed 128-bit representation as `(..., 2)` uint64 lanes."""
    return np.asarray(lax.bitcast_convert_type(a, fnp.uint64))


class EmbedBaseTest(absltest.TestCase):
    def test_embeds_as_leading_coefficient(self) -> None:
        # base b -> (b, 0, 0, 0): coeff 0 is b, higher coeffs zero.
        base = fnp.array(np.array([3, 7, 11], dtype=np.uint32), dtype=KB)
        emb = embed_base(base, KX)
        self.assertEqual(emb.dtype, KX)
        coeffs = split_coeffs(emb)  # (3, 4)
        self.assertTrue(fnp.array_equal(coeffs[:, 0], base))
        self.assertTrue(bool(fnp.all(coeffs[:, 1:] == fnp.zeros((3, 3), KB))))

    def test_base_times_embedded_is_scalar_scaling(self) -> None:
        # A base scalar embedded then multiplied scales an extension exactly.
        b = embed_base(fnp.array(np.array([5], dtype=np.uint32), dtype=KB), KX)[0]
        x = join_coeffs(
            fnp.array(np.array([[1, 2, 3, 4]], dtype=np.uint32), dtype=KB), KX
        )[0]
        got = split_coeffs(b * x)
        want = split_coeffs(x) * fnp.array(5, KB)
        self.assertTrue(fnp.array_equal(got, want))

    def test_prime_dtype_is_unchanged(self) -> None:
        base = fnp.array(np.array([1, 2], dtype=np.uint32), dtype=KB)
        self.assertTrue(fnp.array_equal(embed_base(base, KB), base))

    def test_wrong_base_field_raises(self) -> None:
        wrong = fnp.array(np.array([1], dtype=np.uint32), dtype=zk_dtypes.babybear_mont)
        with self.assertRaises(ValueError):
            embed_base(wrong, KX)  # extension destination
        with self.assertRaises(ValueError):
            embed_base(wrong, KB)  # prime destination — must also reject foreign


class LimbViewTest(absltest.TestCase):
    """`split_coeffs` / `join_coeffs`: the view between an extension array and
    its base-field coefficients."""

    def _ext(self, shape: tuple[int, ...], start: int = 1) -> Array:
        n = int(np.prod(shape)) * 4  # KX is degree 4
        limbs = fnp.array(np.arange(start, start + n, dtype=np.uint64), dtype=KB)
        return join_coeffs(limbs.reshape(*shape, 4), KX)

    def test_round_trip_recovers_the_elements(self) -> None:
        values = self._ext((7,))
        rows = split_coeffs(values)
        self.assertEqual(rows.shape, (7, 4))
        self.assertEqual(rows.dtype, fnp.dtype(KB))
        self.assertTrue(fnp.array_equal(join_coeffs(rows, KX), values))

    def test_each_row_is_one_element(self) -> None:
        # The layout every caller depends on: row i holds element i's
        # coefficients, not a stride across the array.
        values = self._ext((4,))
        rows = split_coeffs(values)
        for i in range(4):
            one = join_coeffs(rows[i : i + 1], KX)
            self.assertTrue(fnp.array_equal(one.reshape(()), values[i]))

    def test_degree_comes_from_the_dtype(self) -> None:
        # The point of the helper: a caller writing `.reshape(n, 4)` by hand is
        # pinned to a degree-4 field, this is not.
        self.assertEqual(split_coeffs(self._ext((3,))).shape[-1], 4)

    def test_leading_axes_are_preserved(self) -> None:
        self.assertEqual(split_coeffs(self._ext((2, 5))).shape, (2, 5, 4))

    def test_a_base_array_passes_through(self) -> None:
        # Not (n, 1): a base element is one coefficient, and a length-1 axis
        # would imply an extension that is not there.
        values = naturals(6, KB)
        self.assertTrue(fnp.array_equal(split_coeffs(values), values))
        self.assertTrue(fnp.array_equal(join_coeffs(values, KB), values))

    def test_rejects_a_trailing_axis_that_is_not_the_degree(self) -> None:
        for bad in (3, 5, 8):
            with self.subTest(trailing=bad):
                limbs = naturals(bad, KB).reshape(1, bad)
                with self.assertRaisesRegex(ValueError, "must be the degree"):
                    join_coeffs(limbs, KX)


class BinaryFieldReduceAddTest(parameterized.TestCase):
    """GF(2^128) addition is a bitwise XOR of the 128-bit representation, so a host
    XOR-reduce over the packed uint64 lanes is an exact, reduce-free oracle for the
    native `fnp.sum`. This pins the binary-field reduce-add lowering that
    `ring_switch.inner_product` — and any field reduction — relies on: a wheel that
    regressed it (cf. the historical CUDA reduce-add SIGSEGV, zorch#400) would fail
    here rather than deep in a prover. Run under `jit`, where the full-reduction
    (0-d) case lowers cleanly."""

    @parameterized.named_parameters(
        dict(testcase_name="1d_even_full", shape=(8,), axis=None),
        dict(testcase_name="1d_odd_full", shape=(5,), axis=None),
        dict(testcase_name="1d_odd_axis0", shape=(7,), axis=0),
        dict(testcase_name="2d_axis0_oddlead", shape=(5, 4), axis=0),
        dict(testcase_name="2d_axis1", shape=(3, 4), axis=1),
        dict(testcase_name="2d_axis0_single", shape=(1, 4), axis=0),
    )
    def test_jnp_sum_matches_xor_reduce(
        self, shape: tuple[int, ...], axis: int | None
    ) -> None:
        n = int(np.prod(shape))
        x = _gh(n).reshape(shape)
        got = _u64(frx.jit(lambda a: fnp.sum(a, axis=axis))(x))
        red = tuple(range(len(shape))) if axis is None else axis
        want = np.bitwise_xor.reduce(_u64(x), axis=red)
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(bool(np.array_equal(got, want)))


if __name__ == "__main__":
    absltest.main()
