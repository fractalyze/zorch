# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""AdditiveReedSolomon: checked against implementation-independent invariants.

The encode is one `lax.ntt` (the compiler's LCH lowering, pinned by the
StableHLO binary-field NTT constraint suite), so the tests target the fold
family: the full fold chain of an honest codeword must land on a constant
(the IOPP terminal property — a tampered codeword must not), the verifier's
`fold_values` must agree with the prover's `fold` through `pair_indices`
(the seam contract), and `layer_positions` must walk each query onto the leg
of the next layer's pair that the fold lands on."""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.coding.additive_reed_solomon import AdditiveReedSolomon
from zorch.coding.foldable_code import FoldableCode
from zorch.coding.linear_code import LinearCode

F = zk_dtypes.binary_field_ghash


def _rand(n: int, rng: np.random.Generator, dtype: Any = F) -> Array:
    itemsize = np.dtype(dtype).itemsize
    return fnp.asarray(np.frombuffer(rng.bytes(n * itemsize), dtype=dtype))


def _bytes_equal(a: Array, b: Array) -> bool:
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


class AdditiveReedSolomonTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rng = np.random.default_rng(0)
        self.code = AdditiveReedSolomon(64, 4, F)
        self.num_rounds = self.code.message_len.bit_length() - 1

    def _fold_chain(self, cw: Array, betas: list[Array]) -> Array:
        for beta in betas:
            cw = self.code.fold(cw, beta)
        return cw

    def test_seam_conformance(self) -> None:
        self.assertIsInstance(self.code, LinearCode)
        self.assertIsInstance(self.code, FoldableCode)

    def test_value_equality_for_jit_keys(self) -> None:
        self.assertEqual(self.code, AdditiveReedSolomon(64, 4, F))
        self.assertNotEqual(self.code, AdditiveReedSolomon(32, 4, F))
        self.assertEqual(hash(self.code), hash(AdditiveReedSolomon(64, 4, F)))

    def test_honest_full_fold_is_constant(self) -> None:
        msg = _rand(self.code.message_len, self.rng)
        cw = self.code.encode(msg)
        betas = [_rand(1, self.rng)[0] for _ in range(self.num_rounds)]
        final = self._fold_chain(cw, betas)
        # The blowup survives the chain.
        self.assertEqual(final.shape[0], self.code.block_len >> self.num_rounds)
        self.assertTrue(bool(self.code.check_final(final, final[0])))

    def test_tampered_codeword_final_not_constant(self) -> None:
        msg = _rand(self.code.message_len, self.rng)
        cw_bytes = bytearray(np.asarray(self.code.encode(msg)).tobytes())
        cw_bytes[17] ^= 0x40
        cw = fnp.asarray(np.frombuffer(bytes(cw_bytes), dtype=F))
        betas = [_rand(1, self.rng)[0] for _ in range(self.num_rounds)]
        final = self._fold_chain(cw, betas)
        self.assertFalse(bool(self.code.check_final(final, final[0])))

    def test_fold_values_agrees_with_fold(self) -> None:
        num_rounds = self.num_rounds
        cw = self.code.encode(_rand(self.code.message_len, self.rng))
        for level in range(num_rounds):
            beta = _rand(1, self.rng)[0]
            folded = self.code.fold(cw, beta)
            pos = fnp.asarray(self.rng.integers(0, folded.shape[0], (8,)), fnp.int32)
            lo_i, hi_i = self.code.pair_indices(pos, level)
            got = self.code.fold_values(cw[lo_i], cw[hi_i], beta, pos, level)
            self.assertTrue(_bytes_equal(got, folded[pos]))
            cw = folded

    def test_layer_positions_walk_pairs(self) -> None:
        num_rounds = self.num_rounds
        pos = fnp.asarray(self.rng.integers(0, self.code.block_len, (8,)), fnp.int32)
        walk = self.code.layer_positions(pos, num_rounds)
        self.assertLen(walk, num_rounds)
        for i in range(num_rounds - 1):
            lo_i, hi_i = self.code.pair_indices(walk[i + 1], i + 1)
            landing = walk[i]
            member = (landing == lo_i) | (landing == hi_i)
            self.assertTrue(bool(fnp.all(member)))

    def test_pair_leaves_matches_pair_indices(self) -> None:
        cw = self.code.encode(_rand(self.code.message_len, self.rng))
        leaves = self.code.pair_leaves(cw)
        pos = fnp.asarray(self.rng.integers(0, leaves.shape[0], (8,)), fnp.int32)
        lo_i, hi_i = self.code.pair_indices(pos, 0)
        self.assertTrue(_bytes_equal(leaves[pos, 0], cw[lo_i]))
        self.assertTrue(_bytes_equal(leaves[pos, 1], cw[hi_i]))

    def test_encode_rejects_wrong_message_len(self) -> None:
        with self.assertRaises(ValueError):
            self.code.encode(_rand(63, self.rng))

    def test_rejects_non_binary_dtype(self) -> None:
        with self.assertRaises(TypeError):
            AdditiveReedSolomon(64, 4, zk_dtypes.koalabear_mont)

    def test_zero_layer_code_constructs(self) -> None:
        code = AdditiveReedSolomon(1, 1, F)
        msg = _rand(1, self.rng)
        self.assertTrue(_bytes_equal(code.encode(msg), msg))


if __name__ == "__main__":
    absltest.main()
