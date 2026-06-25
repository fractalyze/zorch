# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Reed-Solomon encode and the FRI fold: checked against independent oracles.

The oracle never reuses the encoder. The NTT evaluation domain is recovered
straight from `lax.fft` of an impulse (NTT(e_1)_j = w^j), then the codeword is
compared to a Horner evaluation of the message polynomial on that domain — a
path that shares no code with pad + NTT. The fold is checked the same way: a
hand-computed linear-polynomial fold plus the fold/encode commute invariant.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array, lax

from zorch.coding.foldable_code import FoldableCode, KFoldableCode
from zorch.coding.linear_code import LinearCode
from zorch.coding.reed_solomon import (
    BitReversedReedSolomon,
    ReedSolomon,
    eval_domain,
    fri_fold,
    fri_fold_k_values,
    fri_fold_values,
)
from zorch.testkit.random_field import rand_ext_field, rand_field

F = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


def _bit_reverse_perm(n: int) -> Array:
    """The bit-reversal permutation of [0, n), built bit-by-bit so it shares
    no code with the implementation's `lax.bit_reverse`."""
    bits = n.bit_length() - 1
    rev = [0] * n
    for i in range(n):
        for b in range(bits):
            rev[i] |= ((i >> b) & 1) << (bits - 1 - b)
    return jnp.array(rev, dtype=jnp.int32)


def _domain(n: int, dtype: Any) -> Array:
    """The order-n NTT evaluation domain [w^0, ..., w^{n-1}] for the canonical
    root, recovered independently of the encoder: NTT(e_1)_j = w^j."""
    e1 = jnp.zeros((n,), dtype).at[1].set(jnp.ones((), dtype))
    return lax.ntt(e1, ntt_type="NTT", ntt_length=n)


def _horner(coeffs: Array, points: Array) -> Array:
    """Evaluate the polynomial with `coeffs` at every point in `points`."""
    acc = points * jnp.zeros((), points.dtype)
    for i in range(coeffs.shape[0] - 1, -1, -1):
        acc = acc * points + coeffs[i]
    return acc


class ReedSolomonTest(absltest.TestCase):
    def test_implements_linear_code_protocol(self) -> None:
        rs = ReedSolomon(message_len=4, blowup=2, dtype=F)
        self.assertIsInstance(rs, LinearCode)
        self.assertIsInstance(rs, FoldableCode)
        self.assertEqual(rs.message_len, 4)
        self.assertEqual(rs.block_len, 8)
        self.assertEqual(rs.dtype, F)

    def test_fold_values_matches_whole_codeword_fold(self) -> None:
        # The invariant the BaseFold verifier relies on: folding opened pairs
        # at queried positions equals the whole-codeword fold at those positions.
        k, blowup = 8, 2
        rs = ReedSolomon(k, blowup, F)
        cw = rs.encode(rand_field(3, (k,), F))
        beta = rand_field(4, (), F)
        half = rs.block_len // 2
        positions = jnp.array([0, 3, half - 1])
        folded = rs.fold(cw, beta)
        got = rs.fold_values(cw[positions], cw[positions + half], beta, positions, 0)
        self.assertTrue(bool(jnp.all(got == folded[positions])))

    def test_fold_values_matches_whole_codeword_fold_on_coset(self) -> None:
        # Same invariant on a coset, one level down: both sides must apply the
        # level's shift coset_shift^(2^level), not the code's base shift.
        k, blowup = 8, 2
        h = jnp.array(3, dtype=F)
        rs = ReedSolomon(k, blowup, F, coset_shift=h)
        layer1 = rs.fold(rs.encode(rand_field(8, (k,), F)), rand_field(9, (), F))
        beta = rand_field(10, (), F)
        half = layer1.shape[0] // 2
        positions = jnp.array([0, 2, half - 1])
        folded = rs.fold(layer1, beta)
        got = rs.fold_values(
            layer1[positions], layer1[positions + half], beta, positions, 1
        )
        self.assertTrue(bool(jnp.all(got == folded[positions])))

    def test_fri_fold_k_values_k2_equals_butterfly(self) -> None:
        # The k-ary fold's k=2 case must equal the conjugate-pair butterfly: for
        # points (x, -x), interpolation and fri_fold_values are the same map
        # (independent formulas — Lagrange vs the closed-form butterfly).
        x = rand_field(1, (), F)
        fx, fnx, beta = rand_field(2, (), F), rand_field(3, (), F), rand_field(4, (), F)
        group = jnp.stack([fx, fnx])
        points = jnp.stack([x, -x])
        self.assertEqual(
            fri_fold_k_values(group, beta, points), fri_fold_values(fx, fnx, beta, x)
        )

    def test_fri_fold_k_values_matches_lagrange_oracle(self) -> None:
        # A folding factor > 2 interpolates; check against an independent
        # product-form Lagrange evaluation (k=4, extension-field values).
        k = 4
        group = rand_ext_field(7, (k,), F, EF)
        points = rand_field(8, (k,), F)
        beta = rand_field(9, (), F)

        def oracle(group: Array, points: Array, r: Array) -> Array:
            acc = jnp.zeros((), group.dtype)
            for i in range(k):
                num = jnp.ones((), points.dtype)
                den = jnp.ones((), points.dtype)
                for j in range(k):
                    if j != i:
                        num = num * (r - points[j])
                        den = den * (points[i] - points[j])
                acc = acc + group[i] * (num / den)
            return acc

        self.assertEqual(
            fri_fold_k_values(group, beta, points), oracle(group, points, beta)
        )

    def test_check_final_accepts_only_the_constant_claim(self) -> None:
        rs = ReedSolomon(message_len=1, blowup=4, dtype=F)
        claim = rand_field(5, (), F)
        good = jnp.full((rs.block_len,), claim, F)
        self.assertTrue(bool(rs.check_final(good, claim)))
        bad = good.at[1].set(claim + jnp.ones((), F))
        self.assertFalse(bool(rs.check_final(bad, claim)))

    def test_encode_matches_polynomial_evaluation(self) -> None:
        k, blowup = 4, 4
        rs = ReedSolomon(k, blowup, F)
        coeffs = rand_field(1, (k,), F)
        want = _horner(coeffs, _domain(k * blowup, F))
        self.assertTrue(bool(jnp.all(rs.encode(coeffs) == want)))

    def test_codeword_is_low_degree(self) -> None:
        k, blowup = 8, 2
        rs = ReedSolomon(k, blowup, F)
        coeffs = rand_field(2, (k,), F)
        rec = lax.ntt(rs.encode(coeffs), ntt_type="INTT", ntt_length=k * blowup)
        self.assertTrue(bool(jnp.all(rec[:k] == coeffs)))
        self.assertTrue(bool(jnp.all(rec[k:] == jnp.zeros(k * blowup - k, F))))

    def test_encode_is_linear(self) -> None:
        rs = ReedSolomon(4, 2, F)
        a, b = rand_field(3, (4,), F), rand_field(4, (4,), F)
        self.assertTrue(bool(jnp.all(rs.encode(a + b) == rs.encode(a) + rs.encode(b))))

    def test_encode_batched_rows(self) -> None:
        k, blowup, rows = 4, 2, 3
        rs = ReedSolomon(k, blowup, F)
        msg = rand_field(5, (rows, k), F)
        cw = rs.encode(msg)
        self.assertEqual(cw.shape, (rows, k * blowup))
        for r in range(rows):
            self.assertTrue(bool(jnp.all(cw[r] == rs.encode(msg[r]))))

    def test_encode_batched_rows_extension_field(self) -> None:
        # The LinearCode seam promises leading batch axes ride through on any
        # field dtype: a [rows, k] extension-field message must match the 1-D
        # encode row by row, coset included.
        k, blowup, rows = 4, 2, 3
        shift = jnp.array(3, dtype=EF)
        for coset_shift in (None, shift):
            rs = ReedSolomon(k, blowup, EF, coset_shift=coset_shift)
            msg = rand_ext_field(25, (rows, k), F, EF)
            cw = rs.encode(msg)
            self.assertEqual(cw.shape, (rows, k * blowup))
            for r in range(rows):
                self.assertTrue(bool(jnp.all(cw[r] == rs.encode(msg[r]))))

    def test_coset_encode_matches_shifted_evaluation(self) -> None:
        k, blowup = 4, 2
        shift = jnp.array(3, dtype=F)
        rs = ReedSolomon(k, blowup, F, coset_shift=shift)
        coeffs = rand_field(6, (k,), F)
        want = _horner(coeffs, shift * _domain(k * blowup, F))
        self.assertTrue(bool(jnp.all(rs.encode(coeffs) == want)))

    def test_domain_matches_independent_recovery(self) -> None:
        k, blowup = 4, 2
        plain = ReedSolomon(k, blowup, F)
        self.assertTrue(bool(jnp.all(plain.domain() == _domain(k * blowup, F))))
        shift = jnp.array(3, dtype=F)
        coset = ReedSolomon(k, blowup, F, coset_shift=shift)
        self.assertTrue(bool(jnp.all(coset.domain() == shift * _domain(k * blowup, F))))

    def test_wrong_message_length_raises(self) -> None:
        rs = ReedSolomon(4, 2, F)
        with self.assertRaises(ValueError):
            rs.encode(rand_field(7, (5,), F))

    def test_non_power_of_two_raises(self) -> None:
        with self.assertRaises(ValueError):
            ReedSolomon(message_len=3, blowup=2, dtype=F)
        with self.assertRaises(ValueError):
            ReedSolomon(message_len=4, blowup=3, dtype=F)

    def test_value_equality_across_fresh_instances(self) -> None:
        # A code seats in static jit-zone keys (#214): same-config builds must
        # compare and hash equal regardless of instance identity, and every
        # config axis — geometry, dtype, coset shift, layout — must break it.
        a = ReedSolomon(message_len=8, blowup=2, dtype=F)
        b = ReedSolomon(message_len=8, blowup=2, dtype=F)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, ReedSolomon(message_len=8, blowup=4, dtype=F))
        self.assertNotEqual(a, ReedSolomon(message_len=8, blowup=2, dtype=EF))
        shift = jnp.array(3, dtype=F)
        s = ReedSolomon(message_len=8, blowup=2, dtype=F, coset_shift=shift)
        t = ReedSolomon(message_len=8, blowup=2, dtype=F, coset_shift=shift)
        self.assertEqual(s, t)
        self.assertEqual(hash(s), hash(t))
        self.assertNotEqual(a, s)
        br_a = BitReversedReedSolomon(message_len=8, blowup=2, dtype=F)
        br_b = BitReversedReedSolomon(message_len=8, blowup=2, dtype=F)
        self.assertEqual(br_a, br_b)
        self.assertEqual(hash(br_a), hash(br_b))
        self.assertNotEqual(br_a, a)


class ReedSolomonKaryTest(absltest.TestCase):
    def test_implements_k_foldable_protocol(self) -> None:
        rs = ReedSolomon(message_len=16, blowup=4, dtype=F, fold_factor=4)
        self.assertIsInstance(rs, KFoldableCode)
        self.assertEqual(rs.fold_factor, 4)

    def test_fold_factor_must_be_power_of_two(self) -> None:
        with self.assertRaises(ValueError):
            ReedSolomon(message_len=16, blowup=4, dtype=F, fold_factor=3)

    def test_fold_factor_breaks_value_equality(self) -> None:
        # fold_factor changes the fold map, so it must be a config axis: same
        # geometry but different factor must not compare/hash equal (#214).
        a = ReedSolomon(8, 2, F, fold_factor=2)
        b = ReedSolomon(8, 2, F, fold_factor=4)
        self.assertNotEqual(a, b)
        self.assertEqual(a, ReedSolomon(8, 2, F))  # default factor is 2
        self.assertEqual(hash(b), hash(ReedSolomon(8, 2, F, fold_factor=4)))

    def test_group_leaves_k2_equals_pair_leaves(self) -> None:
        # At k=2 the k-group layout is the conjugate pair: row p = (cw[p], cw[p+n/2]).
        rs = ReedSolomon(8, 2, F)  # default fold_factor 2
        cw = rs.encode(rand_field(30, (8,), F))
        self.assertTrue(bool(jnp.all(rs.group_leaves(cw) == rs.pair_leaves(cw))))

    def test_group_indices_k2_equals_pair_indices(self) -> None:
        rs = ReedSolomon(8, 2, F)
        positions = jnp.array([0, 2, 5])
        lo, hi = rs.pair_indices(positions, 0)
        g0, g1 = rs.group_indices(positions, 0)
        self.assertTrue(bool(jnp.all(g0 == lo)))
        self.assertTrue(bool(jnp.all(g1 == hi)))

    def test_group_leaves_groups_kth_root_coset(self) -> None:
        # Row p holds the k entries a sub-layer apart {p, p+n/k, ..., p+(k-1)n/k}
        # — the k-th-root coset that folds to position p.
        k = 4
        rs = ReedSolomon(16, 4, F, fold_factor=k)  # block_len 64
        cw = rs.encode(rand_field(31, (16,), F))
        leaves = rs.group_leaves(cw)
        n = rs.block_len
        sub = n // k
        self.assertEqual(leaves.shape, (sub, k))
        for p in (0, 1, sub - 1):
            for m in range(k):
                self.assertEqual(leaves[p, m], cw[p + m * sub])

    def test_fold_group_k2_equals_binary_butterfly(self) -> None:
        # F1: the k-ary Lagrange fold at k=2 is the same map as the conjugate
        # butterfly (kept separate only for cost), so fold_group must match fold.
        rs = ReedSolomon(8, 2, F)
        cw = rs.encode(rand_field(32, (8,), F))
        beta = rand_field(33, (), F)
        self.assertTrue(bool(jnp.all(rs.fold_group(cw, beta) == rs.fold(cw, beta))))

    def test_fold_group_rejects_short_layer(self) -> None:
        # Fail loud at the seam boundary (like fri_fold's n<2 guard) instead of
        # an opaque reshape error: a layer too short to form a full k-group is
        # rejected.
        rs = ReedSolomon(16, 4, F, fold_factor=4)  # block_len 64, k=4
        with self.assertRaises(ValueError):
            rs.fold_group(rand_field(70, (2,), F), rand_field(71, (), F))

    def test_fold_group_encode_commute(self) -> None:
        # Independent oracle: a k-ary fold by beta combines the k coefficient
        # cosets as sum_m beta^m * p[m::k], on the x^k domain.
        k = 4
        L = 16
        p = rand_field(34, (L,), F)
        beta = rand_field(35, (), F)
        cw = ReedSolomon(L, 1, F, fold_factor=k).encode(p)
        folded = ReedSolomon(L, 1, F, fold_factor=k).fold_group(cw, beta)
        p_fold = jnp.zeros((L // k,), F)
        bp = jnp.ones((), F)
        for m in range(k):
            p_fold = p_fold + bp * p[m::k]
            bp = bp * beta
        expected = ReedSolomon(L // k, 1, F, fold_factor=k).encode(p_fold)
        self.assertEqual(folded.shape, (L // k,))
        self.assertTrue(bool(jnp.all(folded == expected)))

    def test_fold_group_encode_commute_on_coset(self) -> None:
        # Same commute on a coset h*H: the folded codeword lives on the (x^k)
        # domain h^k*H^k, so the next layer's code carries shift h^k.
        k = 4
        L = 16
        h = jnp.array(3, dtype=F)
        p = rand_field(36, (L,), F)
        beta = rand_field(37, (), F)
        cw = ReedSolomon(L, 1, F, coset_shift=h, fold_factor=k).encode(p)
        folded = ReedSolomon(L, 1, F, coset_shift=h, fold_factor=k).fold_group(cw, beta)
        p_fold = jnp.zeros((L // k,), F)
        bp = jnp.ones((), F)
        for m in range(k):
            p_fold = p_fold + bp * p[m::k]
            bp = bp * beta
        hk = h
        for _ in range(k.bit_length() - 1):
            hk = hk * hk  # h^k
        expected = ReedSolomon(L // k, 1, F, coset_shift=hk, fold_factor=k).encode(
            p_fold
        )
        self.assertTrue(bool(jnp.all(folded == expected)))

    def test_fold_group_values_matches_whole_codeword_fold(self) -> None:
        # The verifier invariant: folding an opened k-group at queried positions
        # equals the whole-codeword fold at those positions.
        k = 4
        rs = ReedSolomon(16, 4, F, fold_factor=k)  # block_len 64
        cw = rs.encode(rand_field(38, (16,), F))
        beta = rand_field(39, (), F)
        positions = jnp.array([0, 3, rs.block_len // k - 1])
        idx = jnp.stack(rs.group_indices(positions, 0), axis=-1)
        folded = rs.fold_group(cw, beta)
        got = rs.fold_group_values(cw[idx], beta, positions, 0)
        self.assertTrue(bool(jnp.all(got == folded[positions])))

    def test_fold_group_values_matches_whole_codeword_fold_on_coset(self) -> None:
        # Same invariant one level down on a coset: both sides must apply the
        # level's shift h^(k^level), not the base shift.
        k = 4
        h = jnp.array(3, dtype=F)
        rs = ReedSolomon(16, 4, F, coset_shift=h, fold_factor=k)  # block_len 64
        layer1 = rs.fold_group(
            rs.encode(rand_field(50, (16,), F)), rand_field(51, (), F)
        )
        beta = rand_field(52, (), F)
        sub = layer1.shape[0] // k
        positions = jnp.array([0, 1, sub - 1])
        idx = jnp.stack(rs.group_indices(positions, 1), axis=-1)
        folded = rs.fold_group(layer1, beta)
        got = rs.fold_group_values(layer1[idx], beta, positions, 1)
        self.assertTrue(bool(jnp.all(got == folded[positions])))

    def test_fold_group_values_extension_field(self) -> None:
        # k-ary fold of an EF-valued layer; the x-coordinates stay base-field.
        k = 4
        rs = ReedSolomon(16, 4, EF, fold_factor=k)
        cw = rs.encode(rand_ext_field(53, (16,), F, EF))
        beta = rand_ext_field(54, (), F, EF)
        positions = jnp.array([0, 5, rs.block_len // k - 1])
        idx = jnp.stack(rs.group_indices(positions, 0), axis=-1)
        folded = rs.fold_group(cw, beta)
        got = rs.fold_group_values(cw[idx], beta, positions, 0)
        self.assertTrue(bool(jnp.all(got == folded[positions])))

    def test_group_layer_positions_chain(self) -> None:
        # k-ary query-index chain: a_i = q mod (n / k^{i+1}).
        k = 4
        rs = ReedSolomon(16, 4, F, fold_factor=k)  # block_len 64
        q = jnp.array([37, 6])
        a = rs.group_layer_positions(q, 3)
        self.assertTrue(bool(jnp.all(a[0] == q % (rs.block_len // k))))
        self.assertTrue(bool(jnp.all(a[1] == a[0] % (rs.block_len // k**2))))
        self.assertTrue(bool(jnp.all(a[2] == a[1] % (rs.block_len // k**3))))

    def test_group_layer_positions_k2_equals_binary(self) -> None:
        rs = ReedSolomon(8, 2, F)
        q = jnp.array([13, 6])
        self.assertEqual(len(rs.group_layer_positions(q, 3)), 3)
        for kary, binary in zip(
            rs.group_layer_positions(q, 3), rs.layer_positions(q, 3)
        ):
            self.assertTrue(bool(jnp.all(kary == binary)))


class BitReversedReedSolomonTest(absltest.TestCase):
    def test_implements_foldable_code_protocol(self) -> None:
        code = BitReversedReedSolomon(message_len=4, blowup=2, dtype=F)
        self.assertIsInstance(code, LinearCode)
        self.assertIsInstance(code, FoldableCode)
        self.assertEqual(code.message_len, 4)
        self.assertEqual(code.block_len, 8)
        self.assertEqual(code.dtype, F)

    def test_encode_is_bit_reversed_natural_codeword(self) -> None:
        k, blowup = 8, 2
        msg = rand_field(11, (k,), F)
        nat = ReedSolomon(k, blowup, F).encode(msg)
        br = BitReversedReedSolomon(k, blowup, F).encode(msg)
        self.assertTrue(bool(jnp.all(br == nat[_bit_reverse_perm(k * blowup)])))

    def test_domain_matches_codeword_layout(self) -> None:
        # domain()[j] must be the evaluation point of codeword[j], so the
        # bit-reversed code's domain rides the same permutation as encode.
        k, blowup = 4, 2
        shift = jnp.array(3, dtype=F)
        nat = ReedSolomon(k, blowup, F, coset_shift=shift)
        br = BitReversedReedSolomon(k, blowup, F, coset_shift=shift)
        self.assertTrue(
            bool(jnp.all(br.domain() == nat.domain()[_bit_reverse_perm(k * blowup)]))
        )

    def test_fold_commutes_with_bit_reversal(self) -> None:
        # br.fold on the bit-reversed codeword is the bit-reversal of the
        # natural fold — the layout is fold-stable, layer by layer.
        k, blowup = 8, 2
        nat = ReedSolomon(k, blowup, F)
        br = BitReversedReedSolomon(k, blowup, F)
        cw = nat.encode(rand_field(12, (k,), F))
        beta0, beta1 = rand_field(13, (), F), rand_field(14, (), F)
        n = k * blowup
        layer1_nat = nat.fold(cw, beta0)
        layer1_br = br.fold(cw[_bit_reverse_perm(n)], beta0)
        self.assertTrue(
            bool(jnp.all(layer1_br == layer1_nat[_bit_reverse_perm(n // 2)]))
        )
        layer2_nat = nat.fold(layer1_nat, beta1)
        layer2_br = br.fold(layer1_br, beta1)
        self.assertTrue(
            bool(jnp.all(layer2_br == layer2_nat[_bit_reverse_perm(n // 4)]))
        )

    def test_fold_commutes_with_bit_reversal_on_coset(self) -> None:
        k, blowup = 8, 2
        h = jnp.array(3, dtype=F)
        nat = ReedSolomon(k, blowup, F, coset_shift=h)
        br = BitReversedReedSolomon(k, blowup, F, coset_shift=h)
        cw = nat.encode(rand_field(15, (k,), F))
        beta = rand_field(16, (), F)
        n = k * blowup
        got = br.fold(cw[_bit_reverse_perm(n)], beta)
        want = nat.fold(cw, beta)[_bit_reverse_perm(n // 2)]
        self.assertTrue(bool(jnp.all(got == want)))

    def test_fold_extension_field_codeword(self) -> None:
        # The basefold open folds RLC'd extension-field codewords through the
        # same seam; the fold's x-coordinates stay base-field.
        k, blowup = 8, 2
        nat = ReedSolomon(k, blowup, EF)
        br = BitReversedReedSolomon(k, blowup, EF)
        cw = nat.encode(rand_ext_field(17, (k,), F, EF))
        beta = rand_ext_field(18, (), F, EF)
        n = k * blowup
        got = br.fold(cw[_bit_reverse_perm(n)], beta)
        want = nat.fold(cw, beta)[_bit_reverse_perm(n // 2)]
        self.assertTrue(bool(jnp.all(got == want)))

    def test_fold_values_matches_whole_codeword_fold(self) -> None:
        # The seam contract: with (l, h) = pair_indices(p, level),
        # fold(layer, beta)[p] == fold_values(layer[l], layer[h], beta, p, level).
        k, blowup = 8, 2
        br = BitReversedReedSolomon(k, blowup, F)
        cw = br.encode(rand_field(19, (k,), F))
        beta = rand_field(20, (), F)
        half = br.block_len // 2
        positions = jnp.array([0, 3, half - 1])
        lo_idx, hi_idx = br.pair_indices(positions, 0)
        folded = br.fold(cw, beta)
        got = br.fold_values(cw[lo_idx], cw[hi_idx], beta, positions, 0)
        self.assertTrue(bool(jnp.all(got == folded[positions])))

    def test_fold_values_matches_whole_codeword_fold_on_coset(self) -> None:
        k, blowup = 8, 2
        h = jnp.array(3, dtype=F)
        br = BitReversedReedSolomon(k, blowup, F, coset_shift=h)
        layer1 = br.fold(br.encode(rand_field(21, (k,), F)), rand_field(22, (), F))
        beta = rand_field(23, (), F)
        half = layer1.shape[0] // 2
        positions = jnp.array([0, 2, half - 1])
        lo_idx, hi_idx = br.pair_indices(positions, 1)
        folded = br.fold(layer1, beta)
        got = br.fold_values(layer1[lo_idx], layer1[hi_idx], beta, positions, 1)
        self.assertTrue(bool(jnp.all(got == folded[positions])))

    def test_pair_indices_are_adjacent(self) -> None:
        br = BitReversedReedSolomon(8, 2, F)
        positions = jnp.array([0, 2, 5])
        lo_idx, hi_idx = br.pair_indices(positions, 0)
        self.assertTrue(bool(jnp.all(lo_idx == positions * 2)))
        self.assertTrue(bool(jnp.all(hi_idx == positions * 2 + 1)))

    def test_layer_positions_shift_chain(self) -> None:
        # A query at q tracks the value landing at q >> (i+1) after layer i.
        br = BitReversedReedSolomon(8, 2, F)
        q = jnp.array([13, 6])
        a = br.layer_positions(q, 3)
        self.assertTrue(bool(jnp.all(a[0] == q >> 1)))
        self.assertTrue(bool(jnp.all(a[1] == q >> 2)))
        self.assertTrue(bool(jnp.all(a[2] == q >> 3)))

    def test_check_final_accepts_only_the_constant_claim(self) -> None:
        br = BitReversedReedSolomon(message_len=1, blowup=4, dtype=F)
        claim = rand_field(24, (), F)
        good = jnp.full((br.block_len,), claim, F)
        self.assertTrue(bool(br.check_final(good, claim)))
        bad = good.at[1].set(claim + jnp.ones((), F))
        self.assertFalse(bool(br.check_final(bad, claim)))


class FriValuesTest(absltest.TestCase):
    def test_eval_domain_is_negation_symmetric(self) -> None:
        n = 8
        d = eval_domain(F, n)  # [ω⁰..ω⁷]
        self.assertEqual(d.shape, (n,))
        half = n // 2
        # ω^{j+half} = -ω^j  (second half negates the first)
        self.assertTrue(bool(jnp.all(d[half:] + d[:half] == jnp.zeros(half, F))))

    def test_eval_domain_order_one_is_unity(self) -> None:  # trivial subgroup {1}
        self.assertTrue(bool(jnp.all(eval_domain(F, 1) == jnp.ones(1, F))))

    def test_eval_domain_rejects_non_power_of_two(self) -> None:
        with self.assertRaises(ValueError):
            eval_domain(F, 6)

    def test_fold_values_recovers_linear_poly(self) -> None:
        # Independent oracle: for f(X)=a+bX, f(x)=a+bx and f(-x)=a-bx, so the
        # conjugate fold must be exactly a + β·b — no division to hand-compute,
        # and it shares no expression with fri_fold_values.
        a, b, x, beta = (jnp.array(v, dtype=F) for v in (3, 5, 4, 6))
        fx = a + b * x
        fnx = a - b * x
        got = fri_fold_values(fx, fnx, beta, x)
        self.assertTrue(bool(got == a + beta * b))


class FriFoldCommuteTest(absltest.TestCase):
    def test_fri_fold_rejects_length_one_codeword(self) -> None:
        # n=1 has no conjugate pair to fold: half=0 broadcasts the empty
        # first-half against the length-1 second-half to a silent empty result.
        # Reject it so misuse fails loudly instead of returning garbage.
        with self.assertRaises(ValueError):
            fri_fold(jnp.ones(1, F), jnp.array(6, dtype=F))

    def test_fold_encode_commute(self) -> None:
        # fold(encode(p), β) == encode(p_even + β·p_odd) on the squared domain.
        k = 4
        L = 1 << k
        p = rand_field(40, (L,), F)  # polynomial coefficients
        beta = rand_field(41, (), F)
        cw = ReedSolomon(L, 1, F).encode(p)  # blowup=1: DFT on order-L domain
        folded = fri_fold(cw, beta)  # (L/2,)
        p_fold = p[0::2] + beta * p[1::2]  # even + β·odd
        expected = ReedSolomon(L // 2, 1, F).encode(p_fold)
        self.assertEqual(folded.shape, (L // 2,))
        self.assertTrue(bool(jnp.all(folded == expected)))

    def test_fold_encode_commute_on_coset(self) -> None:
        # Same commute on a coset h·H: the fold's x-coordinates are h·d, and the
        # folded codeword lives on the squared coset h²·H² — so `fri_fold` must
        # be told the shift, and the next layer's code carries shift h².
        k = 4
        L = 1 << k
        h = jnp.array(3, dtype=F)
        p = rand_field(42, (L,), F)
        beta = rand_field(43, (), F)
        cw = ReedSolomon(L, 1, F, coset_shift=h).encode(p)
        folded = fri_fold(cw, beta, shift=h)
        p_fold = p[0::2] + beta * p[1::2]
        expected = ReedSolomon(L // 2, 1, F, coset_shift=h * h).encode(p_fold)
        self.assertTrue(bool(jnp.all(folded == expected)))


if __name__ == "__main__":
    absltest.main()
