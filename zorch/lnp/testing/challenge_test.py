# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LNP challenge space — structure, the η gate, and the byte-stream contract.

Structural correctness without goldens, like `ajtai_test`: every property the
challenge space definition states (σ₋₁-invariance, the ℓ∞ bound, the
operator-norm gate, invertibility in the partial-split ring) is checked on
sampled challenges, and the gate's ℓ1 quantity is recomputed here with an
independent pure-Python negacyclic product rather than through the module's
own arithmetic. Streams are seeded numpy bytes; crafted streams pin the
fall-through and exhaustion branches deterministically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest
from lattice_frx.sampler import uniform_bytes_needed, uniform_from_bytes
from lattice_frx.split_ring import HostSplitRing

from zorch.lnp.challenge import challenge_bytes_needed, challenge_from_bytes

# The paper's Figure-3 instantiation (eprint 2022/284): d=128, κ=2, η=59,
# k=32 — the parameter point the LNP layer pins.
_D = 128
_KAPPA = 2
_ETA = 59
_K = 32

# One candidate's byte block: d/2 uniform draws at modulus 2κ+1.
_MODULUS = 2 * _KAPPA + 1
_BLOCK = uniform_bytes_needed(_MODULUS, _D // 2)

# ~32-bit partial-split prime (≡ 5 mod 8), from lattice-frx's
# `find_nearest_split_primes(32, 1)` — any split prime works for the
# Lemma-2.6 smoke; this one is pinned for determinism.
_SPLIT_Q = 4294967197


def _full_stream(seed: int) -> np.ndarray:
    """A whole default-budget stream of seeded random bytes."""
    rng = np.random.default_rng(seed)
    size = challenge_bytes_needed(_D, _KAPPA, _ETA, _K)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def _block_of_draws(value: int) -> np.ndarray:
    """A candidate block whose every accepted chunk decodes to `value`.

    Chunk values below the modulus are always accepted and contribute
    themselves, so a block of constant little-endian `value` chunks makes
    `uniform_from_bytes` return `value` at every position — the handle the
    fall-through tests use to plant a chosen candidate."""
    return np.full(_BLOCK // 8, value, dtype="<u8").view(np.uint8)


def _embed(draws: np.ndarray) -> np.ndarray:
    """The documented draw→coefficient embedding, re-derived independently:
    balanced draws fill `c_0 .. c_{d/2-1}`, `c_{d/2} = 0`, and the upper
    half mirrors negated (`c_{d-i} = -c_i`)."""
    c = np.zeros(_D, dtype=np.int64)
    c[: _D // 2] = draws.astype(np.int64) - _KAPPA
    for i in range(1, _D // 2):
        c[_D - i] = -c[i]
    return c


def _negacyclic_mul(a: list[int], b: list[int]) -> list[int]:
    """Schoolbook negacyclic product over exact Python ints — the
    independent oracle for the module's gate arithmetic."""
    d = len(a)
    conv = [0] * (2 * d)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                conv[i + j] += ai * bj
    return [conv[i] - conv[i + d] for i in range(d)]


def _gate_l1(c: np.ndarray) -> int:
    """`‖σ₋₁(c^k)·c^k‖₁` over `Z[X]/(X^d + 1)`, recomputed from scratch."""
    p = [int(v) for v in c]
    for _ in range(_K.bit_length() - 1):  # k is a power of two: 5 squarings
        p = _negacyclic_mul(p, p)
    sigma = [p[0]] + [-p[_D - i] for i in range(1, _D)]
    return sum(abs(v) for v in _negacyclic_mul(sigma, p))


class ChallengeFromBytesTest(absltest.TestCase):
    def test_identical_bytes_yield_the_identical_challenge(self) -> None:
        data = _full_stream(0)
        first = challenge_from_bytes(data, _D, _KAPPA, _ETA, _K)
        second = challenge_from_bytes(data.copy(), _D, _KAPPA, _ETA, _K)
        np.testing.assert_array_equal(first, second)

    def test_different_bytes_yield_a_different_challenge(self) -> None:
        first = challenge_from_bytes(_full_stream(0), _D, _KAPPA, _ETA, _K)
        second = challenge_from_bytes(_full_stream(1), _D, _KAPPA, _ETA, _K)
        self.assertFalse(np.array_equal(first, second))

    def test_the_challenge_is_sigma_minus_one_invariant(self) -> None:
        c = challenge_from_bytes(_full_stream(2), _D, _KAPPA, _ETA, _K)
        self.assertEqual(c[_D // 2], 0)
        for i in range(1, _D // 2):
            self.assertEqual(c[_D - i], -c[i])

    def test_coefficients_stay_within_kappa(self) -> None:
        c = challenge_from_bytes(_full_stream(3), _D, _KAPPA, _ETA, _K)
        self.assertLessEqual(int(np.abs(c).max()), _KAPPA)

    def test_an_accepted_challenge_passes_the_eta_gate(self) -> None:
        c = challenge_from_bytes(_full_stream(4), _D, _KAPPA, _ETA, _K)
        self.assertTrue(c.any())
        self.assertLessEqual(_gate_l1(c), _ETA ** (2 * _K))

    def test_the_challenge_is_a_unit_in_the_split_ring(self) -> None:
        """Lemma 2.6 smoke: a nonzero σ₋₁-invariant element of S_κ is
        invertible in the partial-split ring."""
        c = challenge_from_bytes(_full_stream(5), _D, _KAPPA, _ETA, _K)
        ring = HostSplitRing((_SPLIT_Q,), _D)
        self.assertTrue(ring.is_invertible(ring.from_signed(c.tolist())))

    def _assert_planted_block_falls_through(self, seed: int, draw: int) -> None:
        """Plant `_block_of_draws(draw)` as candidate block 0 of a seeded
        stream and require the sampler to return block 1's embedding — the
        shared crafted-stream shape of the fall-through tests."""
        data = _full_stream(seed)
        data[:_BLOCK] = _block_of_draws(draw)
        expected = _embed(
            uniform_from_bytes(data[_BLOCK : 2 * _BLOCK], _MODULUS, _D // 2)
        )
        got = challenge_from_bytes(data, _D, _KAPPA, _ETA, _K)
        np.testing.assert_array_equal(got, expected)

    def test_a_zero_candidate_falls_through_to_the_next_block(self) -> None:
        """All draws equal to κ embed to the zero polynomial, which passes
        the η gate numerically but is not a challenge — the sampler must
        move to the next block."""
        self._assert_planted_block_falls_through(seed=6, draw=_KAPPA)

    def test_an_over_eta_candidate_falls_through_to_the_next_block(self) -> None:
        """All draws at 2κ embed to the extremal ±κ ladder, whose gate ℓ1
        is astronomically over η^2k — the sampler must reject it on the
        gate and take the following block."""
        bad = _embed(np.full(_D // 2, 2 * _KAPPA, dtype=np.int64))
        # Guard the crafting assumption with the independent gate oracle.
        self.assertGreater(_gate_l1(bad), _ETA ** (2 * _K))
        self._assert_planted_block_falls_through(seed=7, draw=2 * _KAPPA)

    def test_a_stream_of_only_rejected_candidates_raises(self) -> None:
        attempts = (
            challenge_bytes_needed(_D, _KAPPA, _ETA, _K, fail_prob=2.0**-4) // _BLOCK
        )
        data = np.concatenate([_block_of_draws(2 * _KAPPA)] * attempts)
        with self.assertRaisesRegex(RuntimeError, "challenge_from_bytes"):
            challenge_from_bytes(data, _D, _KAPPA, _ETA, _K, fail_prob=2.0**-4)

    def test_random_streams_mostly_accept_the_first_candidate(self) -> None:
        """The budget's stated per-candidate rejection bound is 1/2; the
        measured rate at the Figure-3 parameters is ~1%. Accepting the
        first block on at least 36 of 40 honest streams keeps the stated
        bound comfortably conservative."""
        first_block_hits = 0
        for seed in range(40):
            data = _full_stream(100 + seed)
            got = challenge_from_bytes(data, _D, _KAPPA, _ETA, _K)
            first = _embed(uniform_from_bytes(data[:_BLOCK], _MODULUS, _D // 2))
            if np.array_equal(got, first):
                first_block_hits += 1
        self.assertGreaterEqual(first_block_hits, 36)


class ChallengeBytesNeededTest(absltest.TestCase):
    def test_the_budget_is_attempts_times_the_block(self) -> None:
        """128 attempts at the default 2^-128 failure probability under the
        stated 1/2 per-candidate rejection bound; 4 at 2^-4."""
        self.assertEqual(challenge_bytes_needed(_D, _KAPPA, _ETA, _K), 128 * _BLOCK)
        self.assertEqual(
            challenge_bytes_needed(_D, _KAPPA, _ETA, _K, fail_prob=2.0**-4),
            4 * _BLOCK,
        )

    def test_a_wrong_byte_count_raises(self) -> None:
        data = _full_stream(8)
        with self.assertRaises(ValueError):
            challenge_from_bytes(data[:-1], _D, _KAPPA, _ETA, _K)
        with self.assertRaises(ValueError):
            challenge_from_bytes(
                np.concatenate([data, np.zeros(1, dtype=np.uint8)]),
                _D,
                _KAPPA,
                _ETA,
                _K,
            )

    def test_fractional_parameters_raise(self) -> None:
        """A float parameter is a different kind of bug than an
        out-of-range one — and a fractional eta would silently turn the
        exact integer gate into a float comparison."""
        fractional: list[tuple[Any, Any, Any, Any]] = [
            (128.0, _KAPPA, _ETA, _K),
            (_D, 2.5, _ETA, _K),
            (_D, _KAPPA, 59.5, _K),
            (_D, _KAPPA, _ETA, 32.0),
        ]
        data = _full_stream(9)
        for bad in fractional:
            with self.assertRaises(TypeError):
                challenge_bytes_needed(*bad)
            with self.assertRaises(TypeError):
                challenge_from_bytes(data, *bad)

    def test_numpy_integer_parameters_normalize(self) -> None:
        """np integer scalars are accepted and behave as their Python-int
        values — eta especially, whose 2k-th power would wrap in a fixed
        64-bit lane."""
        d, kappa, eta, k = (np.int64(v) for v in (_D, _KAPPA, _ETA, _K))
        self.assertEqual(
            challenge_bytes_needed(d, kappa, eta, k),
            challenge_bytes_needed(_D, _KAPPA, _ETA, _K),
        )
        data = _full_stream(10)
        np.testing.assert_array_equal(
            challenge_from_bytes(data, d, kappa, eta, k),
            challenge_from_bytes(data, _D, _KAPPA, _ETA, _K),
        )

    def test_a_multidimensional_stream_raises(self) -> None:
        """A 2-D uint8 array of the right total size would slice along
        axis 0, not bytes — rejected as a wrong kind of buffer."""
        data = _full_stream(11)
        with self.assertRaises(TypeError):
            challenge_from_bytes(data.reshape(1, -1), _D, _KAPPA, _ETA, _K)

    def test_invalid_parameters_raise(self) -> None:
        good = (_D, _KAPPA, _ETA, _K)
        for bad in [
            (96, _KAPPA, _ETA, _K),  # not a power of two
            (2, _KAPPA, _ETA, _K),  # too small to carry the invariant shape
            (_D, 0, _ETA, _K),
            (_D, _KAPPA, 0, _K),
            (_D, _KAPPA, _ETA, 0),
        ]:
            with self.assertRaises(ValueError):
                challenge_bytes_needed(*bad)
        for fail_prob in [0.0, 1.0]:
            with self.assertRaises(ValueError):
                challenge_bytes_needed(*good, fail_prob=fail_prob)


if __name__ == "__main__":
    absltest.main()
