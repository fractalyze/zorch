"""Ajtai/BDLOP commit — roundtrip, homomorphism, and the opening predicate.

Structural correctness without goldens, like `merkle_test`: a valid opening
re-commits to the committed value, the additive homomorphism (the folding
prerequisite) is checked as algebra, and an over-bound or substituted opening
is rejected by the predicate rather than by luck. Randomness is a seeded
numpy `Generator` — research-grade by the module's own stance.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest
from lattice_frx.ring import Coeff, Eval, RnsRing

from zorch.commit.ajtai import (
    AjtaiCommitment,
    AjtaiParams,
    BdlopCommitment,
    BdlopOpening,
    BdlopParams,
)

# NTT-friendly 36-bit pair from lattice-frx's own suite; d kept small for
# test wall time (both are 1 mod 2d for every power of two up to 256).
_Q = (34359753217, 34359754753)
_D = 64
_ROWS, _COLS = 2, 3
_BETA = 1  # ternary witnesses


def _ring() -> RnsRing:
    return RnsRing(_Q, _D)


def _uniform_eval(ring: RnsRing, rng: np.random.Generator) -> Eval:
    """One uniform NTT-domain element, host-embedded per limb."""
    host = np.array(
        [rng.integers(0, q, size=_D, dtype=np.uint64) for q in _Q], dtype=np.uint64
    )
    return ring.eval_from_host(host)


def _uniform_matrix(
    ring: RnsRing, rng: np.random.Generator, rows: int, cols: int
) -> Eval:
    return ring.stack(
        [
            ring.stack([_uniform_eval(ring, rng) for _ in range(cols)])
            for _ in range(rows)
        ]
    )


def _assert_equal_limbs(got: Coeff | Eval, want: Coeff | Eval) -> None:
    """Exact per-limb equality, strict on limb count like the module's own
    predicate."""
    for x, y in zip(got.limbs, want.limbs, strict=True):
        np.testing.assert_array_equal(
            np.asarray(x).astype(object), np.asarray(y).astype(object)
        )


def _differ(a: Coeff | Eval, b: Coeff | Eval) -> bool:
    return not all(
        np.array_equal(np.asarray(x).astype(object), np.asarray(y).astype(object))
        for x, y in zip(a.limbs, b.limbs, strict=True)
    )


def _ternary_witness(ring: RnsRing, rng: np.random.Generator, cols: int) -> Coeff:
    """A `[cols, d]` coefficient-domain witness with entries in {-1, 0, 1}."""
    return ring.stack(
        [ring.from_signed(rng.integers(-1, 2, size=_D).tolist()) for _ in range(cols)]
    )


class AjtaiTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ring = _ring()
        self.params = AjtaiParams(
            ring=self.ring, rows=_ROWS, cols=_COLS, beta_inf=_BETA
        )
        self.scheme = AjtaiCommitment(self.params)
        self.rng = np.random.default_rng(0)
        self.matrix = _uniform_matrix(self.ring, self.rng, _ROWS, _COLS)

    def test_a_valid_opening_verifies(self) -> None:
        witness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.matrix, self.ring.ntt(witness))
        self.assertTrue(self.scheme.verify(self.matrix, commitment, witness))

    def test_the_commitment_is_additively_homomorphic(self) -> None:
        """com(s1) + com(s2) == com(s1 + s2) — the folding prerequisite."""
        s1 = _ternary_witness(self.ring, self.rng, _COLS)
        s2 = _ternary_witness(self.ring, self.rng, _COLS)
        lhs = self.ring.add(
            self.scheme.commit(self.matrix, self.ring.ntt(s1)),
            self.scheme.commit(self.matrix, self.ring.ntt(s2)),
        )
        rhs = self.scheme.commit(self.matrix, self.ring.ntt(self.ring.add(s1, s2)))
        _assert_equal_limbs(lhs, rhs)

    def test_an_over_bound_opening_is_rejected(self) -> None:
        # `big` is over the bound by construction — adding it to a ternary
        # witness instead could land a -1 coefficient back inside the bound
        # and turn the test seed-lucky.
        big = self.ring.stack(
            [self.ring.from_signed([_BETA + 1] + [0] * (_D - 1)) for _ in range(_COLS)]
        )
        commitment = self.scheme.commit(self.matrix, self.ring.ntt(big))
        self.assertFalse(self.scheme.verify(self.matrix, commitment, big))

    def test_a_substituted_witness_is_rejected(self) -> None:
        witness = _ternary_witness(self.ring, self.rng, _COLS)
        other = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.matrix, self.ring.ntt(witness))
        self.assertFalse(self.scheme.verify(self.matrix, commitment, other))

    def test_commit_traces_under_jit(self) -> None:
        witness = _ternary_witness(self.ring, self.rng, _COLS)
        eager = self.scheme.commit(self.matrix, self.ring.ntt(witness))
        compiled = frx.jit(lambda a, s: self.scheme.commit(a, s))(
            self.matrix, self.ring.ntt(witness)
        )
        _assert_equal_limbs(compiled, eager)

    def test_a_truncated_commitment_is_rejected(self) -> None:
        """A commitment carrying fewer limbs is a different RNS chain — the
        predicate must not accept it on a matching prefix."""
        witness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.matrix, self.ring.ntt(witness))
        truncated = Eval(commitment.limbs[:1])
        self.assertFalse(self.scheme.verify(self.matrix, truncated, witness))

    def test_a_wrong_shape_is_rejected_at_commit(self) -> None:
        witness = _ternary_witness(self.ring, self.rng, _COLS + 1)
        with self.assertRaises(ValueError):
            self.scheme.commit(self.matrix, self.ring.ntt(witness))


class BdlopTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ring = _ring()
        self.params = BdlopParams(
            ring=self.ring,
            rows=_ROWS,
            randomness_cols=_COLS,
            messages=1,
            beta_inf=_BETA,
        )
        self.scheme = BdlopCommitment(self.params)
        self.rng = np.random.default_rng(1)
        self.b0 = _uniform_matrix(self.ring, self.rng, _ROWS, _COLS)
        self.b1 = _uniform_matrix(self.ring, self.rng, 1, _COLS)

    def _message(self) -> Coeff:
        return self.ring.stack(
            [self.ring.from_signed(self.rng.integers(0, 100, size=_D).tolist())]
        )

    def test_a_valid_opening_verifies(self) -> None:
        message = self._message()
        randomness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.b0, self.b1, message, randomness)
        opening = BdlopOpening(message=message, randomness=randomness)
        self.assertTrue(self.scheme.verify(self.b0, self.b1, commitment, opening))

    def test_a_wrong_message_is_rejected(self) -> None:
        message = self._message()
        randomness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.b0, self.b1, message, randomness)
        forged = BdlopOpening(message=self._message(), randomness=randomness)
        self.assertFalse(self.scheme.verify(self.b0, self.b1, commitment, forged))

    def test_over_bound_randomness_is_rejected(self) -> None:
        message = self._message()
        big = self.ring.stack(
            [self.ring.from_signed([_BETA + 1] + [0] * (_D - 1)) for _ in range(_COLS)]
        )
        commitment = self.scheme.commit(self.b0, self.b1, message, big)
        opening = BdlopOpening(message=message, randomness=big)
        self.assertFalse(self.scheme.verify(self.b0, self.b1, commitment, opening))

    def test_different_randomness_changes_the_commitment(self) -> None:
        """Hiding smoke only — the distributional statement is the scheme's
        parameter choice, not this seam's."""
        message = self._message()
        r1 = _ternary_witness(self.ring, self.rng, _COLS)
        r2 = _ternary_witness(self.ring, self.rng, _COLS)
        c1 = self.scheme.commit(self.b0, self.b1, message, r1)
        c2 = self.scheme.commit(self.b0, self.b1, message, r2)
        self.assertTrue(_differ(c1.t0, c2.t0))
        # t1 = B1·r + m must move with r too — checking t0 alone would let a
        # regression that drops the B1·r term hide behind verify, which
        # recomputes through the same implementation.
        self.assertTrue(_differ(c1.t1, c2.t1))

    def test_commit_traces_under_jit(self) -> None:
        message = self._message()
        randomness = _ternary_witness(self.ring, self.rng, _COLS)
        eager = self.scheme.commit(self.b0, self.b1, message, randomness)
        compiled = frx.jit(lambda b0, b1, m, r: self.scheme.commit(b0, b1, m, r))(
            self.b0, self.b1, message, randomness
        )
        _assert_equal_limbs(compiled.t0, eager.t0)
        _assert_equal_limbs(compiled.t1, eager.t1)


if __name__ == "__main__":
    absltest.main()
