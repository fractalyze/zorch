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
from lattice_frx.split_ring import HostSplitRing

from zorch.commit.ajtai import (
    AbdlopCommitment,
    AbdlopPair,
    AjtaiCommitment,
    BdlopCommitment,
    _equal,
)

# NTT-friendly 36-bit pair from lattice-frx's own suite; d kept small for
# test wall time (both are 1 mod 2d for every power of two up to 256).
_Q = (34359753217, 34359754753)
_D = 64
_ROWS, _COLS = 2, 3
_BETA = 1  # ternary witnesses

# Partial-split 36-bit pair (≡ 5 mod 8), from lattice-frx's
# `find_nearest_split_primes(36, 2)` — the mutually exclusive modulus shape
# the ABDLOP scheme lives on.
_SPLIT_Q = (68719476493, 68719476853)


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


def _over_bound_witness(ring: RnsRing, cols: int) -> Coeff:
    """Over the bound by construction — adding a spike to a ternary witness
    instead could land a -1 coefficient back inside the bound and leave the
    rejection branch seed-lucky."""
    return ring.stack(
        [ring.from_signed([_BETA + 1] + [0] * (_D - 1)) for _ in range(cols)]
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
        self.scheme = AjtaiCommitment(self.ring, _ROWS, _COLS, _BETA)
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
        big = _over_bound_witness(self.ring, _COLS)
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
        compiled = frx.jit(self.scheme.commit)(self.matrix, self.ring.ntt(witness))
        _assert_equal_limbs(compiled, eager)

    def test_a_truncated_commitment_is_rejected(self) -> None:
        """A commitment carrying fewer limbs is a different RNS chain — the
        predicate must not accept it on a matching prefix."""
        witness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.matrix, self.ring.ntt(witness))
        truncated = Eval(commitment.limbs[:1])
        self.assertFalse(self.scheme.verify(self.matrix, truncated, witness))

    def test_a_wrong_domain_opening_raises(self) -> None:
        witness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.matrix, self.ring.ntt(witness))
        with self.assertRaises(TypeError):
            self.scheme.verify(self.matrix, commitment, self.ring.ntt(witness))

    def test_a_wrong_shape_is_rejected_at_commit(self) -> None:
        witness = _ternary_witness(self.ring, self.rng, _COLS + 1)
        with self.assertRaises(ValueError):
            self.scheme.commit(self.matrix, self.ring.ntt(witness))


class BdlopTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ring = _ring()
        self.scheme = BdlopCommitment(self.ring, _ROWS, _COLS, 1, _BETA)
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
        self.assertTrue(
            self.scheme.verify(self.b0, self.b1, commitment, message, randomness)
        )

    def test_a_wrong_message_is_rejected(self) -> None:
        message = self._message()
        randomness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.b0, self.b1, message, randomness)
        self.assertFalse(
            self.scheme.verify(
                self.b0, self.b1, commitment, self._message(), randomness
            )
        )

    def test_over_bound_randomness_is_rejected(self) -> None:
        message = self._message()
        big = _over_bound_witness(self.ring, _COLS)
        commitment = self.scheme.commit(self.b0, self.b1, message, big)
        self.assertFalse(self.scheme.verify(self.b0, self.b1, commitment, message, big))

    def test_a_wrong_domain_randomness_raises(self) -> None:
        """The domain gate is `_within_bound`'s, shared by both schemes —
        this pins the BDLOP side, which a verify-level guard once missed."""
        message = self._message()
        randomness = _ternary_witness(self.ring, self.rng, _COLS)
        commitment = self.scheme.commit(self.b0, self.b1, message, randomness)
        with self.assertRaises(TypeError):
            self.scheme.verify(
                self.b0, self.b1, commitment, message, self.ring.ntt(randomness)
            )

    def test_different_randomness_changes_the_commitment(self) -> None:
        """Hiding smoke only — the distributional statement is the scheme's
        parameter choice, not this seam's."""
        message = self._message()
        r1 = _ternary_witness(self.ring, self.rng, _COLS)
        r2 = _ternary_witness(self.ring, self.rng, _COLS)
        c1 = self.scheme.commit(self.b0, self.b1, message, r1)
        c2 = self.scheme.commit(self.b0, self.b1, message, r2)
        self.assertFalse(_equal(c1.t0, c2.t0))
        # t1 = B1·r + m must move with r too — checking t0 alone would let a
        # regression that drops the B1·r term hide behind verify, which
        # recomputes through the same implementation.
        self.assertFalse(_equal(c1.t1, c2.t1))

    def test_commit_traces_under_jit(self) -> None:
        message = self._message()
        randomness = _ternary_witness(self.ring, self.rng, _COLS)
        eager = self.scheme.commit(self.b0, self.b1, message, randomness)
        compiled = frx.jit(lambda b0, b1, m, r: self.scheme.commit(b0, b1, m, r))(
            self.b0, self.b1, message, randomness
        )
        _assert_equal_limbs(compiled.t0, eager.t0)
        _assert_equal_limbs(compiled.t1, eager.t1)


def _ternary_split_vector(
    ring: HostSplitRing, rng: np.random.Generator, cols: int
) -> np.ndarray:
    """A `(cols, limbs, d)` stack with coefficients in {-1, 0, 1}."""
    return ring.from_signed_stack(rng.integers(-1, 2, size=(cols, _D)))


def _over_bound_split_vector(ring: HostSplitRing, cols: int) -> np.ndarray:
    """Over the bound by construction, as `_over_bound_witness` above."""
    return ring.from_signed_stack([[_BETA + 1] + [0] * (_D - 1)] * cols)


class AbdlopTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ring = HostSplitRing(_SPLIT_Q, _D)
        self.scheme = AbdlopCommitment(
            self.ring,
            rows=_ROWS,
            s1_cols=_COLS,
            randomness_cols=_COLS,
            messages=1,
            beta1_inf=_BETA,
            beta2_inf=_BETA,
        )
        self.rng = np.random.default_rng(2)
        self.a1 = self.ring.uniform_stack(self.rng, _ROWS, _COLS)
        self.a2 = self.ring.uniform_stack(self.rng, _ROWS, _COLS)
        self.b = self.ring.uniform_stack(self.rng, 1, _COLS)

    def _witnesses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A fresh `(s1, s2, message)` triple — s1/s2 ternary, the message
        unconstrained uniform."""
        s1 = _ternary_split_vector(self.ring, self.rng, _COLS)
        s2 = _ternary_split_vector(self.ring, self.rng, _COLS)
        message = self.ring.uniform_stack(self.rng, 1)
        return s1, s2, message

    def test_a_valid_opening_verifies(self) -> None:
        s1, s2, message = self._witnesses()
        commitment = self.scheme.commit(self.a1, self.a2, self.b, s1, s2, message)
        self.assertTrue(
            self.scheme.verify(self.a1, self.a2, self.b, commitment, s1, s2, message)
        )

    def test_the_commitment_is_additively_homomorphic(self) -> None:
        """com(s1,s2,m) + com(s1',s2',m') == com(s1+s1', s2+s2', m+m') —
        linear in the whole witness triple, the folding prerequisite."""
        s1, s2, message = self._witnesses()
        t1, u2, other = self._witnesses()
        first = self.scheme.commit(self.a1, self.a2, self.b, s1, s2, message)
        second = self.scheme.commit(self.a1, self.a2, self.b, t1, u2, other)
        lhs_t_a = self.ring.add(first.t_a, second.t_a)
        lhs_t_b = self.ring.add(first.t_b, second.t_b)
        rhs = self.scheme.commit(
            self.a1,
            self.a2,
            self.b,
            self.ring.add(s1, t1),
            self.ring.add(s2, u2),
            self.ring.add(message, other),
        )
        np.testing.assert_array_equal(lhs_t_a, rhs.t_a)
        np.testing.assert_array_equal(lhs_t_b, rhs.t_b)

    def test_an_over_bound_s1_is_rejected(self) -> None:
        _, s2, message = self._witnesses()
        big = _over_bound_split_vector(self.ring, _COLS)
        commitment = self.scheme.commit(self.a1, self.a2, self.b, big, s2, message)
        self.assertFalse(
            self.scheme.verify(self.a1, self.a2, self.b, commitment, big, s2, message)
        )

    def test_an_over_bound_randomness_is_rejected(self) -> None:
        s1, _, message = self._witnesses()
        big = _over_bound_split_vector(self.ring, _COLS)
        commitment = self.scheme.commit(self.a1, self.a2, self.b, s1, big, message)
        self.assertFalse(
            self.scheme.verify(self.a1, self.a2, self.b, commitment, s1, big, message)
        )

    def test_a_substituted_witness_is_rejected(self) -> None:
        s1, s2, message = self._witnesses()
        other1, _, _ = self._witnesses()
        commitment = self.scheme.commit(self.a1, self.a2, self.b, s1, s2, message)
        self.assertFalse(
            self.scheme.verify(
                self.a1, self.a2, self.b, commitment, other1, s2, message
            )
        )

    def test_a_wrong_message_is_rejected(self) -> None:
        s1, s2, message = self._witnesses()
        commitment = self.scheme.commit(self.a1, self.a2, self.b, s1, s2, message)
        other = self.ring.uniform_stack(self.rng, 1)
        self.assertFalse(
            self.scheme.verify(self.a1, self.a2, self.b, commitment, s1, s2, other)
        )

    def test_a_truncated_commitment_is_rejected(self) -> None:
        """A commitment carrying fewer limbs is a different RNS chain — the
        predicate must not accept it on a matching prefix."""
        s1, s2, message = self._witnesses()
        commitment = self.scheme.commit(self.a1, self.a2, self.b, s1, s2, message)
        truncated = AbdlopPair(
            t_a=commitment.t_a[:, :1, :], t_b=commitment.t_b[:, :1, :]
        )
        self.assertFalse(
            self.scheme.verify(self.a1, self.a2, self.b, truncated, s1, s2, message)
        )

    def test_different_randomness_changes_the_commitment(self) -> None:
        """Hiding smoke only, as in `BdlopTest`: every part must move with
        s2 — t_a through A2·s2, t_b through B·s2."""
        s1, s2, message = self._witnesses()
        _, other, _ = self._witnesses()
        c1 = self.scheme.commit(self.a1, self.a2, self.b, s1, s2, message)
        c2 = self.scheme.commit(self.a1, self.a2, self.b, s1, other, message)
        self.assertFalse(np.array_equal(c1.t_a, c2.t_a))
        self.assertFalse(np.array_equal(c1.t_b, c2.t_b))

    def test_a_wrong_shape_is_rejected_at_commit(self) -> None:
        s1, s2, message = self._witnesses()
        wide = _ternary_split_vector(self.ring, self.rng, _COLS + 1)
        with self.assertRaises(ValueError):
            self.scheme.commit(self.a1, self.a2, self.b, wide, s2, message)
        with self.assertRaises(ValueError):
            self.scheme.commit(self.a1, self.a2, self.b, s1, wide, message)
        with self.assertRaises(ValueError):
            self.scheme.commit(self.a1, self.a2, self.b, s1, s2, s1)

    def test_an_ntt_ring_is_rejected_at_construction(self) -> None:
        """A deliberate tripwire on the dependency at this suite's exact
        primes: the two ring modes are mutually exclusive, and if
        lattice-frx ever relaxed its constructor guard, an NTT-friendly
        ring strayed into the ABDLOP scheme would be a soundness gap this
        suite must catch, not a parameter choice."""
        with self.assertRaises(ValueError):
            HostSplitRing(_Q, _D)


if __name__ == "__main__":
    absltest.main()
