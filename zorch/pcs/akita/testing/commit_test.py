# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Akita commit — decomposition exactness, the opening predicate, and the
homomorphism a folding consumer stands on.

Structural correctness without goldens, like the `ajtai_test` below it: the
digits recompose to the polynomials that were committed, an opening that is
short and re-commits is accepted, and openings that are over-bound,
substituted, or recompose to something else are rejected by the predicate
rather than by luck. Randomness is a seeded numpy `Generator` for the public
matrix and `testkit.rand_field` for the polynomials.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from lattice_frx.ring import Eval, RnsRing

from zorch.commit.ajtai import commitments_equal
from zorch.pcs.akita.commit import AkitaCommitter, _balanced_lift
from zorch.pcs.akita.config import AkitaConfig, Decomposition, SisProfile
from zorch.testkit.random_field import rand_field

# The NTT-friendly 36-bit pair `ajtai_test` uses, at a degree kept small for
# test wall time.
_Q = (34359753217, 34359754753)
_D = 64
_ROWS = 2
_LOG_BASE = 8
# Two messages, deliberately not both a whole number of ring elements: the
# short one exercises the per-message padding, and their different block
# counts would hide a column-layout error that a uniform batch does not.
_LENS = (_D, _D + 5)
_FIELD = zk_dtypes.goldilocks_mont


def _config() -> AkitaConfig:
    modulus = int(zk_dtypes.pfinfo(_FIELD).modulus)
    return AkitaConfig(
        profile=SisProfile(_D, _Q),
        decomposition=Decomposition.covering(_LOG_BASE, modulus // 2),
        rows=_ROWS,
        message_lens=_LENS,
    )


def _uniform_matrix(
    ring: RnsRing, rng: np.random.Generator, rows: int, cols: int
) -> Eval:
    def element() -> Eval:
        host = np.array(
            [rng.integers(0, q, size=ring.d, dtype=np.uint64) for q in ring.q_moduli],
            dtype=np.uint64,
        )
        return ring.eval_from_host(host)

    return ring.stack(
        [ring.stack([element() for _ in range(cols)]) for _ in range(rows)]
    )


def _committer(config: AkitaConfig, seed: int = 7) -> AkitaCommitter:
    matrix = _uniform_matrix(
        config.profile.ring, np.random.default_rng(seed), config.rows, config.cols
    )
    return AkitaCommitter(config, matrix)


def _polys(seed: int) -> list[frx.Array]:
    return [rand_field(seed + i, (length,), _FIELD) for i, length in enumerate(_LENS)]


class AkitaCommitTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = _config()
        self.committer = _committer(self.config)
        self.polys = _polys(1)

    def test_digits_recompose_to_the_committed_polynomials(self) -> None:
        _, data = self.committer.commit(self.polys)
        self.assertEqual(
            self.committer.recompose(data.witness),
            [_balanced_lift(poly) for poly in self.polys],
        )

    def test_module_width_follows_the_parameter_point(self) -> None:
        # One column per (digit, block): 1 + 2 blocks for the two messages at
        # this degree, times the digit count the field forces.
        self.assertEqual(self.config.blocks_per_message, (1, 2))
        self.assertEqual(self.config.cols, self.config.decomposition.num_digits * 3)

    def test_opens_to_accepts_the_honest_opening(self) -> None:
        commitment, data = self.committer.commit(self.polys)
        self.assertTrue(self.committer.opens_to(commitment, data.witness, self.polys))

    def test_opens_to_rejects_a_different_polynomial(self) -> None:
        commitment, data = self.committer.commit(self.polys)
        self.assertFalse(self.committer.opens_to(commitment, data.witness, _polys(2)))

    def test_verify_rejects_a_substituted_opening(self) -> None:
        commitment, _ = self.committer.commit(self.polys)
        _, other = self.committer.commit(_polys(2))
        self.assertFalse(self.committer.verify(commitment, other.witness))

    def test_verify_rejects_an_over_bound_opening(self) -> None:
        # A spike past the digit bound rather than a perturbation: adding to a
        # digit could land back inside the interval and leave the rejection
        # branch seed-lucky.
        ring = self.config.profile.ring
        spike = ring.from_signed([self.config.beta_inf + 1] + [0] * (_D - 1))
        over = ring.stack([spike] * self.config.cols)
        self.assertFalse(
            self.committer.verify(self.committer.commit(self.polys)[0], over)
        )

    def test_commitment_is_additively_homomorphic_in_the_witness(self) -> None:
        # The property a folding consumer needs, restated at the scheme layer:
        # the decomposition changes the witness, not the linearity of `A·s`.
        # (The digits themselves do not add — carries — which is why this is a
        # statement about witnesses and not about polynomials.)
        ring = self.config.profile.ring
        left = self.committer.decompose(self.polys)
        right = self.committer.decompose(_polys(2))
        summed = self.committer.scheme.commit(
            self.committer.matrix, ring.ntt(ring.add(left, right))
        )
        parts = ring.add(
            self.committer.scheme.commit(self.committer.matrix, ring.ntt(left)),
            self.committer.scheme.commit(self.committer.matrix, ring.ntt(right)),
        )
        self.assertTrue(commitments_equal(summed, parts))

    def test_decompose_refuses_a_batch_of_the_wrong_arity(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2 polynomials"):
            self.committer.decompose(self.polys[:1])

    def test_decompose_refuses_a_polynomial_of_the_wrong_length(self) -> None:
        short = [self.polys[0], self.polys[1][:-1]]
        with self.assertRaisesRegex(ValueError, "config declares"):
            self.committer.decompose(short)

    def test_recompose_refuses_an_ntt_domain_witness(self) -> None:
        # Digits are coefficients; a value-domain element wearing the same
        # shape would recompose to noise rather than fail.
        _, data = self.committer.commit(self.polys)
        with self.assertRaisesRegex(TypeError, "coefficient-domain"):
            self.committer.recompose(self.config.profile.ring.ntt(data.witness))

    def test_balanced_lift_centers_the_field(self) -> None:
        modulus = int(zk_dtypes.pfinfo(_FIELD).modulus)
        values = fnp.array([0, 1, modulus - 1], dtype=_FIELD)
        self.assertEqual(_balanced_lift(values), [0, 1, -1])


if __name__ == "__main__":
    absltest.main()
