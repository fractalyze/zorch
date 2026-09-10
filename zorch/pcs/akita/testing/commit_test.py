# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Akita commit — the two tiers' shapes, the fold identity the outer tier
exists to make checkable, and decomposition exactness.

Structural correctness plus one known answer. The structural half is the
`ajtai_test` discipline below it: the digits recompose to the polynomials that
were committed, an opening that is short and re-commits is accepted, and
openings that are over-bound, substituted, or recompose to something else are
rejected by the predicate rather than by luck. The known answer is what a
structural test cannot reach — that the layout, the two decompositions and the
two module products compose into the payload a second implementation computes,
here a host integer one over `Q` (`_reference_commit`).

Randomness is a seeded numpy `Generator` for the public matrices and
`testkit.rand_field` for the polynomials.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from lattice_frx import gadget
from lattice_frx.ring import Coeff, Eval, RnsRing

from zorch.commit.ajtai import centered_lift, commitments_equal
from zorch.pcs.akita.commit import AkitaCommitter, _balanced_lift
from zorch.pcs.akita.config import AkitaConfig, Decomposition, SisProfile
from zorch.testkit.random_field import rand_field

# The NTT-friendly 36-bit pair `ajtai_test` uses, at a degree kept small for
# test wall time.
_Q = (34359753217, 34359754753)
_D = 64
_INNER_ROWS = 2
_OUTER_ROWS = 2
_LOG_BASE = 8
# Two messages, deliberately not both a whole number of ring elements: the
# short one exercises the per-message padding, and their different block
# counts would hide a layout error that a uniform batch does not.
_LENS = (_D, _D + 5)
# Any field wider than the digit base serves — the digit count is sized against
# whichever one this is, not the other way round.
_FIELD = zk_dtypes.goldilocks


def _config() -> AkitaConfig:
    modulus = int(zk_dtypes.pfinfo(_FIELD).modulus)
    profile = SisProfile(_D, _Q)
    return AkitaConfig(
        profile=profile,
        # The inner tier shortens field coefficients, the outer tier ring
        # coefficients modulo `Q` — hence two digit counts, each covering its
        # own magnitude.
        inner_decomposition=Decomposition.covering(_LOG_BASE, modulus // 2),
        outer_decomposition=Decomposition.covering(_LOG_BASE, profile.modulus // 2),
        inner_rows=_INNER_ROWS,
        outer_rows=_OUTER_ROWS,
        message_lens=_LENS,
    )


def _uniform_coefficient(rng: np.random.Generator, modulus: int) -> int:
    """One coefficient in `[0, Q)`, drawn as two 32-bit halves because `Q` is
    wider than the 64-bit bound `rng.integers` accepts."""
    return int(rng.integers(0, modulus >> 32)) << 32 | int(rng.integers(0, 1 << 32))


def _uniform_matrix(
    ring: RnsRing, rng: np.random.Generator, rows: int, cols: int, modulus: int
) -> tuple[Eval, list[list[list[int]]]]:
    """A public matrix, and the host integers modulo `Q` it was built from.

    The integers come back because `_reference_commit` needs the same matrix in
    a representation it can multiply with Python arithmetic; drawing residues
    per limb instead would leave the reference nothing to reconstruct from.
    """
    host = [
        [
            [_uniform_coefficient(rng, modulus) for _ in range(ring.d)]
            for _ in range(cols)
        ]
        for _ in range(rows)
    ]
    coefficients = ring.stack(
        [ring.stack([ring.from_signed(column) for column in row]) for row in host]
    )
    return ring.ntt(coefficients), host


def _committer(
    config: AkitaConfig, seed: int = 7
) -> tuple[AkitaCommitter, list[list[list[int]]], list[list[list[int]]]]:
    ring = config.profile.ring
    rng = np.random.default_rng(seed)
    modulus = config.profile.modulus
    inner, inner_host = _uniform_matrix(
        ring, rng, config.inner_rows, config.inner_cols, modulus
    )
    outer, outer_host = _uniform_matrix(
        ring, rng, config.outer_rows, config.outer_cols, modulus
    )
    return AkitaCommitter(config, inner, outer), inner_host, outer_host


def _polys(seed: int) -> list[frx.Array]:
    return [rand_field(seed + i, (length,), _FIELD) for i, length in enumerate(_LENS)]


def _lead(element: Coeff | Eval) -> tuple[int, ...]:
    """The module axes of a ring element — its shape as the scheme states
    it, with the per-coefficient axis dropped."""
    return tuple(element.limbs[0].shape[:-1])


def _block(element: Coeff | Eval, index: int) -> Coeff | Eval:
    """One block out of a batched ring element — the per-block view the fold
    identity is stated over, which the ring itself has no accessor for."""
    return type(element)(tuple(limb[index] for limb in element.limbs))


def _fold(ring: RnsRing, batched: Eval, challenges: list[Eval]) -> Eval:
    """`Σ_b c_b·x_b` over the block axis of a batched ring element."""
    folded = ring.mul(challenges[0], _block(batched, 0))
    for block, challenge in enumerate(challenges[1:], start=1):
        folded = ring.add(folded, ring.mul(challenge, _block(batched, block)))
    return folded


def _center(value: int, modulus: int) -> int:
    """`reconstruct_centered`'s balanced representative, including its
    non-strict boundary (lattice-frx's `rns.py`)."""
    return value - modulus if value >= modulus >> 1 else value


def _negacyclic_mul(a: list[int], b: list[int], modulus: int) -> list[int]:
    """`a·b` in `Z_Q[X]/(X^d+1)`, the schoolbook way: the wrap past degree `d`
    changes sign, which is the whole content of "negacyclic"."""
    degree = len(a)
    out = [0] * degree
    for i, x in enumerate(a):
        if not x:
            continue
        for j, y in enumerate(b):
            if i + j < degree:
                out[i + j] += x * y
            else:
                out[i + j - degree] -= x * y
    return [value % modulus for value in out]


def _matvec(
    matrix: list[list[list[int]]], vector: list[list[int]], modulus: int
) -> list[list[int]]:
    degree = len(vector[0])
    out: list[list[int]] = []
    for row in matrix:
        total = [0] * degree
        for entry, column in zip(row, vector):
            for index, value in enumerate(_negacyclic_mul(entry, column, modulus)):
                total[index] += value
        out.append([value % modulus for value in total])
    return out


def _reference_commit(
    config: AkitaConfig,
    inner: list[list[list[int]]],
    outer: list[list[list[int]]],
    polys: list[frx.Array],
) -> list[int]:
    """`u = B·t̂` over host integers modulo `Q`, as balanced coefficients.

    An independent route to the payload: exact Python arithmetic in place of the
    NTT ring, and the layout spelled out as loops rather than read off
    `AkitaConfig`. Only `gadget` is shared with the implementation — the digit
    interval is the substrate's definition, not this scheme's.
    """
    degree = config.profile.degree
    modulus = config.profile.modulus
    padded: list[int] = []
    for poly, length, blocks in zip(
        polys, config.message_lens, config.blocks_per_message
    ):
        padded.extend(_balanced_lift(poly))
        padded.extend([0] * (blocks * degree - length))

    planes = gadget.decompose_vector(
        padded,
        config.inner_decomposition.log_base,
        config.inner_decomposition.num_digits,
    )
    images: list[list[int]] = []
    for block in range(config.blocks):
        witness = [plane[block * degree : (block + 1) * degree] for plane in planes]
        images.extend(_matvec(inner, witness, modulus))

    lifted = [_center(value, modulus) for image in images for value in image]
    outer_planes = gadget.decompose_vector(
        lifted,
        config.outer_decomposition.log_base,
        config.outer_decomposition.num_digits,
    )
    hat = [
        plane[image * degree : (image + 1) * degree]
        for plane in outer_planes
        for image in range(config.images)
    ]
    return [
        _center(value, modulus) for row in _matvec(outer, hat, modulus) for value in row
    ]


class AkitaCommitTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = _config()
        self.committer, self.inner_host, self.outer_host = _committer(self.config)
        self.polys = _polys(1)
        self.ring = self.config.profile.ring

    def test_digits_recompose_to_the_committed_polynomials(self) -> None:
        _, data = self.committer.commit(self.polys)
        self.assertEqual(
            self.committer.recompose(data.witness),
            [_balanced_lift(poly) for poly in self.polys],
        )

    def test_module_widths_follow_the_parameter_point(self) -> None:
        # One inner column per digit plane, one inner image per (block, row),
        # one outer column per (digit, image).
        self.assertEqual(self.config.blocks_per_message, (1, 2))
        self.assertEqual(self.config.blocks, 3)
        self.assertEqual(self.config.inner_cols, 9)
        self.assertEqual(self.config.images, 3 * _INNER_ROWS)
        self.assertEqual(
            self.config.outer_cols,
            self.config.outer_decomposition.num_digits * 3 * _INNER_ROWS,
        )

    def test_commit_shapes_are_the_two_tiers(self) -> None:
        commitment, data = self.committer.commit(self.polys)
        blocks = self.config.blocks
        self.assertEqual(_lead(data.witness), (blocks, self.config.inner_cols))
        self.assertEqual(_lead(data.images), (blocks, _INNER_ROWS))
        self.assertEqual(_lead(commitment), (_OUTER_ROWS,))

    def test_payload_matches_a_host_integer_reference(self) -> None:
        commitment, _ = self.committer.commit(self.polys)
        self.assertEqual(
            centered_lift("commitment", self.ring, self.ring.intt(commitment)),
            _reference_commit(
                self.config, self.inner_host, self.outer_host, self.polys
            ),
        )

    def test_payload_is_the_outer_commitment_of_the_retained_images(self) -> None:
        # What makes the retention meaningful: the images the prover keeps are
        # exactly the table the public payload was taken over, so an opening
        # protocol may fold them without re-deriving the commitment.
        commitment, data = self.committer.commit(self.polys)
        self.assertTrue(
            commitments_equal(self.committer.outer_commit(data.images), commitment)
        )

    def test_folded_images_recommit_to_the_folded_witness(self) -> None:
        # The identity the two-tier layout exists for: `Σ_b c_b·t_b =
        # A·(Σ_b c_b·s_b)`, which a verifier can only state because it can name
        # the individual `t_b`. A single flat `A·s` over the batch is one sum
        # they are not recoverable from, so no such identity exists there.
        _, data = self.committer.commit(self.polys)
        ring = self.ring
        rng = np.random.default_rng(11)
        challenges = [
            ring.ntt(ring.from_signed(rng.integers(-1, 2, size=_D)))
            for _ in range(self.config.blocks)
        ]
        folded_witness = _fold(ring, ring.ntt(data.witness), challenges)
        folded_images = _fold(ring, ring.ntt(data.images), challenges)
        self.assertTrue(
            commitments_equal(
                self.committer.inner.commit(
                    self.committer.inner_matrix, folded_witness
                ),
                folded_images,
            )
        )

    def test_inner_images_are_additively_homomorphic_in_the_witness(self) -> None:
        # The property a folding consumer needs, restated at the scheme layer:
        # the decomposition changes the witness, not the linearity of `A·s_b`.
        # (The digits themselves do not add — carries — which is why this is a
        # statement about witnesses and not about polynomials, and why the
        # *outer* tier is not homomorphic.)
        ring = self.ring
        left = self.committer.decompose(self.polys)
        right = self.committer.decompose(_polys(2))
        summed = self.committer.inner_images(ring.add(left, right))
        parts = ring.add(
            self.committer.inner_images(left), self.committer.inner_images(right)
        )
        self.assertTrue(commitments_equal(ring.ntt(summed), ring.ntt(parts)))

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
        ring = self.ring
        spike = ring.from_signed([self.config.inner_beta_inf + 1] + [0] * (_D - 1))
        block = ring.stack([spike] * self.config.inner_cols)
        over = ring.stack([block] * self.config.blocks)
        self.assertFalse(
            self.committer.verify(self.committer.commit(self.polys)[0], over)
        )

    def test_decompose_refuses_a_batch_of_the_wrong_arity(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2 polynomials"):
            self.committer.decompose(self.polys[:1])

    def test_decompose_refuses_a_polynomial_of_the_wrong_length(self) -> None:
        short = [self.polys[0], self.polys[1][:-1]]
        with self.assertRaisesRegex(ValueError, "config declares"):
            self.committer.decompose(short)

    def test_inner_images_refuse_a_witness_of_the_wrong_block_width(self) -> None:
        ring = self.ring
        narrow = ring.stack(
            [ring.stack([ring.from_signed([0] * _D)] * (self.config.inner_cols - 1))]
            * self.config.blocks
        )
        with self.assertRaisesRegex(ValueError, "commit_batch: witnesses"):
            self.committer.inner_images(narrow)

    def test_recompose_refuses_an_ntt_domain_witness(self) -> None:
        # Digits are coefficients; a value-domain element wearing the same
        # shape would recompose to noise rather than fail.
        _, data = self.committer.commit(self.polys)
        with self.assertRaisesRegex(TypeError, "coefficient-domain"):
            self.committer.recompose(self.ring.ntt(data.witness))

    def test_balanced_lift_centers_the_field(self) -> None:
        modulus = int(zk_dtypes.pfinfo(_FIELD).modulus)
        values = fnp.array([0, 1, modulus - 1], dtype=_FIELD)
        self.assertEqual(_balanced_lift(values), [0, 1, -1])


if __name__ == "__main__":
    absltest.main()
