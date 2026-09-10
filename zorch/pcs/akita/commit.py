# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The module-lattice commitment: decompose to short digits, then SIS-commit —
once per block, then once over what those commitments produced.

`zorch/commit/ajtai.py` binds a witness that is *already* short. A polynomial
over a prime field is not — its coefficients fill the field — so the scheme
layer is exactly the step that makes one short: balanced base-`2^w` digits,
laid out as ring elements.

**Why the commitment has two tiers.** The witness is cut into blocks, one ring
element's worth of coefficients each, and the inner tier commits every block on
its own: `t_b = A·s_b`. Those per-block images are what the opening protocol
folds — a challenge `c_b` per block, whose response satisfies
`Σ_b c_b·t_b = A·(Σ_b c_b·s_b)` — so a verifier that cannot name the individual
`t_b` has no identity to check a folded response against. A single flat
`t = A·s` over the whole batch destroys exactly that: it is one sum the `t_b`
cannot be recovered from. So the images are retained by the prover and bound by
an **outer** commitment instead — `u = B·t̂`, over `t̂` the digit decomposition
of the images, since a ring element modulo `Q` is no shorter than the witness
it came from and MSIS binds only short things. `u` is the public payload.

Two properties survive the decomposition and are why this shape is worth the
digit blow-up. The inner tier stays **additively homomorphic** in the digits,
which is what a folding consumer needs — the outer tier is not, and cannot be,
since digits carry; and one payload binds the **whole batch**, since every
polynomial's digits are blocks of one module witness — the same single-root
economy BaseFold gets from a matrix commitment, reached a different way.

**Where the host boundary sits.** Both digit decompositions are exact integer
arithmetic over the balanced lift: a residue's lift does not fit a lane, and
lattice-frx pins lifts, norms and `gadget` to the host for that reason. So
`commit` materialises twice — the input before the inner tier, the images
before the outer one — and hands ring elements back to the traced `matvec`.
That is a real cost and a real limit, not a stylistic choice: the substrate
offers no traced decomposition to call. Both host steps sit in `decompose` and
`outer_digits`, so the layout around them stays traced. Neither `matvec` splits
per block — the inner tier is one batched `matvec` over the block axis — so the
device side stays two units regardless of batch size.

The public matrices ride the committer, as a KZG proving key rides its prover:
they are the scheme instance's parameters, not per-call arguments, and a traced
ring element has no place in a frozen parameter point that is written down,
compared and serialized. Nothing here samples them — the CRS expansion from a
seed is the consumer's, matching the randomness stance of the commitment
algebra below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import zk_dtypes
from frx import Array
from lattice_frx import gadget
from lattice_frx.ring import Coeff, Eval

from zorch.commit.ajtai import AjtaiCommitment, centered_lift, within_bound
from zorch.pcs.akita.config import AkitaConfig
from zorch.pcs.stage import Committer

# The `[outer_rows]` module vector `u = B·t̂`. An alias, not a wrapper: it is a
# ring element the consumer observes into a transcript and folds like any other
# (`pcs.md`, "Commitments are aliases until they grow structure").
AkitaCommitment: TypeAlias = Eval


@dataclass(frozen=True)
class AkitaProverData:
    """What `commit` retained for the opening: the digit witness and the
    per-block inner images the outer commitment was taken over.

    Prover-only, and the reason the seam splits committer from opener at all.
    The witness is the polynomial, in a representation an opening can multiply
    by a challenge without leaving the norm bound; the images are the other
    half of that identity, and recomputing them from the witness at opening
    time would repeat the inner tier for values the prover already has.
    """

    witness: Coeff
    images: Coeff


class AkitaCommitter:
    """`commit(polys) = B·decompose(A·decompose(polys) per block)` and its
    opening predicate.

    Configuration rides the constructor — the parameter point, the two module
    heights and the batch shape — with the two public matrices beside it.
    Binding strength is the consumer's parameter choice; this layer enforces
    shapes, the digit bounds, and exact recomposition.
    """

    def __init__(
        self, config: AkitaConfig, inner_matrix: Eval, outer_matrix: Eval
    ) -> None:
        self.config = config
        self.inner_matrix = inner_matrix
        self.outer_matrix = outer_matrix
        ring = config.profile.ring
        self.inner = AjtaiCommitment(
            ring, config.inner_rows, config.inner_cols, config.inner_beta_inf
        )
        self.outer = AjtaiCommitment(
            ring, config.outer_rows, config.outer_cols, config.outer_beta_inf
        )

    def commit(self, polys: Sequence[Array]) -> tuple[AkitaCommitment, AkitaProverData]:
        witness = self.decompose(polys)
        images = self.inner_images(witness)
        commitment = self.outer_commit(images)
        return commitment, AkitaProverData(witness, images)

    def decompose(self, polys: Sequence[Array]) -> Coeff:
        """The batch as a `[blocks, inner_cols]` digit witness — one block's
        digit planes per row.

        Block-major, not the digit-major flattening `decompose_vector` returns,
        because the digit axis is the one the inner `matvec` contracts: `A·s_b`
        needs a block's digits adjacent. The reorder is a host-side read of the
        same rows, not a transpose of anything traced.

        Padding is per polynomial, so a polynomial's blocks are contiguous and
        whole — the property `AkitaConfig.message_lens` exists to keep.
        """
        config = self.config
        degree = config.profile.degree
        if len(polys) != len(config.message_lens):
            raise ValueError(
                f"decompose: expected {len(config.message_lens)} polynomials, "
                f"got {len(polys)}"
            )
        batch = zip(polys, config.message_lens, config.blocks_per_message)
        padded: list[int] = []
        for index, (poly, length, blocks) in enumerate(batch):
            values = _balanced_lift(poly)
            if len(values) != length:
                raise ValueError(
                    f"decompose: polynomial {index} has {len(values)} coefficients, "
                    f"config declares {length}"
                )
            padded.extend(values)
            padded.extend([0] * (blocks * degree - length))

        rows = gadget.decompose_vector(
            padded,
            config.inner_decomposition.log_base,
            config.inner_decomposition.num_digits,
        )
        ring = config.profile.ring
        return ring.stack(
            [
                ring.stack(
                    [
                        ring.from_signed(row[block * degree : (block + 1) * degree])
                        for row in rows
                    ]
                )
                for block in range(config.blocks)
            ]
        )

    def inner_images(self, witness: Coeff) -> Coeff:
        """The per-block images `t_b = A·s_b`, as `[blocks, inner_rows]`.

        Coefficient domain on the way out because that is what the outer tier
        decomposes and what an opening bounds the norm of; the transform in
        between is the ring's, so the value is the same element either way.
        """
        ring = self.config.profile.ring
        return ring.intt(self.inner.commit_batch(self.inner_matrix, ring.ntt(witness)))

    def outer_digits(self, images: Coeff) -> Coeff:
        """`t̂`, the images as one `[outer_cols]` short witness.

        Digit-major, the orientation `decompose_vector` returns and the one
        `AkitaConfig.outer_cols` states: nothing contracts this digit axis, so
        there is no reason to reorder it the way the inner witness is.
        """
        config = self.config
        degree = config.profile.degree
        ring = config.profile.ring
        values = centered_lift("outer_digits", ring, images)
        rows = gadget.decompose_vector(
            values,
            config.outer_decomposition.log_base,
            config.outer_decomposition.num_digits,
        )
        return ring.stack(
            [
                ring.from_signed(row[image * degree : (image + 1) * degree])
                for row in rows
                for image in range(config.images)
            ]
        )

    def outer_commit(self, images: Coeff) -> AkitaCommitment:
        """The public payload `u = B·t̂` for a set of inner images."""
        ring = self.config.profile.ring
        return self.outer.commit(self.outer_matrix, ring.ntt(self.outer_digits(images)))

    def verify(self, commitment: AkitaCommitment, opening: Coeff) -> bool:
        """The digit-level opening predicate: `‖opening‖∞ ≤ β` and the images
        it forces re-commit to `commitment`.

        The outer witness is derived rather than supplied, so both tiers are
        checked against one opening: the inner bound here, the outer bound and
        the re-commitment inside the outer scheme's own predicate. Says nothing
        about which polynomial the digits recompose to — that is `opens_to`.
        """
        config = self.config
        ring = config.profile.ring
        if not within_bound("verify", ring, opening, config.inner_beta_inf):
            return False
        digits = self.outer_digits(self.inner_images(opening))
        return self.outer.verify(self.outer_matrix, commitment, digits)

    def opens_to(
        self, commitment: AkitaCommitment, opening: Coeff, polys: Sequence[Array]
    ) -> bool:
        """The full commitment-level predicate: a valid digit opening that
        recomposes to exactly these polynomials.

        Recomposition is part of the statement and not a caller's afterthought
        — a short witness committing to `u` says nothing on its own, and the
        gap between "some short preimage" and "the digits of *this*
        polynomial" is precisely where a binding argument is lost.
        """
        if not self.verify(commitment, opening):
            return False
        return self.recompose(opening) == [_balanced_lift(poly) for poly in polys]

    def recompose(self, opening: Coeff) -> list[list[int]]:
        """The digit witness back as one balanced-lift coefficient list per
        polynomial, padding dropped.

        Balanced, not reduced: the lift is what the digits were taken of, and
        mapping back into a field is the consumer's step — it is the consumer
        that knows which field.
        """
        config = self.config
        degree = config.profile.degree
        coefficients = centered_lift("recompose", config.profile.ring, opening)
        planes: list[list[int]] = []
        for digit in range(config.inner_cols):
            plane: list[int] = []
            for block in range(config.blocks):
                start = config.column(block, digit) * degree
                plane.extend(coefficients[start : start + degree])
            planes.append(plane)
        values = gadget.recompose_vector(planes, config.inner_decomposition.log_base)
        out: list[list[int]] = []
        offset = 0
        for length, blocks in zip(config.message_lens, config.blocks_per_message):
            out.append(values[offset : offset + length])
            offset += blocks * degree
        return out


def _balanced_lift(poly: Array) -> list[int]:
    """A field-typed coefficient array as balanced integers in
    `[-p/2, p/2]`.

    Balanced rather than the residues in `[0, p)`: the digit count follows the
    magnitude, and the balanced lift halves it — one fewer digit plane, hence
    one fewer block of module width, at every parameter point. Both sides of an
    opening must agree on which lift was decomposed, so it is pinned here.

    `astype(object)` and not `astype(np.uint64)`: the field this commits is the
    consumer's, and a 128-bit one would be truncated by the lane conversion the
    narrower layers below can afford.
    """
    modulus = int(zk_dtypes.pfinfo(poly.dtype).modulus)
    half = modulus >> 1
    residues = [int(value) for value in np.asarray(poly).astype(object).reshape(-1)]
    return [value - modulus if value > half else value for value in residues]


if TYPE_CHECKING:
    _: type[Committer[AkitaCommitment, AkitaProverData]] = AkitaCommitter
