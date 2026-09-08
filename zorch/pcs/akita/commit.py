# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The module-lattice commitment: decompose to short digits, then SIS-commit.

`zorch/commit/ajtai.py` binds a witness that is *already* short. A polynomial
over a prime field is not — its coefficients fill the field — so the scheme
layer is exactly the step that makes one short: balanced base-`2^w` digits,
laid out as ring elements, committed as one `t = A·s`. Binding then says
something about the polynomial rather than about an arbitrary small vector,
because the digits recompose to it uniquely.

Two properties survive the decomposition and are why this shape is worth the
digit blow-up. The commitment stays **additively homomorphic** in the digits,
which is what a folding consumer needs; and one commitment binds the **whole
batch**, since every polynomial's digits are columns of one module vector —
the same single-root economy BaseFold gets from a matrix commitment, reached a
different way.

**Where the host boundary sits.** The digit decomposition is exact integer
arithmetic over the balanced lift: a field element's lift does not fit a lane,
and lattice-frx pins lifts, norms and `gadget` to the host for that reason.
So `commit` materialises its input, decomposes on the host, and hands a ring
element back to the traced `matvec`. That is a real cost and a real limit, not
a stylistic choice — a traced per-limb decomposition is the substrate's own
deferred step, and this layer is written so that arriving would change one
function rather than the layout around it.

The public matrix rides the committer, as a KZG proving key rides its prover:
it is the scheme instance's parameter, not a per-call argument. Nothing here
samples it — the CRS expansion from a seed is the consumer's, matching the
randomness stance of the commitment algebra below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import zk_dtypes
from frx import Array
from lattice_frx import gadget, rns
from lattice_frx.ring import Coeff, Eval

from zorch.commit.ajtai import AjtaiCommitment
from zorch.pcs.akita.config import AkitaConfig
from zorch.pcs.stage import Committer

# The `[rows]` module vector `A·s`. An alias, not a wrapper: it is a ring
# element the consumer observes into a transcript and folds like any other
# (`pcs.md`, "Commitments are aliases until they grow structure").
AkitaCommitment: TypeAlias = Eval


@dataclass(frozen=True)
class AkitaProverData:
    """The digit witness `commit` retained for the opening.

    Prover-only, and the reason the seam splits committer from opener at all:
    the witness is the polynomial, in a representation an opening can multiply
    by a challenge without leaving the norm bound.
    """

    witness: Coeff


class AkitaCommitter:
    """`commit(polys) = A·decompose(polys)` and its opening predicate.

    Configuration rides the constructor — the parameter point, the module
    height and the batch shape — with the public matrix beside it. Binding
    strength is the consumer's parameter choice; this layer enforces shapes,
    the digit bound, and exact recomposition.
    """

    def __init__(self, config: AkitaConfig, matrix: Eval) -> None:
        self.config = config
        self.matrix = matrix
        self.scheme = AjtaiCommitment(
            config.profile.ring, config.rows, config.cols, config.beta_inf
        )

    def commit(self, polys: Sequence[Array]) -> tuple[AkitaCommitment, AkitaProverData]:
        witness = self.decompose(polys)
        ring = self.config.profile.ring
        commitment = self.scheme.commit(self.matrix, ring.ntt(witness))
        return commitment, AkitaProverData(witness)

    def decompose(self, polys: Sequence[Array]) -> Coeff:
        """The batch as one `[cols]` digit witness, in the config's
        digit-major column order.

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

        # Digit-major rows, each already one digit plane across the whole
        # batch — the orientation `decompose_vector` returns, which is why the
        # column formula transposes nothing.
        rows = gadget.decompose_vector(
            padded, config.decomposition.log_base, config.decomposition.num_digits
        )
        ring = config.profile.ring
        return ring.stack(
            [
                ring.from_signed(row[block * degree : (block + 1) * degree])
                for row in rows
                for block in range(config.blocks)
            ]
        )

    def verify(self, commitment: AkitaCommitment, opening: Coeff) -> bool:
        """The digit-level opening predicate: `‖opening‖∞ ≤ β` and it
        re-commits. Says nothing about which polynomial the digits recompose
        to — that is `opens_to`."""
        return self.scheme.verify(self.matrix, commitment, opening)

    def opens_to(
        self, commitment: AkitaCommitment, opening: Coeff, polys: Sequence[Array]
    ) -> bool:
        """The full commitment-level predicate: a valid digit opening that
        recomposes to exactly these polynomials.

        Recomposition is part of the statement and not a caller's afterthought
        — a short witness committing to `t` says nothing on its own, and the
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
        coefficients = self._centered(opening)
        planes: list[list[int]] = []
        for digit in range(config.decomposition.num_digits):
            plane: list[int] = []
            for block in range(config.blocks):
                start = config.column(digit, block) * degree
                plane.extend(coefficients[start : start + degree])
            planes.append(plane)
        values = gadget.recompose_vector(planes, config.decomposition.log_base)
        out: list[list[int]] = []
        offset = 0
        for length, blocks in zip(config.message_lens, config.blocks_per_message):
            out.append(values[offset : offset + length])
            offset += blocks * degree
        return out

    def _centered(self, opening: Coeff) -> list[int]:
        """The witness's balanced lift, flat in `[column, coefficient]` order.

        The full-chain reconstruction rather than limb 0's, for the reason the
        commitment algebra's own bound check gives: a single-limb lift accepts
        residues whose other limbs disagree, so the two would answer differently
        about the same opening.
        """
        if not isinstance(opening, Coeff):
            raise TypeError(
                f"recompose: the witness is a coefficient-domain element "
                f"(digits are coefficients), got {type(opening).__name__}"
            )
        host = np.stack(
            [np.asarray(limb).astype(np.uint64).reshape(-1) for limb in opening.limbs]
        )
        return rns.reconstruct_centered(host, self.config.profile.moduli)


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
