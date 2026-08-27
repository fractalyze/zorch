# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""§5.2's two statements: the exact ℓ2 bound — `‖(s1, m)‖ ≤ β` with β
*tight* (eprint 2022/284, eq. 53) — and eq. 54's binarity, which `ExactL2`
builds its own digits through and a caller can ask for over any committed
columns.

Everything before this module proves an **approximate** bound: `range.py`
reveals a 256-integer projection and concludes `‖⃗s‖ ≤ 2√(256/26)·t·γ·√337·β`,
a factor ~189 above the truth. That slack is inherent to Lemma 2.9 and no
amount of care inside Fig. 9 removes it. §5.2 removes it a different way —
by not bounding the norm at all, but *committing* to how much room is left
under the bound and proving that number is a nonnegative integer.

**The trick.** `‖s‖ ≤ β` over the integers is the same statement as
"`β² − ‖s‖²` is a nonnegative integer", and a nonnegative integer below `β²`
is exactly one with a binary representation. So the prover commits the
binary decomposition `x⃗` of `β² − ‖s‖²` and proves two things about it:

- `⟨x⃗, x⃗ − 1⃗⟩ = 0`, which forces `x⃗` binary. Every term is `x_t(x_t − 1)`,
  which is nonnegative over ℤ and zero only at `0` and `1`, so the sum
  vanishes only when every term does (Lemma 5.2). One evaluation.
- `⟨s, s⟩ + ⟨p⃗, x⃗⟩ = β²`, where `p⃗ = (1, 2, 4, …)` reads the digits back as
  a number. One evaluation.

Together: `‖s‖² = β² − ⟨p⃗, x⃗⟩` and `⟨p⃗, x⃗⟩ ≥ 0`, hence `‖s‖ ≤ β`, with no
slack anywhere.

**Why it still needs an approximate proof underneath.** Both statements are
proved *modulo q*, and neither conclusion survives a wraparound: over `Z_q`
a "nonnegative integer" is not a thing. §5.2 closes that by bounding
`‖(s ‖ x⃗)‖ ≤ B` with the very Fig. 9 machinery this layer is built to
improve on — approximate is enough there, because all it has to rule out is
that the two inner products wrapped. `require_no_wraparound` below states
the conditions that makes sufficient. So the exact proof does not replace
the approximate one; it *consumes* it.

**What this module is, and is not.** It is a statement builder, not a
protocol: it hands back the two evaluations in `AbdlopQuadraticEval`'s
vocabulary and the caller passes them to `ApproximateRange.prove` alongside
the range statement, which is what Fig. 10 does — its `Ψ` carries `G` and
`I_i` next to the legs' own obligations. Nothing here absorbs a transcript
or draws randomness.

**`E` is general.** Fig. 10 states eq. 53 for arbitrary
`‖E_i s − v_i‖ ≤ β_i`, an affine image of the lift, and `ExactL2` takes one:
pass a `quadratic.AffineImage` and eq. 66's inner product is written about
`E⃗s − ⃗v`. `None` keeps the case the paper leads with, the witness itself,
which is what a folding or verifiable-encryption consumer asks for first.

This was deferred once, on the grounds that a general `E` cost `256·k·width`
ring multiplications against a pure-Python O(d²) schoolbook. That accounting
was about the *range* leg — `T(⃗r_i, E⃗s)` for 256 rows — and it stopped being
true when `lattice_frx` grew a batched `matmul`: the contraction became one
`int64` sum over `(cols, d)` with the anticirculant built once and reused, a
measured 112–116x over the `matvec`-per-column spelling it replaces. Here the
cost is not per row at all. `T(E⃗s, E⃗s)` is a single `σ₋₁(E)ᵀ·E` contraction
folded into a block that is built once per object and shared by prove and
verify, so a general `E` costs the same per proof as `E = I` does.

What the two cases do *not* share is density. At `E = I` the quadratic block
is a diagonal scatter over the witness columns; a general `E` fills the whole
`2(m1+ℓ)` square, because `σ₋₁` of a combination reaches the σ copy of
everything the combination touched. That is inherent to the statement rather
than to this implementation.

Eq. 54's `E_bin` is still a selection — see `binarity`, which says why its
`Sequence[int]` columns are the restriction and not an oversight.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

# `υe = 1`: §5.2 sizes the ring degree so `2·log(β) ≤ d`, one ring element
# holds the decomposition, and the `digits > ring.d` gate below is what
# enforces it. Not a widening knob — the radix row and `decompose` are both
# written for a single column, so raising this alone would lose digits
# silently. A second exact-ℓ2 statement needs a layout, not a bigger number.
_DIGITS = 1

import numpy as np
from lattice_frx import gadget, norms
from lattice_frx.split_ring import HostSplitRing

from zorch.lnp.eval import AbdlopQuadraticEval
from zorch.lnp.quadratic import (
    SIGMA_ORDER,
    AffineImage,
    Family,
    constants,
    lift_pairing,
    lift_positions,
    lift_slots,
    require_image,
    sigma_exponent,
    stack_families,
)

if TYPE_CHECKING:
    from zorch.lnp.range import ProjectionLeg


def binarity(
    evaluation: AbdlopQuadraticEval,
    s1_columns: Sequence[int] = (),
    message_columns: Sequence[int] = (),
) -> Family:
    """Eq. 54's `G` — the named committed columns hold binary coefficients.

    Lemma 5.2 in one evaluation: `⟨v⃗, v⃗ − 1⃗⟩ = 0` forces `v⃗ ∈ {0,1}`,
    because every term `v_t(v_t − 1)` is nonnegative over ℤ and vanishes
    only at 0 and 1, so the sum vanishes only when each term does.

    **Over ℤ, and it is proved over `Z_q`.** The argument is an integer one
    — nonnegativity is not a `Z_q` notion — so a caller owes the same
    wraparound premise the exact-ℓ2 proof owes, and gets it the same way:
    the columns named here must be inside a vector the range leg bounds. At
    `E_bin` a *selection* of already-committed columns, which is what this
    implements, that premise is inherited rather than re-proved — the
    columns are part of the witness the leg already projects.

    Columns are named per half because the two halves are separate 0-based
    index spaces with different bounds (`s1_take` against `ell`); one flat
    sequence would make the caller know the lift layout, which `lift_slots`
    owns. Both automorphism copies of each column are read — `T(v⃗, ·)` puts
    `σ₋₁(v)` on one side of the product, so a statement about `v` touches
    `σ(v)`'s slot too — which is why the body builds `identity` and `sigma`
    together.

    **Two restrictions, not one.** `E_bin` is a *selection*, which the
    `Sequence[int]` parameters make un-violable: a general affine image would
    turn this into a dense quadratic form over the whole lift. `ExactL2`
    takes one of those for eq. 53 and eq. 54 could follow the same way;
    nothing has asked for it, so this is scope rather than a gap.

    Note what that costs the paragraph above. The premise is inherited only
    while the leg projects the witness columns these names select — true of
    the `E = I` composition, and *not* true once `ExactL2` carries an image,
    since the leg then bounds `(E⃗s − ⃗v ‖ x⃗)`, which need not contain them.
    A caller pairing binary columns with an imaged exact proof owes the
    premise itself, by appending selecting rows to the composed image.

    `v_bin = 0` is the separate one, pinned by the `ring.zeros(1, 1)` below
    and enforced by nothing — a caller wanting `v⃗` in `{k, k+1}` for public
    `k` has no knob for it.
    """
    ring = evaluation.scheme.ring
    slots = lift_slots(evaluation.s1_take, evaluation.ell)
    s1_take, message = list(s1_columns), list(message_columns)
    for name, chosen, available in (
        ("s1", s1_take, evaluation.s1_take),
        ("message", message, evaluation.ell),
    ):
        if any(not 0 <= c < available for c in chosen):
            raise ValueError(
                f"exact: a binary {name} column outside the {available} the "
                f"statement covers: {chosen}"
            )
    if not s1_take and not message:
        raise ValueError(
            "exact: binarity needs a column to be about — an empty claim is "
            "an evaluation that vanishes for every witness"
        )
    identity = np.concatenate([slots.s1[s1_take], slots.message[message]])
    sigma = np.concatenate([slots.sigma_s1[s1_take], slots.sigma_message[message]])

    width = evaluation.width
    e2 = ring.zeros(1, width, width)
    e1 = ring.zeros(1, width)
    e2[0, sigma, identity] = ring.one()
    e1[0, sigma] = ring.neg(_ones(ring))
    return e2, e1, ring.zeros(1, 1)


def _ones(ring: HostSplitRing) -> np.ndarray:
    """`1⃗` over one ring element — `Σ_t X^t`, the all-ones coefficient
    vector Lemma 5.2 subtracts. Not `ring.one()`, which is the
    multiplicative identity and has a single nonzero coefficient."""
    return ring.from_signed_stack(np.ones((1, ring.d), dtype=np.int64))[0]


class ExactL2:
    """The two evaluations that turn Fig. 9's approximate bound into
    `‖(s1, m)‖ ≤ β` exactly.

    `witness_cols` and `message_cols` name the lift this statement is
    written over — `(s1, m)`, which is also the bounded vector when no
    `image` is given and the columns `E` is indexed against when one is:
    how much of the Ajtai half is witness (the rest of `evaluation.s1_take`
    being the digits appended here), and how much of the message half is the
    caller's own `m` — the number `ApproximateRange.ell` reports, not the
    eval layer's `ell`, which also counts the range legs' mask and sign rows.

    Both are *told*, not derived, for the reason `ProjectionLeg` is told its
    slots: "the columns after the witness" has one solution only while there
    is one contributor to the Ajtai half. `E_bin` binarity adds a second.
    Deriving `witness_cols = s1_take - 1` also read `s1_take` the opposite
    way from `AbdlopQuadraticEval`, whose carve *excludes* the digits — and
    a caller who followed that reading would have indexed its last witness
    column as the digit slot and built a verifiable proof of a different
    statement, with no shape error anywhere.

    **The digits live in the Ajtai half.** They are a small-norm secret, so
    they belong where small-norm secrets go — and footnote 14 of the paper
    is explicit that this only works if they are committed *with* `s1`,
    since an Ajtai commitment is not extendable after the fact. A caller
    that cannot know `x⃗` at commitment time has to put it in the BDLOP half
    instead, which this module does not implement.
    """

    def __init__(
        self,
        evaluation: AbdlopQuadraticEval,
        witness_cols: int,
        message_cols: int,
        bound: int,
        image: AffineImage | None = None,
    ) -> None:
        ring = evaluation.scheme.ring
        if bound <= 0:
            raise ValueError(f"exact: the bound must be positive, got {bound!r}")
        # `2·log(β) ≤ d` (§5.2): the slack `β² − ‖s‖²` is below `β²`, so its
        # binary representation fits `β²`'s bit length, and the paper sizes
        # the ring degree so that is one ring element. A wider bound is a
        # real parameter point — it just needs the `υe > 1` layout this
        # module has not been asked for yet.
        digits = int(bound**2 - 1).bit_length()
        if digits > ring.d:
            raise ValueError(
                f"exact: a bound of {bound} needs {digits} binary digits, "
                f"past the ring degree {ring.d} — §5.2 sizes `d` so one ring "
                f"element holds the decomposition"
            )
        if witness_cols < 1:
            raise ValueError(f"exact: need a witness column, got {witness_cols!r}")
        if not 0 <= message_cols <= evaluation.ell:
            raise ValueError(
                f"exact: a {message_cols}-column message half against the "
                f"{evaluation.ell} the statement covers — this is a *count*, "
                f"and a slice would silently prove a prefix while `c` under-"
                f"counts the projected width by the columns it dropped"
            )
        if witness_cols + _DIGITS >= evaluation.s1_take + 1:
            raise ValueError(
                f"exact: a {witness_cols}-column witness plus the {_DIGITS} the "
                f"decomposition needs does not fit the {evaluation.s1_take} "
                f"Ajtai columns the statement covers — widen `s1_cols`"
            )
        self.ring = ring
        self.bound = bound
        self.digits = digits
        self.witness_cols = witness_cols
        self.message_cols = message_cols
        # The eval layer's *width*, not the eval layer: this is a statement
        # builder and must never reach the inner `prove`, the same invariant
        # `ProjectionLeg` keeps structurally rather than by convention.
        self.width = evaluation.width
        # The Ajtai carve, kept because `range_image` needs it: the leg this
        # statement rides writes its `E` over `(s1‖x, m)`, so composing the
        # two images means knowing how wide that half is. `witness_cols +
        # _DIGITS` is not the same number — the constructor only requires it
        # to fit, and a caller may carve a wider half than this statement
        # fills.
        self._s1_take = evaluation.s1_take

        slots = lift_slots(evaluation.s1_take, evaluation.ell)
        # `(s1, m)`, and its σ image alongside: every inner product here is
        # `Σ σ₋₁(v_i)·v_i`, so both copies are read at once.
        self._witness = np.concatenate(
            [slots.s1[:witness_cols], slots.message[:message_cols]]
        )
        self._sigma_witness = np.concatenate(
            [slots.sigma_s1[:witness_cols], slots.sigma_message[:message_cols]]
        )
        self._digit = slots.s1[witness_cols:]
        # Where an image's columns live: the caller's own halves and both
        # their σ copies, which is the paper's `2(m1+ℓ)` — the digits are
        # *not* among them. Fig. 10 commits the Ajtai half as `(s1, x)` but
        # writes `E_i` over `s` alone, and `x` reaches the statement through
        # the radix row below instead.
        self._image_positions = lift_positions(
            witness_cols, evaluation.s1_take, message_cols, evaluation.ell
        )
        # Every term of `T(·, ·)` reads a position and its σ partner, and
        # once `E` mixes columns the partner of a combination is only
        # reachable through the permutation.
        self._pairing = lift_pairing(witness_cols, message_cols)
        self.image = require_image(image, ring, len(self._image_positions))
        # `p⃗ = (1, 2, 4, …, 2^{digits-1}, 0, …)` — eq. 59's radix row, which
        # reads the digits back as the number they encode. Public, so it is
        # the half that carries the automorphism, exactly as `T`'s first
        # argument does everywhere in this package.
        radix = np.zeros((1, ring.d), dtype=np.int64)
        radix[0, :digits] = 1 << np.arange(digits)
        self._radix = ring.galois(
            ring.from_signed_stack(radix), sigma_exponent(ring.d)
        )[0]
        # The digits' own binarity, through the shared builder: `x⃗` is
        # exactly the `x'` of eq. 63 when no `E_bin` columns join it. Named as
        # the digits rather than as "the columns after the witness" — the two
        # coincide only while the digits are the *only* other contributor to
        # the Ajtai half, which is precisely what this class's docstring says
        # `E_bin` binarity retires. A local, not a field: `_build` is the only
        # reader and runs below, while keeping it would hold a second copy of
        # a ~10 MB block for the object's lifetime.
        digits_binarity = binarity(
            evaluation, s1_columns=range(witness_cols, witness_cols + _DIGITS)
        )
        # Witness-independent and identical on both sides, so it is built
        # once — the same reason `ProjectionLeg` caches `_e1` and its sign
        # relation. At the paper's width this block is ~10 MB, and prove and
        # verify each ask for it.
        self._evaluations = self._build(digits_binarity)

    def decompose(self, witness: np.ndarray) -> np.ndarray:
        """The digits `x⃗` of `β² − ‖(s1, m)‖²`, as a signed-integer stack the
        caller appends to its Ajtai half.

        `witness` is the bounded vector over ℤ — `(s1, m)` itself with no
        image, and `E⃗s − ⃗v` with one, which is what
        `range.ProjectionLeg.project` computes. Either way the caller passes
        the balanced integers whose norm is the statement rather than a
        reconstruction of them. Raises when they are over the bound, since
        the slack has no binary representation then and a silently wrong
        proof is the alternative."""
        slack = self.bound**2 - norms.l2_squared(witness)
        if slack < 0:
            raise ValueError(
                f"exact: the witness has ‖·‖² = {self.bound**2 - slack}, past "
                f"the bound's {self.bound**2} — there is no exact proof of a "
                f"false statement to build"
            )
        out = np.zeros((_DIGITS, self.ring.d), dtype=np.int64)
        out[0, : self.digits] = gadget.decompose_unsigned(slack, 1, self.digits)
        return out

    def evaluations(self) -> Family:
        """`(G, I)` as the `(e2, e1, e0)` family `AbdlopQuadraticEval` takes.

        The same blocks every call: they depend on the public parameters
        alone, and nothing downstream writes into them — `_embed` copies
        into its own wider arrays."""
        return self._evaluations

    def _build(self, digits_binarity: Family) -> Family:
        """The two functions, laid out over the eval layer's lift.

        Both are *evaluations* and not relations: what is claimed is that
        their constant coefficient vanishes, which by Lemma 2.4 is exactly
        the inner-product statement each encodes. Claiming them as ring
        elements would be a strictly stronger and false statement — the
        other coefficients of `T(x⃗, x⃗ − 1⃗)` are whatever the digits make
        them."""
        ring = self.ring
        width = self.width
        # I(s, x⃗) = T(E⃗s − ⃗v, E⃗s − ⃗v) + T(p⃗, x⃗) − β²          (eq. 66)
        e2 = ring.zeros(1, width, width)
        e1 = ring.zeros(1, width)
        e0 = ring.zeros(1, 1)
        if self.image is None:
            e2[0, self._sigma_witness, self._witness] = ring.one()
            constant = ring.neg(constants(ring, [self.bound**2])[0])
        else:
            constant = self._image_terms(self.image, e2, e1)
        e1[0, self._digit] = self._radix
        e0[0, 0] = constant
        # `G` first, matching eq. 69's order within what this builder owns.
        return stack_families([digits_binarity, (e2, e1, e0)])

    def _image_terms(
        self, image: AffineImage, e2: np.ndarray, e1: np.ndarray
    ) -> np.ndarray:
        """`T(E⃗s − ⃗v, E⃗s − ⃗v)` written into the two blocks it spreads
        across, and the public remainder it leaves behind.

        The inner product is bilinear, so it splits four ways:

            T(E⃗s − ⃗v, E⃗s − ⃗v)
                = T(E⃗s, E⃗s) − T(E⃗s, ⃗v) − T(⃗v, E⃗s) + T(⃗v, ⃗v)

        The first is quadratic in the lift, the middle two are linear in it,
        and the last has no witness in it at all. `E = I, ⃗v = 0` collapses
        every one of them back to `e2[σ(c), c] = 1` on the witness columns,
        which is what the branch above spells directly.

        **Why the pairing.** `T(⃗a, ⃗b) = Σ σ₋₁(a_i)·b_i`, so the left factor
        of every term is read from the σ copy of wherever the right one
        lives. At `E = I` that is `_sigma_witness` beside `_witness`; once
        `E` mixes columns, `σ₋₁` of a *combination* spreads over the σ copies
        of everything the combination touched, and the permutation is the
        only thing that still says where.

        Written into the caller's blocks rather than returned, because `e2`
        is `width²` ring elements — ~10 MB at the paper's point — and a
        second one to merge would double the peak for a block that is
        already built once per object lifetime."""
        ring = self.ring
        matrix, offset = image.matrix, image.offset
        exponent = sigma_exponent(ring.d)
        sigma_matrix = ring.galois(matrix, exponent)
        sigma_offset = ring.galois(offset, exponent)
        positions = self._image_positions
        # Where `σ₋₁` of each column's contents lives, in the same order.
        partner = positions[self._pairing]

        # T(E⃗s, E⃗s): the coefficient pairing `σ₋₁(s_c)` with `s_c'` is
        # `Σ_k σ₋₁(E_kc)·E_kc'`, one contraction of `σ₋₁(E)ᵀ` against `E`.
        # Both index arrays are permutations of the same column set, so this
        # covers each `(c, c')` exactly once — no entry is written twice, and
        # the form stays the plain bilinear sum the `E = I` branch builds
        # rather than a symmetrised one.
        e2[0, partner[:, None], positions[None, :]] = ring.matmul(
            np.swapaxes(sigma_matrix, 0, 1), matrix
        )

        # −T(E⃗s, ⃗v) lands on the σ copies and −T(⃗v, E⃗s) on the identity
        # ones, so they are summed in column space and scattered once — the
        # two index sets are the same positions in a different order, and
        # assigning them separately would have the second overwrite the
        # first.
        e1[0, positions] = ring.neg(
            ring.add(
                ring.matmul(np.swapaxes(matrix, 0, 1), sigma_offset[:, None])[:, 0],
                ring.matmul(np.swapaxes(sigma_matrix, 0, 1), offset[:, None])[:, 0][
                    self._pairing
                ],
            )
        )

        # `T(⃗v, ⃗v) − β²`, both public. The whole ring element and not its
        # constant coefficient: only `I`'s constant coefficient is claimed to
        # vanish, and dropping the rest here would be claiming the others
        # were zero to begin with.
        return ring.sub(
            ring.matmul(sigma_offset[None, :], offset[:, None])[0, 0],
            constants(ring, [self.bound**2])[0],
        )

    def composed_cols(self) -> int:
        """The width, in ring elements, of the vector eq. 61 composes — what
        a leg carrying this statement has to bound.

        One derivation, read by `range_image` below and by
        `require_no_wraparound` under it. It used to be two: the leg sized
        its challenge matrix off its image's rows while the wraparound check
        re-derived `(image.rows + _DIGITS)` on its own, and nothing made the
        two agree."""
        if self.image is None:
            # `(s1‖x, m)` — the leg's own carve, digits included.
            return self._s1_take + self.message_cols
        return self.image.rows + _DIGITS

    def range_image(self) -> AffineImage | None:
        """Eq. 61's `e⃗^(e) = [E⃗s − ⃗v ; x⃗]` — the image the range leg
        carrying this statement must be built with, or `None` when the leg
        should bound the witness directly.

        The two layers write their images over *different lifts*, and that
        is the whole content of this method. `E` here is indexed against
        `(s1, σ(s1), m, σ(m))` without the digits, the paper's `2(m1+ℓ)`,
        while the leg's lift has them — Fig. 10 commits the Ajtai half as
        `(s1, x)`. So every column shifts, and `_DIGITS` rows are appended
        to select `x⃗` itself.

        Without this the leg would bound `(s1‖x, m)` while eq. 66's inner
        product is written about `E⃗s − ⃗v`, and Theorem 5.3's wraparound
        premise would be about a different vector than the one it is claimed
        for — which nothing downstream would notice, since both proofs
        verify. That is why this is public API rather than something each
        caller re-derives.

        `None` at `E = I` is not an omission: the imageless leg bounds
        `(s1‖x, m)`, whose coefficients are exactly the composed vector's in
        a different order, so the norm the leg proves is the norm eq. 61
        needs. Building an identity image here would cost a contraction to
        say the same thing — the same trade `AffineImage` documents."""
        if self.image is None:
            return None
        ring = self.ring
        ell = self.message_cols
        # Where this statement's narrow lift sits inside the leg's wider
        # one. Both halves carve from the head of each automorphism copy,
        # which is where a layer that *appends* leaves the caller's columns.
        columns = lift_positions(self.witness_cols, self._s1_take, ell, ell)
        wide = lift_slots(self._s1_take, ell)
        rows = self.image.rows
        # Through `composed_cols` rather than `rows + _DIGITS`: that number
        # is what the wraparound check prices, and spelling it twice is how
        # the composition and its premise drifted apart before.
        matrix = ring.zeros(self.composed_cols(), SIGMA_ORDER * (self._s1_take + ell))
        matrix[:rows, columns] = self.image.matrix
        # The digits enter as a selection, not through `E`: they are what
        # eq. 59's radix row reads, and the leg has to bound them because
        # `⟨p⃗, x⃗⟩ ≥ 0` is the half of §5.2 that makes the bound exact.
        matrix[
            rows + np.arange(_DIGITS),
            wide.s1[self.witness_cols : self.witness_cols + _DIGITS],
        ] = ring.one()
        offset = ring.zeros(self.composed_cols())
        offset[:rows] = self.image.offset
        return AffineImage(matrix=matrix, offset=offset)

    def require_no_wraparound(self, leg: ProjectionLeg, binary_cols: int = 0) -> None:
        """Theorem 5.3's conditions on the range leg that carries this proof
        from `Z_q` to `ℤ`.

        `leg` is that leg, and both numbers the conditions need are read off
        it rather than passed in: `B = leg.masking.proven_norm()`, the ℓ2
        bound Lemma 2.9 concludes about the projected vector, and
        `c = leg.bounded_width()`, that vector's width in integers. Taking
        the leg is what makes the check about the proof actually being
        built. A `B` and a `c` supplied by the caller describe a leg that
        may not exist, and every way of getting them wrong yields a gate
        that passes.

        `binary_cols` is Thm 5.3's `k_bin`, the width of the `E_bin`
        statement's own binary vector: zero when no `binarity(...)` columns
        ride along in the same statement, and a parameter rather than an
        assumption because it belongs to the *composed* statement and not to
        this object. The leg is expected to bound those columns too, so they
        count in the width agreement below.

        **An ℓ∞ leg is refused here**, by `proven_norm` raising. Fig. 10
        runs two legs and only the ℓ2 one, `(e)`, carries an exact
        statement: Theorem 5.3 states all three conditions over `B^(e)`
        alone, because they exist to keep an integer identity proved mod `q`
        from wrapping, and the ℓ∞ leg's eq. 52 is not one. Deriving an ℓ2
        bound from the ℓ∞ gate via `√n` would also cost `21.8×` against the
        `9.7×` this parameter point has to spare — see `masking.LinfBound`.

        The exact statements are proved mod q, and an integer identity that
        wrapped is not an integer identity — these are what rule that out:

        - `B < q/(41·c)` is Lemma 2.9's own precondition (`b ≤ P/(41m)`),
          without which the projection says nothing at all.
        - `B² + √((υe + k_bin)·d)·B < q` makes `⟨x', x' − 1⃗⟩ = 0 mod q` hold
          over ℤ, since both terms of the inner product are then bounded well
          inside one period.
        - `2β² + B² − 1 < q` does the same for `⟨s, s⟩ + ⟨p⃗, x⃗⟩ = β²`.

        Checked rather than assumed because a violation is not an error
        anywhere — it is a proof of a statement about residues that reads
        like a proof about integers.

        What the width agreement does *not* check is that the leg's image is
        this statement's `range_image()` entry for entry — only that it
        bounds a vector of the right width.

        That is the reachable failure. Handing a leg this statement's own
        `E` is already impossible: it is written over `2(m1+ℓ)` and the
        leg's lift is `2(s1_take+ℓ)`, so `require_image` refuses it on
        shape. What survives that is an image over the *right* lift with the
        wrong row count — a caller writing the leg's `E` by hand and leaving
        off the rows that select `x⃗`. An `E` of the right shape about the
        wrong columns stays a statement the caller wrote down and this
        object never sees."""
        if binary_cols < 0:
            raise ValueError(
                f"exact: the binary-column width cannot be negative, got "
                f"{binary_cols!r}"
            )
        if leg.scheme.ring is not self.ring:
            raise ValueError(
                "exact: the leg and this statement must hold one ring — a "
                "bound proved over a different modulus prices nothing here"
            )
        # Raises for Fig. 10's ℓ∞ leg, which proves no ℓ2 bound to price.
        projection_bound = leg.masking.proven_norm()
        width = leg.bounded_width()
        expected = (self.composed_cols() + binary_cols) * self.ring.d
        if width != expected:
            raise ValueError(
                f"exact: this leg bounds {width} integers, but eq. 61's "
                f"composition over this statement is {expected} — the leg "
                f"has to be built with `range_image()` (plus any "
                f"`binary_cols`), or its bound is about a different vector "
                f"than the one Theorem 5.3's conditions are priced for"
            )
        modulus = math.prod(self.ring.q_moduli)
        span = math.isqrt((_DIGITS + binary_cols) * self.ring.d)
        if 41 * width * projection_bound >= modulus:
            raise ValueError(
                f"exact: Lemma 2.9 needs B = {projection_bound} < "
                f"q/(41·c) = {modulus}/{41 * width}, or the "
                f"projection bounds nothing at all — raise q, or shrink the "
                f"projected vector"
            )
        for name, value in (
            ("the binarity check", projection_bound**2 + span * projection_bound),
            ("the norm identity", 2 * self.bound**2 + projection_bound**2 - 1),
        ):
            if value >= modulus:
                raise ValueError(
                    f"exact: {name} needs {value} < q = {modulus}, so the "
                    f"identity it proves modulo q need not hold over ℤ — "
                    f"raise q, or tighten the projection bound B"
                )
