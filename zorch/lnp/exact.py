# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The exact ℓ2 bound — `‖(s1, m)‖ ≤ β` with β *tight* (eprint 2022/284,
§5.2, eq. 53).

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

**`E = I`.** Fig. 10 states eq. 53 for arbitrary `‖E_i s − v_i‖ ≤ β_i`, an
affine image of the lift. This module implements the case the paper leads
with — the witness itself — because that is what a folding or
verifiable-encryption consumer asks for first, and because a general `E`
costs `256·k·width` ring multiplications on the host oracle, which is a
pure-Python O(d²) schoolbook by design. Everything below is written so the
general case is a widening rather than a rewrite: the bounded vector is
named by *positions in the lift*, and `E = I` is the case where those
positions are the witness's own.
"""

from __future__ import annotations

import math

import numpy as np
from lattice_frx.split_ring import HostSplitRing

from zorch.lnp.eval import AbdlopQuadraticEval
from zorch.lnp.quadratic import lift_slots


class ExactL2:
    """The two evaluations that turn Fig. 9's approximate bound into
    `‖(s1, m)‖ ≤ β` exactly.

    `message_cols` is the caller's own `m`, the same number
    `ApproximateRange.ell` reports — not the eval layer's `ell`, which also
    counts the range legs' mask and sign rows. The bounded vector is
    `(s1, m)`: the Ajtai half minus the digits this layer appends to it, and
    the message half's caller-owned prefix.

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
        message_cols: int,
        bound: int,
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
        witness_cols = evaluation.s1_take - _DIGIT_COLS
        if witness_cols < 1 or message_cols < 0:
            raise ValueError(
                f"exact: the Ajtai half carries {evaluation.s1_take} columns, "
                f"too few for a witness plus the {_DIGIT_COLS} the binary "
                f"decomposition needs — widen the scheme's `s1_cols`"
            )
        self.evaluation = evaluation
        self.ring = ring
        self.bound = bound
        self.digits = digits
        self.witness_cols = witness_cols
        self.message_cols = message_cols

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
        self._sigma_digit = slots.sigma_s1[witness_cols:]
        # `p⃗ = (1, 2, 4, …, 2^{digits-1}, 0, …)` — eq. 59's radix row, which
        # reads the digits back as the number they encode. Public, so it is
        # the half that carries the automorphism, exactly as `T`'s first
        # argument does everywhere in this package.
        radix = np.zeros((1, ring.d), dtype=np.int64)
        radix[0, :digits] = 1 << np.arange(digits)
        self._radix = ring.galois(ring.from_signed_stack(radix), 2 * ring.d - 1)[0]
        # `1⃗` over one ring element: `Σ_t X^t`, the all-ones coefficient
        # vector Lemma 5.2 subtracts.
        self._ones = ring.from_signed_stack(np.ones((1, ring.d), dtype=np.int64))[0]

    def decompose(self, witness: np.ndarray) -> np.ndarray:
        """The digits `x⃗` of `β² − ‖(s1, m)‖²`, as a signed-integer stack the
        caller appends to its Ajtai half.

        `witness` is `(s1, m)` over ℤ — the same balanced integers whose norm
        is the statement, so the caller passes what it committed rather than
        a reconstruction. Raises when the witness is over the bound, since
        the slack has no binary representation then and a silently wrong
        proof is the alternative."""
        flat = np.asarray(witness, dtype=object).reshape(-1)
        slack = self.bound**2 - int(sum(int(v) * int(v) for v in flat))
        if slack < 0:
            raise ValueError(
                f"exact: the witness has ‖·‖² = {self.bound**2 - slack}, past "
                f"the bound's {self.bound**2} — there is no exact proof of a "
                f"false statement to build"
            )
        out = np.zeros((_DIGIT_COLS, self.ring.d), dtype=np.int64)
        out[0, : self.digits] = [(slack >> i) & 1 for i in range(self.digits)]
        return out

    def evaluations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(G, I)` as the `(e2, e1, e0)` family `AbdlopQuadraticEval` takes.

        Both are *evaluations* and not relations: what is claimed is that
        their constant coefficient vanishes, which by Lemma 2.4 is exactly
        the inner-product statement each encodes. Claiming them as ring
        elements would be a strictly stronger and false statement — the
        other coefficients of `T(x⃗, x⃗ − 1⃗)` are whatever the digits make
        them."""
        ring = self.ring
        width = self.evaluation.width
        e2 = ring.zeros(2, width, width)
        e1 = ring.zeros(2, width)
        e0 = ring.zeros(2, 1)

        # G(x⃗) = T(x⃗, x⃗ − 1⃗) = Σ σ₋₁(x_i)·x_i − Σ σ₋₁(x_i)·1⃗   (eq. 63)
        e2[0, self._sigma_digit, self._digit] = ring.one()
        e1[0, self._sigma_digit] = ring.neg(self._ones)

        # I(s, x⃗) = T(s, s) + T(p⃗, x⃗) − β²                      (eq. 66)
        e2[1, self._sigma_witness, self._witness] = ring.one()
        e1[1, self._digit] = self._radix
        e0[1, 0] = ring.neg(_constant(ring, self.bound**2))
        return e2, e1, e0

    def require_no_wraparound(self, projection_bound: int) -> None:
        """Theorem 5.3's two conditions on the range leg that carries this
        proof from `Z_q` to `ℤ`.

        `projection_bound` is `B`, the ℓ2 bound the approximate leg actually
        proves about `(s ‖ x⃗)`. The exact statements are proved mod q, and
        an integer identity that wrapped is not an integer identity — these
        are what rule that out:

        - `B² + √((υe + k_bin)·d)·B < q` makes `⟨x', x' − 1⃗⟩ = 0 mod q` hold
          over ℤ, since both terms of the inner product are then bounded well
          inside one period.
        - `2β² + B² − 1 < q` does the same for `⟨s, s⟩ + ⟨p⃗, x⃗⟩ = β²`.

        Checked rather than assumed because a violation is not an error
        anywhere — it is a proof of a statement about residues that reads
        like a proof about integers."""
        modulus = math.prod(self.ring.q_moduli)
        span = math.isqrt(_DIGIT_COLS * self.ring.d)
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


# One ring element holds the decomposition, per §5.2's `2·log(β) ≤ d`. Named
# because it is the `υe = 1` this module implements, and every layout below
# counts in it.
_DIGIT_COLS = 1


def _constant(ring: HostSplitRing, value: int) -> np.ndarray:
    """`value` as the constant polynomial holding it.

    Through `from_signed` because `β²` is an unreduced integer and reducing
    it into `Z_q` is what that constructor owns."""
    row = np.zeros((1, ring.d), dtype=np.int64)
    row[0, 0] = value
    return ring.from_signed_stack(row)[0]
