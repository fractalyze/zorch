# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The approximate range proof — `‖(s1, m)‖₂` is small (eprint 2022/284,
§2.4 and §5.1, Fig. 9).

The first layer here that proves something about the *size* of a witness
rather than about an algebraic relation it satisfies, and so the first one
a folding scheme or a verifiable-encryption consumer can actually use. It
is also the last layer built directly on Π_eval^(2): Fig. 10 composes this
with a caller's own relations, and §5 composes it again into exact ℓ2/ℓ∞
bounds.

**Why a projection.** `s = (s1, m)` has `d·(m1 + ℓ)` integer coefficients,
and revealing them is out of the question. Lemma 2.9 says a random
`R ← Bin_1^{256 × d(m1+ℓ)}` shrinks any `⃗s` to 256 integers `R⃗s` whose
norm still bounds `‖⃗s‖` — up to a constant, and only in one direction,
which is exactly what "approximate" names: from `‖R⃗s + ⃗y‖` small the
verifier concludes `‖⃗s‖ ≤ 2√(256/26)·t·γ·√337·β`, a factor ~189 above the
true bound. The counterpart is that the whole statement costs 256 revealed
integers regardless of how long the witness is.

**Why a mask, and why a sign.** `R⃗s` still leaks `⃗s`, so the prover
commits a Gaussian `y ~ D_{s3}^{256/d}` *before* seeing `R` and reveals
`⃗z = b·R⃗s + ⃗y` instead. The secret sign `b ∈ {−1, 1}` makes `⃗z` bimodal,
which lets Rej0 reach the same repetition rate at a much smaller `s3` than
Rej1 would (`masking.BimodalMasking` holds that argument). The sign is not
free: the verifier has to be told `b` is a sign without being told which,
and that is where most of this module's statement comes from.

**What is handed to Π_eval^(2).** Everything, in one shot. The layer below
proves that quadratic functions of the σ-lift vanish, either as ring
elements or in their constant coefficient alone, so all three obligations
are written in that vocabulary (eq. 42–45):

- `F_i(x) = z_i − b·T(⃗r_i, ⃗s) − y_i` for `i ∈ [256]` — the well-formedness
  of the revealed projection. `T` (eq. 16) puts an integer inner product in
  a constant coefficient, so each `F_i` is one *evaluation*. It is
  quadratic, not linear, because `b` and `⃗s` are both committed.
- `G_j(x) = T(⃗δ_j, b)` for `j ∈ [1, d)` — the `j`-th coefficient of `b`,
  which vanishes exactly when `b` is an integer. Also evaluations.
- `f(x) = b² − 1` — the one *relation*, vanishing as a ring element. With
  `b` already known integral, `b² = 1` over the field `Z_q` leaves
  `b ∈ {−1, 1}`.

`F_i`'s constant coefficient only factors as `b·⟨⃗r_i, ⃗s⟩` because `b` is a
constant polynomial — which is what the `G_j` establish. Dropping them
would not fail a round-trip; it would silently prove a statement about a
different quantity, so they are load-bearing rather than hygiene.

**A leg contributes; it does not compose.** The round above — mask, sign,
`R`, `⃗z` — is `ProjectionLeg`, and the single Π_eval^(2) call is
`ApproximateRange`'s, because Fig. 10 (§5.2) runs *two* legs over one
commitment: the verifier sends both `R` after both legs' commitments, gates
both `Rej0` together, and then one inner proof covers everything. A leg
that owned the inner call would have to build a BDLOP matrix and a message
half containing the other leg's rows, and neither leg knows the other.

**Message layout.** `y` and `b` are committed in the BDLOP half beside the
caller's `m` (Fig. 9's `B2` and `b1` rows), so what the layer below opens
is `m‖y‖b`, and what *it* appends on top of that is its own garbage. The
scheme therefore carries `ℓ + legs·(256/d + 1) + λ` messages, and each
layer carves its share off the end — the same "build it over the extended
scheme" contract `GarbageMasking` states, one level up. The composing
layer owns the order and follows Fig. 10's: every mask, then every sign
(`s* := (s2, (s1, x), (m, y^(d), y^(e), b^(d), b^(e)))`), which for one leg
is the `m‖y‖b` Fig. 9 spells.

The caller's `m` arrives as *signed integers* rather than as a ring stack,
unlike every other layer here, and that is deliberate: `m` is half of the
vector whose norm is the statement, so its balanced representatives are
the object being bounded. Taking a ring stack would mean reconstructing
them, and which centred reconstruction a bound reads is a pinned choice in
this codebase (`zorch/commit/ajtai.py`), not a detail to re-decide here.

Fiat-Shamir shape: the prover absorbs every leg's `(t_mask, t_sign)` and
then draws each `R`, so a projection is bound to *all* the first-round
commitments and not merely to its own leg's; every `⃗z` is absorbed before
the inner proof runs, so the statement Π_eval^(2) is given is bound to the
transcript that produced it. `R` is not on the wire — the verifier
re-derives it, which is what checks it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from zorch.byte_transcript import ByteTranscript
from zorch.lnp import wire
from zorch.lnp.challenge import attempt_budget
from zorch.lnp.eval import AbdlopQuadraticEval, QuadraticEvalProof
from zorch.lnp.masking import BimodalMasking
from zorch.lnp.quadratic import (
    AffineImage,
    Family,
    Publics,
    constants,
    lift,
    lift_positions,
    lift_slots,
    require_image,
    sigma_exponent,
    stack_families,
)
from zorch.lnp.transcript import absorb_signed, absorb_stacks

_LABEL_COMMIT = b"lnp/range/mask"
_LABEL_REVEAL = b"lnp/range/projection"

# `Bin_1` costs exactly two transcript bits: a uniform pair `(hi, lo)` maps
# to `lo − hi`, which is `0` twice, `+1` once and `−1` once — the centred
# binomial exactly, and exactly uniform because the support is a power of
# two, so no rejection budget is needed.
#
# Read off the byte stream directly rather than through `uniform_from_bytes`,
# which is the package's uniform source everywhere else. That helper is
# built for a general modulus and spends a full u64 per draw, so a `256 ×
# d(m1+ℓ)` matrix would squeeze 393 KB — and `sample_scalar` re-absorbs what
# it squeezes, so those bytes then sit in the buffer that every *later*
# squeeze in the proof re-hashes per block. Measured: 0.63s → 0.01s across
# one prove's four squeezes, and the suite roughly halves.
_BIN1_BITS = 2

# Both families flattened, which is what `AbdlopQuadraticEval` takes. Spelled
# out rather than left variadic so a miscount is a type error at the call
# rather than a shape error inside the layer below.
_Blocks = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LegDraw:
    """One attempt's secrets for one leg, and the two commitments they
    complete.

    Named rather than returned as a tuple because four of its six fields
    never leave the prover and two of them are the wire, and a caller
    unpacking six same-typed values positionally is one transposition away
    from committing the sign under the mask's rows."""

    sign: int
    y: np.ndarray
    y_ring: np.ndarray
    sign_ring: np.ndarray
    t_mask: np.ndarray
    t_sign: np.ndarray


@dataclass(frozen=True)
class LegMessage:
    """One leg's share of the Fig. 9 wire: its two first-round commitments
    and its revealed projection.

    `R` is absent for the reason every challenge in this package is absent —
    it is Fiat-Shamir output and the verifier re-derives it from every leg's
    `(t_mask, t_sign)`. `⃗z` is *not* derivable and is the whole point of the
    protocol, so it is sent, as signed integers over unreduced ℤ: its norm
    is the statement, and a mod-q representative would not have one."""

    t_mask: np.ndarray
    t_sign: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class RangeProof:
    """One proof: what each leg revealed, and the Π_eval^(2) proof that all
    of it is well formed.

    A sequence rather than one leg's three fields inline, because Fig. 10's
    wire genuinely carries two of each — `t^(d), t^(d)_b, t^(e), t^(e)_b`
    then `⃗z^(d), ⃗z^(e)` — under a single inner proof."""

    legs: tuple[LegMessage, ...]
    evaluation: QuadraticEvalProof


class ProjectionLeg:
    """One Fig. 9 projection round over an `AbdlopQuadraticEval`'s lift —
    the mask, the sign, the challenge `R`, the revealed `⃗z`, and the
    statement they induce — without the inner proof.

    **Why it is told where its rows are.** A leg's mask and sign occupy
    columns of the shared message half, and "the columns after the caller's
    `m`" has exactly one solution, so a leg that derived them could only
    ever be the only leg. Fig. 10's message is `(m, y^(d), y^(e), b^(d),
    b^(e))` — the two legs' masks are adjacent and their signs are
    elsewhere — which no rule local to a leg produces. `mask_slot` and
    `sign_slot` are message-half indices, and they index the BDLOP matrix's
    rows and the lift's message columns alike, since those correspond one
    for one.

    Only the identity automorphism copy is ever indexed: `T` already
    carries the automorphism, so a statement about `σ(x)` written against
    `σ`'s copy would apply it twice.
    """

    def __init__(
        self,
        evaluation: AbdlopQuadraticEval,
        masking: BimodalMasking,
        ell: int,
        mask_slot: int,
        sign_slot: int,
        image: AffineImage | None = None,
    ) -> None:
        scheme = evaluation.scheme
        ring = scheme.ring
        if masking.ring is not ring:
            raise ValueError(
                "range: the masking and the scheme must hold one ring — a "
                "projection masked over a different ring is a parameter bug"
            )
        self.masking = masking
        self.scheme = scheme
        self.mask_slot = mask_slot
        self.sign_slot = sign_slot
        # The eval layer's *width*, not the eval layer. A leg contributes
        # and does not compose — it must never reach the inner `prove` — so
        # what it keeps of that layer is the width its statement is written
        # against, and the carve below, and nothing else. Holding the
        # protocol object would leave the invariant to convention, and
        # re-fusing the inner call into a leg is exactly the regression this
        # split exists to prevent.
        self.width = evaluation.width

        # Positions in the *eval layer's* lift, `[s1, σ(s1), m‖…, σ(m‖…)]`
        # — the message stack is orbited as a whole, so this leg's two
        # additions land inside each automorphism copy exactly as the
        # garbage below does.
        # Off the eval layer's carve, not the scheme's own width: that layer
        # may cover a prefix of the Ajtai half — Fig. 10 commits to `(s1, x)`
        # and writes its statement about `s1` — and these positions index
        # *its* lift. Reading `scheme.s1_cols` here puts the mask and sign
        # past the end of the lift the statement is written against.
        s1_take = evaluation.s1_take
        slots = lift_slots(s1_take, evaluation.ell)
        self._witness_positions = np.concatenate([slots.s1, slots.message[:ell]])
        self._mask_positions = slots.message[mask_slot : mask_slot + masking.mask_cols]
        self._sign_position = int(slots.message[sign_slot])
        # Where an affine image's columns live in the eval layer's lift:
        # `(s1, σ(s1), m, σ(m))` and nothing else. That is the paper's own
        # `2(m1+ℓ)` (§5.2), and carving it here rather than handing `E` the
        # whole width is what makes "an image may not touch the mask or the
        # sign columns" un-violable instead of merely checked — those hold
        # values the prover redraws per attempt, so a function of them is
        # not a statement anyone could have written down in advance.
        self._image_positions = lift_positions(s1_take, s1_take, ell, evaluation.ell)
        self.image = require_image(image, ring, len(self._image_positions))
        # What the projection is *of*, in ring elements: the witness at
        # `E = I`, and `E`'s own row count otherwise. It sizes the challenge
        # matrix on both sides, so a leg that got this wrong would squeeze a
        # different `R` than its verifier and fail to replay rather than
        # prove the wrong thing.
        self._chunks = (s1_take + ell) if image is None else image.rows
        # σ₋₁ applied to each monomial `X^j`, which is what `T(⃗δ_j, ·)`
        # contributes. Row `j` of the identity *is* `X^j`'s coefficient
        # vector, so the ring's own constructor builds the table — writing
        # the residues in place would reach into the backend's array layout,
        # which `constant_coeff`'s docstring says consumers must not.
        self._sigma_monomials = ring.galois(
            ring.from_signed_stack(np.eye(ring.d, dtype=np.int64)),
            sigma_exponent(ring.d),
        )
        # Everything below is fixed at construction and identical on both
        # sides, so it is built once rather than per proof: the `G_j` half of
        # the linear block, and the whole sign relation.
        self._e1 = self._linear_block()
        self._sign_relation = self._relation()

    def bounded_width(self) -> int:
        """Theorem 5.3's `c` — the width, *in integers*, of the vector this
        leg's projection bounds.

        `_chunks` counts it in ring elements; the conditions of §5.2 are
        stated over `Z_q`, where each of those is `d` coefficients. Public
        because it is the one number a consumer of the approximate bound has
        to agree with the leg about: `exact.ExactL2.require_no_wraparound`
        prices Lemma 2.9's precondition against it, and pricing a width the
        leg does not actually bound would gate a statement nobody proved."""
        return self._chunks * self.scheme.ring.d

    # The prover's half. `randomness` is separate from `draw` so the
    # composing layer can hoist it out of its attempt loop: a rejected
    # attempt redraws `(b, y)` but not the witness-only matvecs, which is
    # the hoist `quadratic.prove` makes for the same reason.

    def randomness(
        self, publics: Publics, s2_ring: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """`B2·s2` and `b1·s2` — the halves of this leg's two commitments
        that depend only on the witness."""
        ring = self.scheme.ring
        return (
            ring.matvec(self._mask_rows(publics), s2_ring),
            ring.matvec(self._sign_rows(publics), s2_ring),
        )

    def draw(
        self, randomness: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
    ) -> LegDraw:
        """A fresh `(b, y)` and the two commitments they complete."""
        ring = self.scheme.ring
        mask_randomness, sign_randomness = randomness
        sign, y = self.masking.draw(rng)
        y_ring = ring.from_signed_stack(y)
        sign_ring = constants(ring, [sign])
        return LegDraw(
            sign=sign,
            y=y,
            y_ring=y_ring,
            sign_ring=sign_ring,
            t_mask=ring.add(mask_randomness, y_ring),
            t_sign=ring.add(sign_randomness, sign_ring),
        )

    def project(self, lift: np.ndarray) -> np.ndarray:
        """`E⃗s − ⃗v` (eq. 61) as the balanced integer coefficients `respond`
        contracts `R` against.

        `lift` is `(s1, σ(s1), m, σ(m))` over the caller's own halves —
        `quadratic.lift` of exactly what the image was written against, in
        the order `_image_positions` reads it back. The arithmetic and the
        premise it reads under are `AffineImage.apply`'s; what this adds is
        the refusal below, which belongs to the leg because only a leg has a
        witness-bounding case to be confused with."""
        if self.image is None:
            raise ValueError(
                "range: this leg bounds the witness itself, which reaches "
                "`respond` as the caller's own integers — there is no image "
                "here to project, and returning the witness instead would "
                "hide a caller that thought it had passed one"
            )
        return self.image.apply(self.scheme.ring, lift)

    def respond(
        self,
        draw: LegDraw,
        projection: np.ndarray,
        flat: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """`⃗z = b·R⃗s + ⃗y`, or `None` when Rej0 rejects it.

        `None` rather than a raise or a retry here, because the composing
        layer gates every leg together — Fig. 10 abandons the attempt unless
        *both* legs' Rej0 accept, since one transcript carries them both —
        so a leg's verdict is a value its caller combines, never control
        flow the leg owns.

        `⃗v = b·R⃗s` is the centre Rej0 is stated against, and `⃗z` its mask.
        Both over unreduced ℤ — int64 is exact here, the entries of `R`
        being ternary and `⃗s` short."""
        centre = draw.sign * (projection @ flat)
        revealed = centre + draw.y.reshape(-1)
        if not self.masking.accepts(rng, revealed, centre):
            return None
        return revealed.reshape(draw.y.shape)

    # What both sides replay, in the order they replay it.

    def observe(
        self, transcript: ByteTranscript, t_mask: np.ndarray, t_sign: np.ndarray
    ) -> ByteTranscript:
        """Bind this leg's two first-round commitments."""
        return absorb_stacks(transcript.observe_label(_LABEL_COMMIT), t_mask, t_sign)

    def challenge(
        self, transcript: ByteTranscript
    ) -> tuple[ByteTranscript, np.ndarray]:
        """Squeeze `R ← Bin_1^{256 × d(m1+ℓ)}` — the one derivation both
        sides replay.

        Separate from `observe` because Fig. 10 sends every commitment
        before any challenge, so the composing layer absorbs all of them and
        only then draws each `R`. No label of its own: two legs squeezing
        back to back still get different matrices, since `sample_scalar`
        re-absorbs what it squeezed, and a separator here would move the
        one-leg bytes Fig. 9's suite is pinned against.

        Shaped `(256, d(m1+ℓ))` rather than as a stack of ring elements
        because that is what it is: a `Z`-linear map on coefficient vectors,
        which only becomes ring-shaped once `T` reads it row by row."""
        rows = self.masking.projection
        count = rows * self._chunks * self.scheme.ring.d
        t, raw = transcript.sample_scalar(-(-count * _BIN1_BITS // 8))
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        pairs = bits[: count * _BIN1_BITS].reshape(count, _BIN1_BITS)
        # Subtracted straight into int64 rather than casting both columns
        # first: the cast would materialise a `(count, 2)` int64 temp, which
        # at production width (`d = 128`, ~20 chunks) is 10.5 MB per attempt
        # per leg on each side.
        drawn = np.subtract(pairs[:, 1], pairs[:, 0], dtype=np.int64)
        return t, drawn.reshape(rows, -1)

    def reveal(self, transcript: ByteTranscript, z: np.ndarray) -> ByteTranscript:
        """Bind the revealed projection, which the inner proof runs after."""
        return absorb_signed(
            transcript.observe_label(_LABEL_REVEAL), self.scheme.ring, z
        )

    def statement(self, projection: np.ndarray, z: np.ndarray) -> tuple[Family, Family]:
        """The relations and the evaluations `(R, ⃗z)` induce, in
        `AbdlopQuadraticEval`'s two families.

        One method rather than a copy on each side: the two must build
        byte-identical statements from `(R, ⃗z)` or the inner proof does not
        replay, and that identity is what this layer's soundness rests on."""
        return self._sign_relation, self._evaluations(projection, z)

    def is_well_formed(self, message: LegMessage) -> bool:
        """Whether this leg's share of the wire is structurally usable, per
        `zorch/lnp/wire.py`. `z` routes through the same raising gate as
        every other signed array here rather than restating its predicate,
        which is the rule that module's docstring states."""
        return (
            isinstance(message, LegMessage)
            and wire.is_stack(self.scheme, message.t_mask, self.masking.mask_cols)
            and wire.is_stack(self.scheme, message.t_sign, 1)
            and wire.is_signed(self.scheme.ring, message.z, self.masking.mask_cols)
        )

    def _mask_rows(self, publics: Publics) -> np.ndarray:
        """Fig. 9's `B2` — the BDLOP rows this leg's mask is committed
        under, carved out of the assembled matrix by the slot it was told.
        The slice is the same arithmetic `_mask_positions` is, one axis
        over."""
        return publics.blocks[self.mask_slot : self.mask_slot + self.masking.mask_cols]

    def _sign_rows(self, publics: Publics) -> np.ndarray:
        """Fig. 9's `b1` — the single row this leg's sign is committed
        under.

        Sliced rather than indexed: the layer below takes a stack, and a
        bare row would reach `matvec` one axis short."""
        return publics.blocks[self.sign_slot : self.sign_slot + 1]

    def _evaluations(self, projection: np.ndarray, z: np.ndarray) -> Family:
        """The `256 + (d − 1)` functions whose constant coefficients vanish:
        `F_i` (eq. 43) then `G_j`.

        `F_i = z_i − b·T(⃗r_i, ⃗s) − y_i` splits across all three pieces —
        the `b·T` term is the only quadratic one in this module, and it sits
        at `(b, s_j)` in the quadratic form because `T` contracts `⃗r_i`'s
        `j`-th chunk against `⃗s`'s. `G_j = T(⃗δ_j, b)` is linear in `b`
        alone. Both are rebuilt identically by the verifier, which is what
        binds `⃗z` to the proof.

        With an affine image the projected vector is `⃗e = E⃗s − ⃗v` (eq. 61)
        rather than the witness, and `F_i` becomes eq. 64/65's `z_i −
        b·T(⃗r_i, ⃗e) − y_i`. Only where the `b·T` term lands changes: `E`
        contracts into the quadratic form and `⃗v` — public — falls out of it
        onto the sign. Nothing else in the leg moves, which is why the two
        cases share every other line here."""
        ring = self.scheme.ring
        d = ring.d
        count = self.masking.projection

        # σ₋₁(⃗r_i) chunk by chunk: `T(⃗a, ⃗b) = Σ_i σ₋₁(a_i)·b_i` (eq. 16),
        # so the public half is what carries the automorphism. One
        # `from_signed_stack` over every chunk at once — it is already
        # "`from_signed` per row", so stacking its results by hand would be
        # re-spelling its own batching.
        # `−σ₋₁(⃗r_i)` with the negation taken on the *signed* input, where
        # it is a numpy sign flip, rather than on the residues afterwards,
        # where `ring.neg` is another full coerce-and-reduce pass over
        # `count·chunks·limbs·d`. Byte-identical, and measurably cheaper.
        rows = ring.from_signed_stack((-projection).reshape(-1, d)).reshape(
            count, self._chunks, len(ring.q_moduli), d
        )
        sigma_rows = ring.galois(rows, sigma_exponent(d))

        e2 = ring.zeros(count + d - 1, self.width, self.width)
        e1 = self._e1
        if self.image is None:
            e2[
                np.arange(count)[:, None], self._sign_position, self._witness_positions
            ] = sigma_rows
        else:
            # `T(⃗r_i, E⃗s)` reassociated onto the lift: the coefficient the
            # statement puts on column `k` is `Σ_j σ₋₁(r_{i,j})·E_{j,k}`, so
            # the public half contracts against `E` once and what reaches the
            # quadratic form is a dense row rather than a scatter. `matmul`
            # is the shape this contraction is: `(256, p) × (p, width)`, and
            # its own docstring names a `Bin_1` matrix against a ~32-bit
            # modulus as the case it keeps in `int64`.
            e2[
                np.arange(count)[:, None], self._sign_position, self._image_positions
            ] = ring.matmul(sigma_rows, self.image.matrix)
            # `−b·T(⃗r_i, E⃗s − ⃗v)` splits, and `+b·T(⃗r_i, ⃗v)` is the half
            # with no witness in it: `⃗v` is public, so the offset leaves the
            # quadratic form entirely and lands on the sign alone, linear in
            # `b`. Copied rather than written into `_e1`, which is built once
            # and shared across every attempt and both sides; the rows below
            # `count` are the `G_j` half and are untouched here.
            e1 = e1.copy()
            e1[:count, self._sign_position] = ring.neg(
                ring.matmul(sigma_rows, self.image.offset[:, None])[:, 0]
            )
        e0 = ring.zeros(count + d - 1, 1)
        e0[:count, 0] = constants(ring, z.reshape(-1))
        return e2, e1, e0

    def _linear_block(self) -> np.ndarray:
        """The linear half of both families, which depends on neither `R`
        nor `⃗z` and is therefore built once.

        `F_i` contributes `−σ₋₁(X^{i mod d})` on the mask element holding
        `y_i`; `G_j` contributes `σ₋₁(X^j)` on the sign, unnegated, because
        it reads the sign's own coefficients rather than subtracting them."""
        ring = self.scheme.ring
        d = ring.d
        count = self.masking.projection
        e1 = ring.zeros(count + d - 1, self.width)
        index = np.arange(count)
        e1[index, self._mask_positions[index // d]] = ring.neg(
            self._sigma_monomials[index % d]
        )
        e1[count + np.arange(d - 1), self._sign_position] = self._sigma_monomials[1:]
        return e1

    def _relation(self) -> Family:
        """`f(b) = b² − 1`, the one function claimed zero as a ring element.

        Paired with the `G_j` above, not usable without them: `b² = 1` alone
        admits any square root of one in `R_q`, and it is integrality that
        cuts those down to `±1`."""
        ring = self.scheme.ring
        width = self.width
        r2 = ring.zeros(1, width, width)
        r2[0, self._sign_position, self._sign_position] = ring.one()
        r1 = ring.zeros(1, width)
        r0 = ring.neg(ring.one())[None, None]
        return r2, r1, r0


class ApproximateRange:
    """Fig. 9's prove/verify over an `AbdlopQuadraticEval`, and the shape
    Fig. 10 composes: `n` `ProjectionLeg`s, one inner proof.

    Built over the eval layer rather than beside it, because the projection
    statement is not provable on its own: `F_i`, `G_j` and `f` are what a
    verifier checks, and only Π_eval^(2) can check them together against one
    commitment. What this layer contributes is the round schedule — every
    leg's commitments absorbed, then every `R` drawn, then every `⃗z`
    responded and gated together — the message half's order, and the single
    call below.

    **The scheme it needs.** `evaluation` must already be built over a
    scheme whose BDLOP half carries `ℓ + legs·(256/d + 1) + λ` messages;
    `self.ell` is what is left for the caller after every leg's mask and
    sign and the layer below's garbage. A scheme sized for the caller's `m`
    alone fails here rather than silently proving a statement about a
    shorter witness.

    **What is proven, and what is not.** A verifying proof says
    `‖(s1, m)‖₂ ≤ 2√(256/26)·t·γ·√337·β` for the `β` the caller derived
    each `mask_std` from — a bound roughly 189β, not β. Anything needing the
    tight bound composes this with §5's exact proof; this layer is where
    that composition gets its "no wraparound mod q" premise.

    Commit-and-prove, not zero-knowledge over a reusable commitment — §3.2,
    inherited from the layer below and made stronger here, since this layer
    appends a `y` and a `b` per leg to the message on every run.
    """

    def __init__(
        self,
        evaluation: AbdlopQuadraticEval,
        *maskings: BimodalMasking,
        images: Sequence[AffineImage | None] = (),
    ) -> None:
        if not maskings:
            raise ValueError(
                "range: a range proof needs at least one projection leg, and "
                "none was given"
            )
        if images and len(images) != len(maskings):
            raise ValueError(
                f"range: {len(images)} affine image(s) against "
                f"{len(maskings)} leg(s) — Fig. 10 pairs each `(E_i, v_i)` "
                f"with the leg that bounds it, so pass one per leg, `None` "
                f"where a leg bounds the witness itself, or none at all"
            )
        masks = sum(masking.mask_cols for masking in maskings)
        rows = masks + len(maskings)
        ell = evaluation.ell - rows
        if ell < 0:
            raise ValueError(
                f"range: the BDLOP half leaves {evaluation.ell} messages to "
                f"the layers above the garbage, too few for {len(maskings)} "
                f"leg(s) needing {rows} on top of a message vector — build it "
                f"over the extended scheme"
            )
        self.evaluation = evaluation
        self.scheme = evaluation.scheme
        self.ell = ell
        # Fig. 10's order for the legs' share of the message half: every
        # mask, then every sign. For one leg that is Fig. 9's `m‖y‖b`.
        mask_slot = ell
        sign_slot = ell + masks
        # The budget first, because it is what each leg's Gaussian sampler
        # has to be sized for — see `_joint_budget` and
        # `BimodalMasking.for_attempts`.
        self.attempts = _joint_budget(maskings)
        legs = []
        for masking, image in zip(
            maskings, images or [None] * len(maskings), strict=True
        ):
            legs.append(
                ProjectionLeg(
                    evaluation,
                    masking.for_attempts(self.attempts),
                    ell,
                    mask_slot,
                    sign_slot,
                    image,
                )
            )
            mask_slot += masking.mask_cols
            sign_slot += 1
        self.legs = tuple(legs)

    def prove(
        self,
        publics: Publics,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
        rng: np.random.Generator,
        transcript: ByteTranscript,
        relations: Family | None = None,
        evaluations: Family | None = None,
    ) -> tuple[RangeProof, ByteTranscript]:
        """One non-interactive proof that `‖(s1, m)‖₂` is within the bound
        each leg's masking was parameterised for.

        Fig. 9's `B2` and `b1` — the BDLOP rows a leg's mask and sign are
        committed under — are carved out of `publics.blocks` rather than
        passed beside it; see `ProjectionLeg._mask_rows`. `s1`, `s2` and
        `message` are signed integer arrays: the first two as everywhere in
        this package, the third because it is half of the vector being
        bounded.

        `relations` and `evaluations` are the caller's own `f_i` and `F_i`
        over the eval layer's width, proved in the same shot as the range
        statement. Fig. 10 takes exactly those alongside its two legs and
        packs them into `ϕ` and `Ψ`, and a caller that instead ran them as a
        second proof would hold two parameter points for one witness and
        could not see them drift — the reason `AbdlopQuadraticEval` proves
        its own two families together one layer down.

        They must not touch the mask or sign columns: those hold values the
        prover draws per attempt, so a function of them is not a statement
        the caller can have written down.
        """
        ring = self.scheme.ring
        publics.require(self.scheme)
        self._require_witness(s1, s2, message)
        s2_ring = ring.from_signed_stack(s2)
        message_ring = ring.from_signed_stack(message) if self.ell else ring.zeros(0)
        bounded = self._bounded(s1, message, message_ring)
        randomness = [leg.randomness(publics, s2_ring) for leg in self.legs]

        for _ in range(self.attempts):
            draws = [
                leg.draw(hoisted, rng)
                for leg, hoisted in zip(self.legs, randomness, strict=True)
            ]
            t, projections = self._round(
                transcript, [(draw.t_mask, draw.t_sign) for draw in draws]
            )
            responses = [
                leg.respond(draw, projection, vector, rng)
                for leg, draw, projection, vector in zip(
                    self.legs, draws, projections, bounded, strict=True
                )
            ]
            # One gate over every leg, not one loop per leg: Fig. 10 abandons
            # the attempt unless *every* Rej0 accepts, because one transcript
            # carries them all and a leg that kept its accepted `⃗z` while a
            # sibling redrew would be revealing a projection of a witness
            # under a challenge the redrawn transcript no longer produces.
            revealed = [z for z in responses if z is not None]
            if len(revealed) != len(responses):
                continue

            t, families = self._statement(t, projections, revealed)
            inner, t = self.evaluation.prove(
                publics,
                *self._families(families, relations, evaluations),
                s1,
                s2,
                _ordered(
                    message_ring,
                    [draw.y_ring for draw in draws],
                    [draw.sign_ring for draw in draws],
                ),
                rng,
                t,
            )
            return (
                RangeProof(
                    legs=tuple(
                        LegMessage(t_mask=draw.t_mask, t_sign=draw.t_sign, z=z)
                        for draw, z in zip(draws, revealed, strict=True)
                    ),
                    evaluation=inner,
                ),
                t,
            )
        raise self._exhausted()

    def verify(
        self,
        publics: Publics,
        t_a: np.ndarray,
        t_b: np.ndarray,
        proof: RangeProof,
        transcript: ByteTranscript,
        relations: Family | None = None,
        evaluations: Family | None = None,
    ) -> tuple[bool, ByteTranscript]:
        """Fig. 9's two checks: every `‖⃗z‖₂` is within the Prop. 5.1 bound,
        and the Π_eval^(2) proof of the statement they induce verifies.

        `t_b` is the caller's commitment to `m` alone; the mask and sign
        commitments arrive on the proof and are appended here, in the order
        the message was built. `relations` and `evaluations` are the
        prover's, and are part of the statement rather than of the proof —
        a verifier given different ones checks a different claim, and the
        inner proof simply fails to replay."""
        # The statement is the caller's and raises; the proof is the
        # prover's and is a verdict. See `zorch/lnp/wire.py`.
        publics.require(self.scheme)
        if not self._is_well_formed(proof):
            return False, transcript
        if not all(
            leg.masking.within_bounds(message.z)
            for leg, message in zip(self.legs, proof.legs, strict=True)
        ):
            return False, transcript

        t, projections = self._round(
            transcript, [(m.t_mask, m.t_sign) for m in proof.legs]
        )
        t, families = self._statement(
            t, projections, [message.z for message in proof.legs]
        )
        return self.evaluation.verify(
            publics,
            *self._families(families, relations, evaluations),
            t_a,
            _ordered(
                t_b,
                [m.t_mask for m in proof.legs],
                [m.t_sign for m in proof.legs],
            ),
            proof.evaluation,
            t,
        )

    def _round(
        self,
        transcript: ByteTranscript,
        commitments: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[ByteTranscript, list[np.ndarray]]:
        """Fig. 10's first two moves: every leg's `(t_mask, t_sign)` bound,
        and only then every leg's `R` drawn.

        One method rather than a copy on each side, for the reason
        `_statement` gives — the two must replay byte for byte or the inner
        proof does not verify — and the reason the leg keeps `observe` and
        `challenge` apart at all: a projection is bound to *all* the
        first-round commitments, not merely its own leg's."""
        t = transcript
        for leg, (t_mask, t_sign) in zip(self.legs, commitments, strict=True):
            t = leg.observe(t, t_mask, t_sign)
        projections = []
        for leg in self.legs:
            t, projection = leg.challenge(t)
            projections.append(projection)
        return t, projections

    def _bounded(
        self, s1: np.ndarray, message: np.ndarray, message_ring: np.ndarray
    ) -> list[np.ndarray]:
        """Each leg's own vector over ℤ — `E_i⃗s − ⃗v_i` (eq. 61) for a leg
        carrying an affine image, and the witness itself for one without.

        Hoisted out of the attempt loop for the same reason `randomness` is:
        it is a function of the witness alone, and what an attempt redraws is
        `⃗y` and `b`.

        The imageless path stays the caller's own integers rather than a
        round-trip through the ring. The values agree — `(s1, m)` are
        canonical and short — but the bound is stated about *those*, and
        reading them back out of `R_q` would put Fig. 9's own case one
        wraparound premise deeper than it needs to be. `s1` is carved to what
        the eval layer's statement covers: the bound is about that prefix,
        and `R` is squeezed to its width."""
        flat = (
            np.concatenate([s1[: self.evaluation.s1_take], message])
            .astype(np.int64)
            .reshape(-1)
        )
        if all(leg.image is None for leg in self.legs):
            return [flat] * len(self.legs)
        ring = self.scheme.ring
        witness = lift(
            ring, ring.from_signed_stack(s1[: self.evaluation.s1_take]), message_ring
        )
        return [
            flat if leg.image is None else leg.project(witness) for leg in self.legs
        ]

    def _statement(
        self,
        transcript: ByteTranscript,
        projections: Sequence[np.ndarray],
        responses: Sequence[np.ndarray],
    ) -> tuple[ByteTranscript, list[tuple[Family, Family]]]:
        """Absorb every revealed projection, then collect the families they
        induce, one `(relations, evaluations)` pair per leg.

        Returned per leg rather than pre-stacked because `_families` has to
        concatenate anyway to put the caller's ahead of them, and stacking
        here would copy every block twice — `e2` alone is
        `(256 + d − 1, width, width)`, the largest array this layer builds.

        Every `⃗z` is bound before any statement is built, which is Fig. 10's
        order and — for one leg — Fig. 9's unchanged. The two sides must
        reach byte-identical statements here or the inner proof does not
        replay."""
        t = transcript
        for leg, z in zip(self.legs, responses, strict=True):
            t = leg.reveal(t, z)
        return t, [
            leg.statement(projection, z)
            for leg, projection, z in zip(
                self.legs, projections, responses, strict=True
            )
        ]

    def _families(
        self,
        legs: Sequence[tuple[Family, Family]],
        relations: Family | None,
        evaluations: Family | None,
    ) -> _Blocks:
        """The caller's two families ahead of the legs' own, flattened into
        the six blocks `AbdlopQuadraticEval` takes.

        **`ϕ` matches Fig. 10 exactly.** Eq. 68 is
        `ϕ = (f_1, …, f_ρ, g^(d), g^(e))` — the caller's relations, then one
        sign relation per leg — which is what the relation half assembles.

        **`Ψ` does not, deliberately.** Eq. 69 groups it
        `((F_i), G, (H_j^(d)), (H_j^(e)), (I_i), (J_j^(i)))` — every
        projection function together, then every coefficient function —
        where this yields `[caller's, leg 0's (H then J), leg 1's (H then
        J), …]`. Matching would mean the caller handing over `F`, `G` and
        `I` as three separately-placed groups so the composer could
        interleave them, which is a materially more complex seam.

        It buys nothing. `Ψ` is aggregated by `Γ`, a fresh uniform matrix
        drawn after the functions are fixed, so any permutation of it is the
        same statement with the columns of `Γ` relabelled — and both sides
        build the order here, so they cannot disagree. The paper's grouping
        is presentational. What *is* load-bearing, and is preserved, is that
        the caller's functions come first and each leg's stay contiguous, so
        a reader can find a given obligation.

        One pass over everything, caller and legs together: stacking the
        legs first and the caller's onto that would copy every leg's blocks
        twice."""
        r2, r1, r0 = stack_families(_ahead(relations, [family for family, _ in legs]))
        e2, e1, e0 = stack_families(_ahead(evaluations, [ev for _, ev in legs]))
        return r2, r1, r0, e2, e1, e0

    def _exhausted(self) -> RuntimeError:
        """The composition's own `exhausted`: the budget it raises against is
        the joint one, so it cannot defer to a single leg's message."""
        return RuntimeError(
            f"range.prove: every attempt was rejected — {self.attempts} "
            f"attempts at all {len(self.legs)} leg(s) accepting together fail "
            f"together with negligible probability, so suspect the parameters "
            f"(a repetition rate far from its Lemma 2.14-3 value), not bad luck."
        )

    def _require_witness(
        self, s1: np.ndarray, s2: np.ndarray, message: np.ndarray
    ) -> None:
        """The witness halves plus the message, all signed — the statement
        side of `zorch/lnp/wire.py`."""
        self.evaluation.require_witness("range.prove", s1, s2)
        wire.require_signed(self.scheme.ring, "range.prove: message", message, self.ell)

    def _is_well_formed(self, proof: RangeProof) -> bool:
        """Whether `proof` is structurally usable — every field of it, in one
        place, per `zorch/lnp/wire.py`. The per-leg share defers to the leg
        that owns it, and the count is checked before the zip so a proof
        carrying the wrong number of legs is a verdict rather than a
        `ValueError` out of `strict=True`."""
        return (
            isinstance(proof, RangeProof)
            and isinstance(proof.legs, tuple)
            and len(proof.legs) == len(self.legs)
            and all(
                leg.is_well_formed(message)
                for leg, message in zip(self.legs, proof.legs)
            )
            and self.evaluation._is_well_formed(proof.evaluation)
        )


def _joint_budget(maskings: Sequence[BimodalMasking]) -> int:
    """The attempt budget for the joint gate.

    Every leg redraws when *any* leg's Rej0 rejects, so one attempt succeeds
    with probability `∏ 1/rep0_i` — the product rule `Masking` already
    applies to its own two Gaussians. `fail_prob` is the tightest the legs
    were built with, since the budget has to satisfy each of them. For one
    leg this is that leg's own number, unchanged, which is why Fig. 9 never
    had to say any of this.

    Module-level so a caller sizing a parameter point can ask what a
    composition of it will need without building the composition first —
    the number was otherwise derivable only by reading `ApproximateRange`
    and re-spelling the rule."""
    accept = math.prod(1.0 / masking.rep0 for masking in maskings)
    return attempt_budget(min(m.fail_prob for m in maskings), accept)


def _ordered(
    head: np.ndarray, masks: Sequence[np.ndarray], signs: Sequence[np.ndarray]
) -> np.ndarray:
    """The legs' share of a stack in the order the message half is built:
    the caller's own rows, then every leg's mask, then every leg's sign.

    Named because three things have to agree on it — the BDLOP matrix the
    caller assembles, the message the prover concatenates, and the
    commitment the verifier reassembles — and a round-trip cannot see them
    disagree, since both sides build the last two the same way. Fig. 10
    writes it as `(m, y^(d), y^(e), b^(d), b^(e))`; for one leg it is
    Fig. 9's `m‖y‖b`."""
    return np.concatenate([head, *masks, *signs])


def _ahead(caller: Family | None, legs: Sequence[Family]) -> Sequence[Family]:
    """The caller's family in front of the legs', or the legs' alone.

    Spelled out rather than tested inline, because `if caller` on a tuple of
    three ring stacks reads as an array truth test and is one edit away from
    becoming one."""
    return list(legs) if caller is None else [caller, *legs]
