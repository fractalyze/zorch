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

**Message layout.** `y` and `b` are committed in the BDLOP half beside the
caller's `m` (Fig. 9's `B2` and `b1` rows), so what the layer below opens
is `m‖y‖b`, and what *it* appends on top of that is its own garbage. The
scheme therefore carries `ℓ + 256/d + 1 + λ` messages, and each layer
carves its share off the end — the same "build it over the extended
scheme" contract `GarbageMasking` states, one level up.

The caller's `m` arrives as *signed integers* rather than as a ring stack,
unlike every other layer here, and that is deliberate: `m` is half of the
vector whose norm is the statement, so its balanced representatives are
the object being bounded. Taking a ring stack would mean reconstructing
them, and which centred reconstruction a bound reads is a pinned choice in
this codebase (`zorch/commit/ajtai.py`), not a detail to re-decide here.

Fiat-Shamir shape: the prover absorbs `(t_y, t_b)` and receives `R`, then
absorbs `⃗z` before the inner proof runs, so the statement Π_eval^(2) is
given is bound to the transcript that produced it. `R` is not on the wire —
the verifier re-derives it, which is what checks it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.lnp import wire
from zorch.lnp.eval import AbdlopQuadraticEval, QuadraticEvalProof
from zorch.lnp.masking import BimodalMasking
from zorch.lnp.quadratic import SIGMA_ORDER
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


@dataclass(frozen=True)
class RangeProof:
    """The Fig. 9 wire: the two extra commitments, the revealed projection,
    and the Π_eval^(2) proof of its well-formedness.

    `R` is absent for the reason every challenge in this package is absent —
    it is Fiat-Shamir output and the verifier re-derives it from `(t_mask,
    t_sign)`. `⃗z` is *not* derivable and is the whole point of the protocol,
    so it is sent, as signed integers over unreduced ℤ: its norm is the
    statement, and a mod-q representative would not have one."""

    t_mask: np.ndarray
    t_sign: np.ndarray
    z: np.ndarray
    evaluation: QuadraticEvalProof


class ApproximateRange:
    """Fig. 9 prove/verify over an `AbdlopQuadraticEval`.

    Built over the eval layer rather than beside it, because the projection
    statement is not provable on its own: `F_i`, `G_j` and `f` are what a
    verifier checks, and only Π_eval^(2) can check them together against one
    commitment. This layer's own contribution is the `(b, y, R, ⃗z)` round
    and the norm gate on `⃗z`.

    **The scheme it needs.** `evaluation` must already be built over a
    scheme whose BDLOP half carries `ℓ + 256/d + 1 + λ` messages; `self.ell`
    is what is left for the caller after this layer's mask and sign and the
    layer below's garbage. A scheme sized for the caller's `m` alone fails
    here rather than silently proving a statement about a shorter witness.

    **What is proven, and what is not.** A verifying proof says
    `‖(s1, m)‖₂ ≤ 2√(256/26)·t·γ·√337·β` for the `β` the caller derived
    `mask_std` from — a bound roughly 189β, not β. Anything needing the
    tight bound composes this with §5's exact proof; this layer is where
    that composition gets its "no wraparound mod q" premise.

    Commit-and-prove, not zero-knowledge over a reusable commitment — §3.2,
    inherited from the layer below and made stronger here, since this layer
    appends `y` and `b` to the message on every run.
    """

    def __init__(
        self,
        evaluation: AbdlopQuadraticEval,
        masking: BimodalMasking,
    ) -> None:
        scheme = evaluation.scheme
        ring = scheme.ring
        if masking.ring is not ring:
            raise ValueError(
                "range: the masking and the scheme must hold one ring — a "
                "projection masked over a different ring is a parameter bug"
            )
        ell = evaluation.ell - masking.mask_cols - 1
        if ell < 0:
            raise ValueError(
                f"range: the BDLOP half leaves {evaluation.ell} messages to "
                f"the layers above the garbage, too few for a {masking.mask_cols}"
                f"-element mask and a sign on top of a message vector — build "
                f"it over the extended scheme"
            )
        self.evaluation = evaluation
        self.masking = masking
        self.scheme = scheme
        self.ell = ell

        # Positions in the *eval layer's* lift, `[s1, σ(s1), m, y, b, σ(m),
        # σ(y), σ(b)]` — the message stack is orbited as a whole, so this
        # layer's two additions land inside each automorphism copy exactly
        # as the garbage below does. Only the identity copy is ever indexed:
        # `T` already carries the automorphism, so a statement about `σ(x)`
        # written against `σ`'s copy would apply it twice.
        # Off the eval layer's carve, not the scheme's own width: that layer
        # may cover a prefix of the Ajtai half — Fig. 10 commits to `(s1, x)`
        # and writes its statement about `s1` — and these positions index
        # *its* lift. Reading `scheme.s1_cols` here puts the mask and sign
        # past the end of the lift the statement is written against.
        s1_take = evaluation.s1_take
        s1_span = SIGMA_ORDER * s1_take
        self._witness_positions = np.concatenate(
            [np.arange(s1_take), s1_span + np.arange(ell)]
        )
        self._mask_positions = s1_span + ell + np.arange(masking.mask_cols)
        self._sign_position = s1_span + ell + masking.mask_cols
        self._chunks = s1_take + ell
        # σ₋₁ applied to each monomial `X^j`, which is what `T(⃗δ_j, ·)`
        # contributes. Row `j` of the identity *is* `X^j`'s coefficient
        # vector, so the ring's own constructor builds the table — writing
        # the residues in place would reach into the backend's array layout,
        # which `constant_coeff`'s docstring says consumers must not.
        self._sigma_monomials = ring.galois(
            ring.from_signed_stack(np.eye(ring.d, dtype=np.int64)), 2 * ring.d - 1
        )
        # Everything below is fixed at construction and identical on both
        # sides, so it is built once rather than per proof: the `G_j` half of
        # the linear block, and the whole sign relation.
        self._e1 = self._linear_block()
        self._sign_relation = self._relation()

    def prove(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        b_mask: np.ndarray,
        b_sign: np.ndarray,
        bg: np.ndarray,
        b_quad: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
        rng: np.random.Generator,
        transcript: ByteTranscript,
    ) -> tuple[RangeProof, ByteTranscript]:
        """One non-interactive proof that `‖(s1, m)‖₂` is within the bound
        `masking` was parameterised for.

        `b_mask` and `b_sign` are Fig. 9's `B2` and `b1`, the BDLOP rows the
        mask and the sign are committed under — distinct from the message's
        `b`, from `bg`, and from Fig. 6's `b_quad`. `s1`, `s2` and `message`
        are signed integer arrays: the first two as everywhere in this
        package, the third because it is half of the vector being bounded.
        """
        ring = self.scheme.ring
        self._require_statement(b, b_mask, b_sign)
        self._require_witness(s1, s2, message)
        s2_ring = ring.from_signed_stack(s2)
        message_ring = ring.from_signed_stack(message) if self.ell else ring.zeros(0)
        # The projected witness, over ℤ: `⃗s` is the concatenated balanced
        # coefficients of `(s1, m)`, which is the vector Lemma 2.9 bounds.
        # `s1` is carved to what the eval layer's statement covers — the
        # bound is about that prefix, and `R` is squeezed to its width.
        flat = (
            np.concatenate([s1[: self.evaluation.s1_take], message])
            .astype(np.int64)
            .reshape(-1)
        )
        blocks = self._blocks(b, b_mask, b_sign)
        # Witness-only, so a rejected attempt would recompute them unchanged
        # — the hoist `quadratic.prove` makes for the same reason. Only the
        # `add` of the fresh mask is per-attempt.
        mask_randomness = ring.matvec(b_mask, s2_ring)
        sign_randomness = ring.matvec(b_sign, s2_ring)

        for _ in range(self.masking.attempts):
            sign, y = self.masking.draw(rng)
            y_ring = ring.from_signed_stack(y)
            sign_ring = _constants(ring, [sign])
            t_mask = ring.add(mask_randomness, y_ring)
            t_sign = ring.add(sign_randomness, sign_ring)

            advanced, projection = self._challenge(transcript, t_mask, t_sign)
            # `v = b·R⃗s` is the centre Rej0 is stated against, and `z` its
            # mask. Both over unreduced ℤ — int64 is exact here, the entries
            # of `R` being ternary and `⃗s` short.
            centre = sign * (projection @ flat)
            revealed = centre + y.reshape(-1)
            if not self.masking.accepts(rng, revealed, centre):
                continue

            z = revealed.reshape(y.shape)
            t, (r2, r1, r0, e2, e1, e0) = self._statement(advanced, projection, z)
            inner, t = self.evaluation.prove(
                a1,
                a2,
                blocks,
                bg,
                b_quad,
                r2,
                r1,
                r0,
                e2,
                e1,
                e0,
                s1,
                s2,
                np.concatenate([message_ring, y_ring, sign_ring]),
                rng,
                t,
            )
            return (
                RangeProof(t_mask=t_mask, t_sign=t_sign, z=z, evaluation=inner),
                t,
            )
        raise self.masking.exhausted("range.prove")

    def verify(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        b_mask: np.ndarray,
        b_sign: np.ndarray,
        bg: np.ndarray,
        b_quad: np.ndarray,
        t_a: np.ndarray,
        t_b: np.ndarray,
        proof: RangeProof,
        transcript: ByteTranscript,
    ) -> tuple[bool, ByteTranscript]:
        """Fig. 9's two checks: `‖⃗z‖₂` is within the Prop. 5.1 bound, and
        the Π_eval^(2) proof of the statement `⃗z` induces verifies.

        `t_b` is the caller's commitment to `m` alone; the mask and sign
        commitments arrive on the proof and are appended here, in the order
        the message was built."""
        # The statement is the caller's and raises; the proof is the
        # prover's and is a verdict. See `zorch/lnp/wire.py`.
        self._require_statement(b, b_mask, b_sign)
        if not self._is_well_formed(proof):
            return False, transcript
        if not self.masking.within_bounds(proof.z):
            return False, transcript

        advanced, projection = self._challenge(transcript, proof.t_mask, proof.t_sign)
        t, (r2, r1, r0, e2, e1, e0) = self._statement(advanced, projection, proof.z)
        return self.evaluation.verify(
            a1,
            a2,
            self._blocks(b, b_mask, b_sign),
            bg,
            b_quad,
            r2,
            r1,
            r0,
            e2,
            e1,
            e0,
            t_a,
            np.concatenate([t_b, proof.t_mask, proof.t_sign]),
            proof.evaluation,
            t,
        )

    def _statement(
        self, transcript: ByteTranscript, projection: np.ndarray, z: np.ndarray
    ) -> tuple[
        ByteTranscript,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Absorb the revealed projection, then the six blocks it induces.

        One method rather than a copy on each side: the two must build
        byte-identical statements from `(R, ⃗z)` or the inner proof does not
        replay, and that identity is what this layer's soundness rests on.
        The order is `AbdlopQuadraticEval`'s own — one relation, then the
        evaluations."""
        t = absorb_signed(transcript.observe_label(_LABEL_REVEAL), self.scheme.ring, z)
        return t, (*self._sign_relation, *self._evaluations(projection, z))

    def _blocks(
        self, b: np.ndarray, b_mask: np.ndarray, b_sign: np.ndarray
    ) -> np.ndarray:
        """The BDLOP matrix the layer below opens `m‖y‖b` against.

        Named because the order here, the order the message is concatenated
        in, and the order `_mask_positions`/`_sign_position` were derived in
        are one contract asserted in three places — and a round-trip cannot
        see them disagree, since both sides build them the same way."""
        return np.concatenate([b, b_mask, b_sign])

    def _challenge(
        self, transcript: ByteTranscript, t_mask: np.ndarray, t_sign: np.ndarray
    ) -> tuple[ByteTranscript, np.ndarray]:
        """Absorb the mask and sign commitments and squeeze
        `R ← Bin_1^{256 × d(m1+ℓ)}` — the one derivation both sides replay.

        Shaped `(256, d(m1+ℓ))` rather than as a stack of ring elements
        because that is what it is: a `Z`-linear map on coefficient vectors,
        which only becomes ring-shaped once `T` reads it row by row."""
        rows = self.masking.projection
        count = rows * self._chunks * self.scheme.ring.d
        t = absorb_stacks(transcript.observe_label(_LABEL_COMMIT), t_mask, t_sign)
        t, raw = t.sample_scalar(-(-count * _BIN1_BITS // 8))
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        pairs = bits[: count * _BIN1_BITS].reshape(count, _BIN1_BITS).astype(np.int64)
        return t, (pairs[:, 1] - pairs[:, 0]).reshape(rows, -1)

    def _evaluations(
        self, projection: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The `256 + (d − 1)` functions whose constant coefficients vanish:
        `F_i` (eq. 43) then `G_j`.

        `F_i = z_i − b·T(⃗r_i, ⃗s) − y_i` splits across all three pieces —
        the `b·T` term is the only quadratic one in this module, and it sits
        at `(b, s_j)` in the quadratic form because `T` contracts `⃗r_i`'s
        `j`-th chunk against `⃗s`'s. `G_j = T(⃗δ_j, b)` is linear in `b`
        alone. Both are rebuilt identically by the verifier, which is what
        binds `⃗z` to the proof."""
        ring = self.scheme.ring
        d = ring.d
        count = self.masking.projection

        # σ₋₁(⃗r_i) chunk by chunk: `T(⃗a, ⃗b) = Σ_i σ₋₁(a_i)·b_i` (eq. 16),
        # so the public half is what carries the automorphism. One
        # `from_signed_stack` over every chunk at once — it is already
        # "`from_signed` per row", so stacking its results by hand would be
        # re-spelling its own batching.
        rows = ring.from_signed_stack(projection.reshape(-1, d)).reshape(
            count, self._chunks, len(ring.q_moduli), d
        )
        sigma_rows = ring.neg(ring.galois(rows, 2 * d - 1))

        e2 = ring.zeros(count + d - 1, self.evaluation.width, self.evaluation.width)
        e2[np.arange(count)[:, None], self._sign_position, self._witness_positions] = (
            sigma_rows
        )
        e0 = ring.zeros(count + d - 1, 1)
        e0[:count, 0] = _constants(ring, z.reshape(-1))
        return e2, self._e1, e0

    def _linear_block(self) -> np.ndarray:
        """The linear half of both families, which depends on neither `R`
        nor `⃗z` and is therefore built once.

        `F_i` contributes `−σ₋₁(X^{i mod d})` on the mask element holding
        `y_i`; `G_j` contributes `σ₋₁(X^j)` on the sign, unnegated, because
        it reads the sign's own coefficients rather than subtracting them."""
        ring = self.scheme.ring
        d = ring.d
        count = self.masking.projection
        e1 = ring.zeros(count + d - 1, self.evaluation.width)
        index = np.arange(count)
        e1[index, self._mask_positions[index // d]] = ring.neg(
            self._sigma_monomials[index % d]
        )
        e1[count + np.arange(d - 1), self._sign_position] = self._sigma_monomials[1:]
        return e1

    def _relation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`f(b) = b² − 1`, the one function claimed zero as a ring element.

        Paired with the `G_j` above, not usable without them: `b² = 1` alone
        admits any square root of one in `R_q`, and it is integrality that
        cuts those down to `±1`."""
        ring = self.scheme.ring
        width = self.evaluation.width
        r2 = ring.zeros(1, width, width)
        r2[0, self._sign_position, self._sign_position] = ring.one()
        r1 = ring.zeros(1, width)
        r0 = ring.neg(ring.one())[None, None]
        return r2, r1, r0

    def _require_statement(
        self, b: np.ndarray, b_mask: np.ndarray, b_sign: np.ndarray
    ) -> None:
        """The three BDLOP blocks this layer's message is committed under."""
        scheme = self.scheme
        cols = scheme.randomness_cols
        scheme.require_stack("range: b", b, self.ell, cols)
        scheme.require_stack("range: b_mask", b_mask, self.masking.mask_cols, cols)
        scheme.require_stack("range: b_sign", b_sign, 1, cols)

    def _require_witness(
        self, s1: np.ndarray, s2: np.ndarray, message: np.ndarray
    ) -> None:
        """The witness halves plus the message, all signed — the statement
        side of `zorch/lnp/wire.py`."""
        self.evaluation.require_witness("range.prove", s1, s2)
        wire.require_signed(self.scheme.ring, "range.prove: message", message, self.ell)

    def _is_well_formed(self, proof: RangeProof) -> bool:
        """Whether `proof` is structurally usable — every field of it, in one
        place, per `zorch/lnp/wire.py`. `z` routes through the same raising
        gate as every other signed array here rather than restating its
        predicate, which is the rule that module's docstring states."""
        return (
            isinstance(proof, RangeProof)
            and wire.is_stack(self.scheme, proof.t_mask, self.masking.mask_cols)
            and wire.is_stack(self.scheme, proof.t_sign, 1)
            and wire.is_signed(self.scheme.ring, proof.z, self.masking.mask_cols)
            and self.evaluation._is_well_formed(proof.evaluation)
        )


def _constants(ring: HostSplitRing, values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Each value as the constant polynomial holding it, stacked.

    Built through `from_signed` rather than by writing the residue in
    place, because the value is a signed integer over unreduced ℤ — `⃗z`'s
    entries, or the sign `±1` — and the reduction into `Z_q` is exactly
    what that constructor owns."""
    rows = np.zeros((len(values), ring.d), dtype=np.int64)
    rows[:, 0] = values
    return ring.from_signed_stack(rows)
