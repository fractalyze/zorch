# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π^(2) — quadratic relations in the committed message, over R_q.

The third protocol layer of the LNP framework (eprint 2022/284, §4), and
the one the norm statements are actually built on. `opening.py` proves
relations that are *linear* in `(s1, m)`; this proves ones that are
quadratic:

    f(s) = sᵀ·R2·s + r1ᵀ·s + r0 = 0.

Two classes, the paper's Fig. 6 and Fig. 7: `AbdlopQuadratic` proves one
such relation, and `AbdlopQuadraticMany` proves N of them by aggregating
with a Fiat-Shamir challenge before delegating to the first — so N
relations still commit exactly one garbage term.

**Why a quadratic layer buys norms.** The automorphism `σ₋₁ : X ↦ X⁻¹`
puts `⟨a, b⟩` — the integer inner product of two coefficient vectors — in
the constant coefficient of `σ₋₁(a)·b` (§2.3). So `⟨s, s⟩ = ‖s‖²`, the
quantity every norm bound is stated in, is a *quadratic* function of the
witness and its automorphism image. That is the whole reason §4 exists,
and why the challenge space was built σ₋₁-invariant three modules ago.

**The trick that makes it provable.** The masked response is `z = c·s + y`,
so `zᵀR2z` expands to `c²·(sᵀR2s) + c·g1 + g0` — the quadratic term the
verifier wants is buried under two garbage terms in `c`. Because every
`c ∈ C` satisfies `σ(c) = c` (`challenge.py`), `σ` passes through the
challenge and the expansion stays a polynomial in `c` with the *witness*
term isolated at `c²`. The prover therefore commits `g1` (as
`t = bᵀ·s2 + g1`, before seeing `c`) and sends `g0 + bᵀ·y2` in the clear,
which pins both garbage terms and leaves the verifier checking

    zᵀR2z + c·r1ᵀz + c²·r0 − f = v,     f := c·t − bᵀ·z2

— an identity that holds exactly when `f(s) = 0` (eq. 31).

**The lift.** `s` is not the witness as committed; it is the witness and
its automorphism images stacked, `[(σⁱ(s1))ᵢ ; (σⁱ(m))ᵢ]` (eq. 29/30), so
a statement may mention `s1`, `m` and their `σ` images at once. The
masking is lifted the same way, with the message half carrying `−B·y2`
because the verifier reaches `m` only through `z_m = c·t_B − B·z2`.

**σ is pinned to σ₋₁, so the automorphism order is 2.** The paper states
§4 for a general `σ ∈ Aut(R_q)` of order `k`, but soundness needs the
challenge space to be `σ`-invariant, and `challenge.py` builds exactly the
σ₋₁-invariant one (§2.7). Taking `σ` as a parameter here would let a
caller pair a challenge space with an automorphism it does not fix, which
is a silent soundness break rather than an error. A second automorphism
becomes expressible when a second challenge space does — §6.5 and §7 are
where the paper needs one.

⚠ `SIGMA_ORDER` below is §4's `k`, the order of the automorphism. It is
**not** `ChallengeParams.k`, which is the exponent in the operator-norm
gate `²ᵏ√‖σ₋₁(cᵏ)cᵏ‖₁ ≤ η` (§2.7, k = 32 at the paper's point). The paper
reuses the letter for both; conflating them is a parameter bug that no
shape check would catch.

Fiat-Shamir shape, and what is on the wire: the prover absorbs `(w, t, v)`
and answers with `(c, z1, z2, t)`. `w` and `v` are absent for the reason
they are absent in `opening.py` — the verifier recomputes both from the
verification equations and accepts iff the replayed challenge matches, so
hashing them is what checks them. `t` is *not* recomputable: it commits
`g1`, which depends on the secret masking, so it is sent and absorbed.

The masking, rejection budget and norm bounds are `masking.py`'s, shared
with Fig. 4 — see there for the host/device boundary and the randomness
posture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from lattice_frx.sampler import uniform_bytes_needed, uniform_from_bytes
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.lnp import wire
from zorch.lnp.masking import Masking

_LABEL = b"lnp/quad"
_LABEL_MANY = b"lnp/quad/aggregate"

# `µ` is drawn coefficient-wise off the transcript, and the byte sampler
# builds its draws from little-endian u64 chunks — so q must fit one.
_MAX_MODULUS = 1 << 64

# §4's `k`: the order of σ₋₁, which is an involution. See the module
# docstring on why this is pinned rather than taken, and on the collision
# with `ChallengeParams.k`.
SIGMA_ORDER = 2


@dataclass(frozen=True)
class QuadraticProof:
    """The non-interactive Π^(2) wire: the challenge, the two masked
    responses (signed integer coefficient vectors, as in `opening.py`), and
    the commitment `t` to the linear garbage term `g1`.

    `w` and `v` are absent by design — the verifier recomputes them; see
    the module docstring."""

    c: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    t: np.ndarray


class AbdlopQuadratic:
    """Π^(2) prove/verify over an ABDLOP commitment (Fig. 6).

    Built over a `Masking` rather than over an `AbdlopOpening`: this is
    Fig. 4's *sibling*, masking against the same parameter point but
    absorbing its own messages and checking its own equation. Nothing here
    nests an opening."""

    def __init__(self, masking: Masking) -> None:
        self.masking = masking
        self.scheme = masking.scheme
        # n = k·(m1 + ℓ), the width of the lifted witness the statement is
        # written against — derived, so a caller cannot disagree with it.
        self.width = SIGMA_ORDER * (self.scheme.s1_cols + self.scheme.messages)

    def prove(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        b_quad: np.ndarray,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
        rng: np.random.Generator,
        transcript: ByteTranscript,
    ) -> tuple[QuadraticProof, ByteTranscript]:
        """One non-interactive proof that `f(s) = 0`.

        `b_quad` is Fig. 6's `b`, the R_q^{m2} vector the cross-term
        commitment `t` is taken against — distinct from the BDLOP matrix
        `b`, and from Fig. 8's `B_g`. `s1`/`s2` are signed integer
        `(m_i, d)` arrays; `message` is the ring stack `m` that `commit`
        was called with. The commitment is absent for the reason it is
        absent in `opening.py` — the transcript arrived bound to it."""
        ring = self.scheme.ring
        masking = self.masking
        self._require_statement(b_quad, r2, r1, r0)
        masking.require_witness("quadratic.prove", s1, s2)
        s1_ring = ring.from_signed_stack(s1)
        s2_ring = ring.from_signed_stack(s2)
        # Witness-only, so a rejected attempt would recompute them
        # unchanged; the lift and its matvec are the costly ones.
        s = self._lift(s1_ring, message)
        r2s = ring.matvec(r2, s)
        b_s2 = _dot(ring, b_quad, s2_ring)

        for _ in range(masking.attempts):
            y1, y2 = masking.draw(rng)
            y1_ring = ring.from_signed_stack(y1)
            y2_ring = ring.from_signed_stack(y2)
            w = ring.add(ring.matvec(a1, y1_ring), ring.matvec(a2, y2_ring))
            # eq. 29: the message half masks `m` through `−B·y2`, because
            # that is the only way the verifier reaches `m`.
            y = self._lift(y1_ring, ring.neg(ring.matvec(b, y2_ring)))
            # g1 = sᵀR2y + yᵀR2s + r1ᵀy, g0 = yᵀR2y (eq. 31). `R2·s` is
            # loop-invariant; `R2·y` is not.
            r2y = ring.matvec(r2, y)
            g1 = ring.add(
                ring.add(_dot(ring, s, r2y), _dot(ring, y, r2s)),
                _dot(ring, r1, y),
            )
            t = ring.add(b_s2, g1)
            v = ring.add(_dot(ring, y, r2y), _dot(ring, b_quad, y2_ring))

            advanced, c = masking.challenge_from(transcript, _LABEL, w, t, v)
            cs1, cs2 = masking.respond(c, s1, s2)
            z1 = cs1 + y1
            z2 = cs2 + y2
            if masking.accepts(rng, z1, cs1, z2, cs2):
                return QuadraticProof(c=c, z1=z1, z2=z2, t=t), advanced
        raise masking.exhausted("quadratic.prove")

    def verify(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        b_quad: np.ndarray,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        t_a: np.ndarray,
        t_b: np.ndarray,
        proof: QuadraticProof,
        transcript: ByteTranscript,
    ) -> tuple[bool, ByteTranscript]:
        """Fig. 6's checks in their non-interactive shape: both norm bounds,
        then the recomputed `(w, v)` must replay to the proof's challenge —
        which folds the commitment equation and the quadratic identity into
        the hash."""
        ring = self.scheme.ring
        # The statement is the caller's and raises; the proof is the
        # prover's and is a verdict. See `zorch/lnp/wire.py`.
        self._require_statement(b_quad, r2, r1, r0)
        if not self._is_well_formed(proof):
            return False, transcript
        if not self.masking.within_bounds(proof.z1, proof.z2):
            return False, transcript

        c_elem = ring.from_signed(proof.c)
        c_stack = c_elem[None]
        z1_ring = ring.from_signed_stack(proof.z1)
        z2_ring = ring.from_signed_stack(proof.z2)
        w = ring.sub(
            ring.add(ring.matvec(a1, z1_ring), ring.matvec(a2, z2_ring)),
            ring.scale(c_elem, t_a),
        )
        # eq. 30's `z`: the response half, and the message half the verifier
        # can form — `z_m = c·t_B − B·z2` (`opening.py` computes the same
        # quantity for the linear check).
        z_m = ring.sub(ring.scale(c_elem, t_b), ring.matvec(b, z2_ring))
        z = self._lift(z1_ring, z_m)
        # f := c·t − bᵀ·z2, then v := zᵀR2z + c·r1ᵀz + c²·r0 − f.
        f = ring.sub(ring.scale(c_elem, proof.t), _dot(ring, b_quad, z2_ring))
        c_sq = ring.scale(c_elem, c_stack)[0]
        v = ring.sub(
            ring.add(
                ring.add(
                    _dot(ring, z, ring.matvec(r2, z)),
                    ring.scale(c_elem, _dot(ring, r1, z)),
                ),
                ring.scale(c_sq, r0),
            ),
            f,
        )
        advanced, c = self.masking.challenge_from(transcript, _LABEL, w, proof.t, v)
        return bool(np.array_equal(c, proof.c)), advanced

    def _lift(self, s1_part: np.ndarray, message_part: np.ndarray) -> np.ndarray:
        """`[(σⁱ(s1_part))ᵢ ; (σⁱ(message_part))ᵢ]` for `i ∈ [k]` (eq. 29).

        The two halves are lifted separately and then concatenated, which is
        the paper's order — `s1`'s `k` images first, then the message's —
        and the order the statement's `R2`/`r1` are indexed against."""
        ring = self.scheme.ring
        return np.concatenate([_orbit(ring, s1_part), _orbit(ring, message_part)])

    def _is_well_formed(self, proof: QuadraticProof) -> bool:
        """Whether `proof` is structurally usable — every field of it, in
        one place, per `zorch/lnp/wire.py`. `t` is a ring element and gates
        as one; the response triple is the shared half."""
        return (
            isinstance(proof, QuadraticProof)
            and self.masking.is_response(proof.c, proof.z1, proof.z2)
            and wire.is_stack(self.scheme, proof.t, 1)
        )

    def _require_statement(
        self,
        b_quad: np.ndarray,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
    ) -> None:
        """The quadratic function is three aligned pieces over the lifted
        width, plus the vector its garbage commitment is taken against; a
        mismatch would otherwise surface deep inside a ring op."""
        scheme = self.scheme
        n = self.width
        for name, arr, lead in (
            ("b_quad", b_quad, (scheme.randomness_cols,)),
            ("r2", r2, (n, n)),
            ("r1", r1, (n,)),
            ("r0", r0, (1,)),
        ):
            scheme.require_stack(f"quadratic: {name}", arr, *lead)


class AbdlopQuadraticMany:
    """Π_many^(2) prove/verify — N quadratic relations at once (Fig. 7).

    Proving the N relations separately would commit N garbage polynomials.
    Instead the verifier sends `µ ∈ R_q^N` and the prover proves the single
    relation `f = Σ_j µ_j·f_j` through Fig. 6, so exactly one garbage term
    is committed however many relations there are. The cost is an additive
    `q1^{-d/2}` in the soundness error (Lemma 4.3), `q1` being the smallest
    prime factor of `q`: if some `f_j(s) ≠ 0`, a random `µ_j` kills it only
    with that probability, because `X^d + 1` splits into two irreducible
    factors modulo each `q_i`.

    `µ` is drawn from **R_q**, not `Z_q` — a whole ring element per
    relation. `Π_eval`'s γ are `Z_q` scalars because they aggregate
    constant-coefficient statements; these aggregate ring-valued ones, and
    a scalar µ would leave the soundness argument without the degree-`d/2`
    factor it rests on.

    Nothing is added to the wire: `µ` is Fiat-Shamir output, so the proof
    is the inner Fig. 6 proof and the verifier re-derives `µ` itself."""

    def __init__(self, quadratic: AbdlopQuadratic) -> None:
        self.quadratic = quadratic
        self.scheme = quadratic.scheme
        ring = self.scheme.ring
        modulus = math.prod(ring.q_moduli)
        if modulus >= _MAX_MODULUS:
            raise ValueError(
                f"quadratic: µ is drawn coefficient-wise from u64 chunks, so "
                f"q must be below 2^64; this ring's q has "
                f"{modulus.bit_length()} bits. An RNS chain this wide needs a "
                f"wider sampler first."
            )
        self.modulus = modulus

    def prove(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        b_quad: np.ndarray,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
        rng: np.random.Generator,
        transcript: ByteTranscript,
    ) -> tuple[QuadraticProof, ByteTranscript]:
        """One proof that every `f_j(s) = 0`.

        `r2`/`r1`/`r0` carry a leading relation axis over Fig. 6's shapes."""
        self._require_functions(r2, r1, r0)
        advanced, mu = self._mu(transcript, r2.shape[0])
        one_r2, one_r1, one_r0 = self._aggregate(mu, r2, r1, r0)
        return self.quadratic.prove(
            a1,
            a2,
            b,
            b_quad,
            one_r2,
            one_r1,
            one_r0,
            s1,
            s2,
            message,
            rng,
            advanced,
        )

    def verify(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        b_quad: np.ndarray,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        t_a: np.ndarray,
        t_b: np.ndarray,
        proof: QuadraticProof,
        transcript: ByteTranscript,
    ) -> tuple[bool, ByteTranscript]:
        """Re-derive `µ`, aggregate the same way, and defer to Fig. 6."""
        self._require_functions(r2, r1, r0)
        advanced, mu = self._mu(transcript, r2.shape[0])
        one_r2, one_r1, one_r0 = self._aggregate(mu, r2, r1, r0)
        return self.quadratic.verify(
            a1,
            a2,
            b,
            b_quad,
            one_r2,
            one_r1,
            one_r0,
            t_a,
            t_b,
            proof,
            advanced,
        )

    def _mu(
        self, transcript: ByteTranscript, relations: int
    ) -> tuple[ByteTranscript, np.ndarray]:
        """Squeeze `µ ∈ R_q^N` — `d` coefficients per relation, the one
        derivation both sides replay.

        No absorb precedes it: Fig. 7 opens with the verifier's message, and
        the transcript arrived already bound to the statement."""
        ring = self.scheme.ring
        count = relations * ring.d
        t = transcript.observe_label(_LABEL_MANY)
        t, raw = t.sample_scalar(uniform_bytes_needed(self.modulus, count))
        draws = uniform_from_bytes(raw, self.modulus, count)
        return t, ring.from_signed_stack(draws.reshape(relations, ring.d))

    def _aggregate(
        self,
        mu: np.ndarray,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(Σ_j µ_j·R2_j, Σ_j µ_j·r1_j, Σ_j µ_j·r0_j)` — one aggregated
        quadratic function.

        Each piece is a contraction of the relation axis against `µ`, which
        is `matvec` with the relation axis moved last: the module
        convention's contraction *is* the ring-weighted sum, so this needs
        no ring op of its own. `R2`'s two matrix axes are flattened for the
        contraction and restored after, since `matvec` contracts one axis."""
        ring = self.scheme.ring
        n = self.quadratic.width
        tail = r2.shape[-2:]
        flat = r2.reshape(r2.shape[0], n * n, *tail)
        return (
            ring.matvec(np.moveaxis(flat, 0, 1), mu).reshape(n, n, *tail),
            ring.matvec(np.moveaxis(r1, 0, 1), mu),
            ring.matvec(np.moveaxis(r0, 0, 1), mu),
        )

    def _require_functions(
        self, r2: np.ndarray, r1: np.ndarray, r0: np.ndarray
    ) -> None:
        """The N functions are three aligned pieces; a mismatch would
        otherwise surface as a `µ` of the wrong width."""
        relations = getattr(r2, "shape", (0,))[0]
        if relations < 1:
            raise ValueError("quadratic: need at least one quadratic function")
        n = self.quadratic.width
        for name, arr, lead in (
            ("r2", r2, (relations, n, n)),
            ("r1", r1, (relations, n)),
            ("r0", r0, (relations, 1)),
        ):
            self.scheme.require_stack(f"quadratic: {name}", arr, *lead)


def _orbit(ring: HostSplitRing, stack: np.ndarray) -> np.ndarray:
    """`(σⁱ(stack))_{i ∈ [k]}` stacked, `σ = σ₋₁`.

    `σ₋₁` is an involution, so at `k = 2` the orbit is `[x, σ₋₁(x)]`; the
    loop is written against `SIGMA_ORDER` rather than spelled as a pair so
    that the shape stays right if a later challenge space admits an
    automorphism of higher order. `galois` carries the batch axis, so each
    image is one call over the whole stack rather than one per element."""
    exponent = 2 * ring.d - 1  # σ₋₁ : X ↦ X^{-1} = X^{2d-1}
    images = [stack]
    for _ in range(SIGMA_ORDER - 1):
        images.append(ring.galois(images[-1], exponent))
    return np.concatenate(images)


def _dot(ring: HostSplitRing, row: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """`⟨row, vec⟩` over R_q as a one-element stack — `matvec`'s rank-one
    case, which is how a ring inner product is spelled in the module
    convention. One-element rather than bare so it composes with `add`,
    `sub` and `scale` without a rank change at every site."""
    return ring.matvec(row[None, :], vec)
