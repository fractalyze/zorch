# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π_eval — statements over Z_q, as constant coefficients over R_q.

The LNP framework's two evaluation protocols (eprint 2022/284): `Fig. 5`,
which aggregates *linear* functions of the committed `(s1, m)`, and
`Fig. 8`, which aggregates *quadratic* ones over the σ-lift and carries a
batch of ring-valued relations alongside. They share this file because
they share the step that defines them — `GarbageMasking` below — and
differ only in what is aggregated and which protocol proves the aggregate
well-formed: `opening.py`'s Π_many for the first, `quadratic.py`'s
Π_many^(2) for the second.

The bulk of what follows describes Fig. 5; `AbdlopQuadraticEval` at the
bottom describes where Fig. 8 departs from it.

The second protocol layer of the LNP framework (Fig. 5).
Π_many (`opening.py`) proves that a linear function of the committed
`(s1, m)` is zero *as a ring element*; this layer proves the strictly
weaker — and, for inner products, the interesting — statement that its
**constant coefficient** is zero. That is what turns a statement over
`R_q` into one over `Z_q`: §2.3's identity puts the inner product of two
integer vectors in the constant coefficient of a polynomial product, so
"this linear function over `Z_q` vanishes" is exactly "this constant
coefficient vanishes".

The obstacle is that revealing a linear function's constant coefficient
must not reveal the rest of it, so the protocol masks:

1. The prover draws garbage `g = (g_1..g_λ)`, uniform over `R_q` except
   for a zero constant coefficient, and commits it *alongside* the
   message under its own public matrix `B_g`: `t_g = B_g·s2 + g`. The
   BDLOP half's message becomes `m‖g`.
2. The verifier answers with `γ ∈ Z_q^{λ×M}` — λ independent random
   aggregations of the M statements.
3. The prover sends `h_j = g_j + Σ_u γ_{j,u}·F_u(s1, m)` in the clear.
   `g_j` hides every coefficient of the aggregate except the constant
   one, which `g_j` leaves alone.
4. The verifier checks `h̃_j = 0` for every j, and that the `h_j` really
   were computed from the committed values — which is a *linear* relation
   over `(s1, m‖g)`, so Π_many proves it (eq. 28):
       `f_j(s1, m‖g) := g_j + Σ_u γ_{j,u}·F_u(s1, m) − h_j = 0`.

Soundness is `q1^{-λ}` where `q1` is the **smallest prime factor of q**:
`g` is committed before γ arrives, so if some `F̃_u ≠ 0` then each `h̃_j`
is a fresh random aggregation and vanishes with probability at most
`1/q1`. The smallest factor, not `q` itself, is what bounds it — a
nonzero element of a composite `Z_q` can still be killed by a zero
divisor.

**Commit-and-prove, not zero-knowledge over a reusable commitment.** §3.2
is explicit that appending `g` means `(t_A, t_B)` cannot be reused: each
run leaks more about `s2`. The seam reflects that — the caller passes the
commitment in per proof and must not run two proofs against one.

Statement shape. The M linear functions arrive the way Π_many's do, as
matrix rows plus a target: `F_u(s1, m) = Fs1_u·s1 + Fm_u·m − target_u`.
The claim proved is `F̃_u(s1, m) = 0` for every u — not `F_u = 0`, which
is what the layer below is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from lattice_frx.sampler import uniform_bytes_needed, uniform_from_bytes

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp import wire
from zorch.lnp.opening import AbdlopOpening, OpeningProof
from zorch.lnp.quadratic import (
    SIGMA_ORDER,
    AbdlopQuadraticMany,
    Publics,
    QuadraticProof,
    evaluate,
    lift,
    lift_positions,
    lift_slots,
)
from zorch.lnp.transcript import absorb_stacks

# `γ` is drawn as a Z_q scalar off the transcript, and the byte sampler
# builds its draws from little-endian u64 chunks — so q must fit one.
_MAX_MODULUS = 1 << 64


class GarbageMasking:
    """The ENS20 garbage-and-aggregate step, and the transcript labels it
    hashes under — what Fig. 5 and Fig. 8 share.

    Both protocols prove that functions of the committed witness have a
    *vanishing constant coefficient*, and both do it the same way: commit
    λ garbage polynomials that are themselves constant-coefficient-free,
    take `Γ ∈ Z_q^{λ×M}` from the verifier, and reveal the λ aggregates in
    the clear for the `h̃_j = 0` check. What differs is only *what* is
    aggregated — linear functions of `(s1, m)` in Fig. 5, quadratic ones
    over the σ-lift in Fig. 8 — and that half stays with the caller.

    Extracted for the reason `masking.py` was: the two protocols do not
    merely resemble each other here, they must agree, and a second spelling
    of one derivation is a fork no single-protocol suite can see.

    `domain` separates the two transcripts, so it is a constructor argument
    rather than a module constant: deriving `Γ` from the same absorbed
    bytes under the same label in two protocols is what would let a proof
    of one be replayed against the other.
    """

    def __init__(self, scheme: AbdlopCommitment, lam: int, domain: bytes) -> None:
        if lam < 1:
            raise ValueError(f"eval: lam must be positive, got {lam!r}")
        ell = scheme.messages - lam
        if ell < 0:
            raise ValueError(
                f"eval: the scheme's BDLOP half carries {scheme.messages} "
                f"messages, too few for lam={lam} garbage terms on top of a "
                f"message vector — build it over the extended scheme"
            )
        modulus = math.prod(scheme.ring.q_moduli)
        if modulus >= _MAX_MODULUS:
            raise ValueError(
                f"eval: γ is a Z_q scalar drawn from u64 chunks, so q must be "
                f"below 2^64; this ring's q has {modulus.bit_length()} bits. "
                f"An RNS chain this wide needs a wider γ sampler first."
            )
        self.scheme = scheme
        self.lam = lam
        self.ell = ell
        self.modulus = modulus
        self._commit_label = domain + b"/garbage"
        self._mask_label = domain + b"/masked"

    def commit(
        self, bg: np.ndarray, s2_ring: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw the λ garbage terms and commit them: `(g, t_g = B_g·s2 + g)`.

        The line that actually *defines* the extension, and therefore the
        one that had least business being spelled once per protocol. `bg`
        is gated here for the same reason — it was the only public matrix
        in this package with no shape gate, so a wrong row count surfaced
        from a ring `matvec` instead of naming itself."""
        ring = self.scheme.ring
        self.scheme.require_stack("eval: bg", bg, self.lam, self.scheme.randomness_cols)
        g = self.sample(rng)
        return g, ring.add(ring.matvec(bg, s2_ring), g)

    def blocks(self, b: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """The BDLOP matrix the inner protocol opens `m‖g` against."""
        return np.concatenate([b, bg])

    def rows(self, blocks: np.ndarray) -> np.ndarray:
        """`B_g` — this layer's own rows of an *already assembled* matrix.

        `blocks` above builds the layout; this reads it back. Both live here
        because this is the class that owns `ell` and `lam`, and "the
        garbage is the tail" asserted in two classes in two spellings is the
        drift `GarbageMasking` was extracted to prevent — no round-trip can
        see them disagree, since both sides would carve the same wrong way.

        The caller that assembles the whole matrix up front (`Publics`, once
        a range leg is in the picture) takes this; the caller that still
        concatenates its own takes `blocks`."""
        return blocks[self.ell :]

    def commitment(self, t_b: np.ndarray, t_g: np.ndarray) -> np.ndarray:
        """The commitment to `m‖g`, in the order `blocks` is stacked in.

        Named beside `blocks` because the two are one ordering contract, and
        a round-trip cannot see them disagree — both sides build them the
        same way."""
        return np.concatenate([t_b, t_g])

    def aggregate(self, gamma: np.ndarray, *blocks: np.ndarray) -> tuple:
        """`Σ_u γ_{j,u}·block_u` for each row `j` of Γ, per block.

        Γ's own contraction, so it belongs on the seam that owns Γ. Both
        protocols aggregate — Fig. 5 over `(Fs1, Fm, target)`, Fig. 8 over
        `(e2, e1, e0)` — and each block keeps whatever it holds past the
        contracted axis, so one call serves a stack of elements and a stack
        of whole matrices alike.

        Returns block-major stacks, the shape both callers ultimately index
        against; the per-row form one of them used to return had to be
        transposed back before it could be used."""
        ring = self.scheme.ring
        return tuple(
            np.stack([ring.combine(row, block) for row in gamma]) for block in blocks
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """`g ← {x ∈ R_q : x̃ = 0}^λ` — the ring's uniform stack with the
        constant coefficient forced to zero. Private coins: this is masking,
        like the Gaussian `y` of the layer below, so it comes off the
        caller's generator and never off the transcript.

        The zeroing is the protocol's own — `uniform_stack` is the module
        convention's uniform constructor, and `x̃ = 0` is the condition on
        the garbage, not a ring-level shape."""
        g = self.scheme.ring.uniform_stack(rng, self.lam)
        g[..., 0] = 0
        return g

    def gamma(
        self, transcript: ByteTranscript, t_g: np.ndarray, relations: int
    ) -> tuple[ByteTranscript, np.ndarray]:
        """Absorb the garbage commitment and squeeze `Γ ∈ Z_q^{λ×M}` — the
        one derivation both sides replay."""
        count = self.lam * relations
        t = absorb_stacks(transcript.observe_label(self._commit_label), t_g)
        t, raw = t.sample_scalar(uniform_bytes_needed(self.modulus, count))
        draws = uniform_from_bytes(raw, self.modulus, count)
        return t, draws.reshape(self.lam, relations)

    def observe(self, transcript: ByteTranscript, h: np.ndarray) -> ByteTranscript:
        """Bind the revealed aggregates, which every later challenge in the
        proof is drawn after."""
        return absorb_stacks(transcript.observe_label(self._mask_label), h)

    def vanishes(self, h: np.ndarray) -> bool:
        """The `h̃_j = 0` check, over every limb at once.

        Zero mod each `q_i` is zero mod `q` by CRT, so the ring's own
        constant-coefficient reading answers it directly."""
        return not self.scheme.ring.constant_coeff(h).any()


@dataclass(frozen=True)
class EvalProof:
    """The Π_eval wire: the garbage commitment, the masked aggregates, and
    the Π_many proof underneath. `γ` is absent — it is Fiat-Shamir output,
    and the verifier re-derives it from `t_g`."""

    t_g: np.ndarray
    h: np.ndarray
    opening: OpeningProof


class AbdlopEval:
    """Π_eval prove/verify over an `AbdlopOpening` (Fig. 5).

    The opening it wraps must already be built over the **extended**
    scheme: its BDLOP half carries `ℓ + λ` messages, because `m‖g` is what
    the inner Π_many opens. `ell` is derived from that rather than taken,
    so the two counts cannot disagree.

    `lam` (λ) is the soundness parameter — the proof costs λ garbage
    commitments and λ masked aggregates, and buys soundness `q1^{-λ}`.
    Choosing it against the target security level is the consumer's
    parameter work, like every other number this package's seams take."""

    def __init__(self, opening: AbdlopOpening, lam: int) -> None:
        self.garbage = GarbageMasking(opening.scheme, lam, b"lnp/eval")
        self.opening = opening
        # Named here for the reason its Fig. 8 sibling names it: a consumer
        # should not have to know which layer down the scheme lives.
        self.scheme = opening.scheme
        self.lam = lam
        self.ell = self.garbage.ell
        # eq. 28's `e_j` selector, stacked over all j: the λ×λ identity over
        # the ring. Depends only on `lam`, so it is built once rather than on
        # every prove and every verify.
        self._selector = self.scheme.ring.zeros(lam, lam)
        diagonal = np.arange(lam)
        self._selector[diagonal, diagonal] = self.scheme.ring.one()

    def prove(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        bg: np.ndarray,
        fs1: np.ndarray,
        fm: np.ndarray,
        target: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
        rng: np.random.Generator,
        transcript: ByteTranscript,
    ) -> tuple[EvalProof, ByteTranscript]:
        """One non-interactive proof that every `F̃_u(s1, m)` is zero.

        `s1`/`s2` are signed integer `(m_i, d)` arrays as in `opening.py`;
        `message` is the ring stack `m` that `commit` was called with. The
        commitment is absent for the same reason it is absent there — the
        transcript arrived bound to it."""
        ring = self.scheme.ring
        self._require_functions(fs1, fm, target)
        s1_ring = ring.from_signed_stack(s1)
        s2_ring = ring.from_signed_stack(s2)

        # (1) garbage with a zero constant coefficient, committed under B_g.
        g, t_g = self.garbage.commit(bg, s2_ring, rng)

        # (2) γ, once t_g is bound.
        t, gamma = self.garbage.gamma(transcript, t_g, fs1.shape[0])

        # (3) the masked aggregates. Aggregating the M functions *before*
        #     applying them, rather than evaluating all M and summing, is
        #     the same value by linearity —
        #       h_j = g_j + Σ_u γ_{j,u}·(Fs1_u·s1 + Fm_u·m − target_u)
        #           = g_j + (Σ_u γ_{j,u}Fs1_u)·s1 + (Σ_u γ_{j,u}Fm_u)·m − …
        #     — and costs λ matvecs instead of M, over a ring whose mul is
        #     a deliberate O(d²) host oracle. The blocks are the eq.-28
        #     relation's own, so they are computed once and used twice.
        aggregated = self.garbage.aggregate(gamma, fs1, fm, target)
        weighted_s1, weighted_m, weighted_target = aggregated
        h = ring.sub(
            ring.add(
                g,
                ring.add(
                    ring.matvec(weighted_s1, s1_ring),
                    ring.matvec(weighted_m, message),
                ),
            ),
            weighted_target,
        )

        # (4) the same aggregation as a linear relation over (s1, m‖g),
        #     proved by the layer below.
        t = self.garbage.observe(t, h)
        # `u` is the verifier's side of eq. 28; the prover only supplies
        # the matrices its opening call takes.
        r1, rm, _ = self._relation(aggregated, h)
        opening, t = self.opening.prove(
            a1, a2, self.garbage.blocks(b, bg), r1, rm, s1, s2, rng, t
        )
        return EvalProof(t_g=t_g, h=h, opening=opening), t

    def verify(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        bg: np.ndarray,
        fs1: np.ndarray,
        fm: np.ndarray,
        target: np.ndarray,
        t_a: np.ndarray,
        t_b: np.ndarray,
        proof: EvalProof,
        transcript: ByteTranscript,
    ) -> tuple[bool, ByteTranscript]:
        """Fig. 5's two checks: every `h_j` has a zero constant coefficient,
        and the Π_many proof of the aggregation relation verifies."""
        # The statement is the caller's and raises; the proof is the
        # prover's and is a verdict. See `zorch/lnp/wire.py`.
        self._require_functions(fs1, fm, target)
        if not self._is_well_formed(proof):
            return False, transcript

        t, gamma = self.garbage.gamma(transcript, proof.t_g, fs1.shape[0])
        t = self.garbage.observe(t, proof.h)
        if not self.garbage.vanishes(proof.h):
            return False, t
        r1, rm, u = self._relation(
            self.garbage.aggregate(gamma, fs1, fm, target), proof.h
        )
        return self.opening.verify(
            a1,
            a2,
            self.garbage.blocks(b, bg),
            r1,
            rm,
            t_a,
            self.garbage.commitment(t_b, proof.t_g),
            u,
            proof.opening,
            t,
        )

    def _is_well_formed(self, proof: EvalProof) -> bool:
        """Whether `proof` is structurally usable — every field of it.

        `opening` is a field like the other two, and the one a per-field
        habit forgets: it is composite, so nothing about it looks like a
        gate, and an `EvalProof` carrying `None` there reached an
        `AttributeError` instead of a verdict. The layer below owns what
        its own wire means, so this defers rather than re-deriving it."""
        scheme = self.scheme
        return (
            isinstance(proof, EvalProof)
            and wire.is_stack(scheme, proof.t_g, self.lam)
            and wire.is_stack(scheme, proof.h, self.lam)
            and self.opening._is_well_formed(proof.opening)
        )

    def _relation(
        self, aggregated: tuple[np.ndarray, np.ndarray, np.ndarray], h: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Eq. 28 in Π_many's `(R1, Rm, u)` shape over the message `m‖g`.

        Row j of `f_j(s1, m‖g) = g_j + Σ_u γ_{j,u}·F_u(s1, m) − h_j = 0`
        rearranges to `R1_j·s1 + Rm_j·(m‖g) = u_j` with

            R1_j  = Σ_u γ_{j,u}·Fs1_u
            Rm_j  = [ Σ_u γ_{j,u}·Fm_u | e_j ]     (`e_j` selects `g_j`)
            u_j   = h_j + Σ_u γ_{j,u}·target_u

        — the `e_j` block being why the inner opening has to be built over
        the extended scheme. Stacked over all j, those blocks are just the
        λ×λ identity over the ring."""
        ring = self.scheme.ring
        weighted_s1, weighted_m, weighted_target = aggregated
        return (
            weighted_s1,
            np.concatenate([weighted_m, self._selector], axis=1),
            ring.add(h, weighted_target),
        )

    def _require_functions(
        self, fs1: np.ndarray, fm: np.ndarray, target: np.ndarray
    ) -> None:
        """The M linear functions are three aligned pieces; a mismatch here
        would otherwise surface as a γ of the wrong width."""
        scheme = self.scheme
        relations = wire.leading(fs1)
        if relations < 1:
            raise ValueError("eval: need at least one linear function")
        # The same gate as `target`'s, not a leading-axes-only variant of
        # it: checking `shape[:2]` left the trailing `(limbs, d)` to be
        # rejected later by whichever ring op reached it, in the ring's
        # vocabulary rather than this layer's.
        for name, arr, lead in (
            ("fs1", fs1, (relations, scheme.s1_cols)),
            ("fm", fm, (relations, self.ell)),
            ("target", target, (relations,)),
        ):
            scheme.require_stack(f"eval: {name}", arr, *lead)


@dataclass(frozen=True)
class QuadraticEvalProof:
    """The Π_eval^(2) wire: the garbage commitment, the revealed
    aggregates, and the Π_many^(2) proof underneath. `Γ` is absent for the
    reason `γ` is absent above — it is Fiat-Shamir output, and the verifier
    re-derives it from `t_g`."""

    t_g: np.ndarray
    h: np.ndarray
    quadratic: QuadraticProof


class AbdlopQuadraticEval:
    """Π_eval^(2) prove/verify over an `AbdlopQuadraticMany` (Fig. 8).

    `AbdlopEval` above proves that *linear* functions of `(s1, m)` have a
    vanishing constant coefficient. This proves the same of *quadratic*
    functions of the σ-lift — which is the form every norm statement takes,
    since `⟨s, s⟩ = ‖s‖²` lives in the constant coefficient of
    `σ₋₁(s)ᵀ·s` (§2.3) and is quadratic in the lift, not linear in `m`.

    Two families arrive, both written against `self.width`, the lift of
    `(s1, m)`:

    - `(r2, r1, r0)` — `N` relations claimed zero *as ring elements*.
      They are carried through untouched (eq. 39); `AbdlopQuadraticMany`
      is what proves them, and `N = 0` is legal.
    - `(e2, e1, e0)` — `M` evaluations claimed to have a zero *constant
      coefficient*, the paper's `F_j`. These are the reason to be here, so
      `M ≥ 1`.

    The two are proved in one shot, against one commitment, because a
    consumer that ran Fig. 7 and Fig. 5 side by side would hold two
    parameter points for one witness and could not see them drift.

    **Why this is not `AbdlopEval` with a quadratic backend.** The λ garbage
    terms are appended to the *message*, so the inner protocol opens
    `m‖g` — and `lift` orbits the message stack as a whole, which puts the
    garbage in each automorphism copy rather than after the message's
    copies. eq. 38 is written against exactly that layout and reads only
    the first copy's `g`, so `_embed` is the map from the caller's width to
    the inner one and is where the layout is owned.

    Soundness is `2/|C| + q1^{-d/2} + q1^{-λ}` (Thm 4.5), `q1` the smallest
    prime factor of `q`: the challenge term is Fig. 6's, the `q1^{-d/2}` is
    Fig. 7's `µ`-aggregation, and `q1^{-λ}` is this layer's Γ. The middle
    term is what a reader of Fig. 5 alone would not expect.

    **Commit-and-prove, not zero-knowledge over a reusable commitment** —
    §3.2, for the same reason as `AbdlopEval`: appending `g` means
    `(t_A, t_B)` leaks more about `s2` on every run. The caller passes the
    commitment in per proof and must not run two proofs against one.
    """

    def __init__(
        self, many: AbdlopQuadraticMany, lam: int, s1_take: int | None = None
    ) -> None:
        scheme = many.scheme
        self.garbage = GarbageMasking(scheme, lam, b"lnp/eval/quad")
        self.many = many
        # Named here for the reason `_is_well_formed` is: the layer above
        # proves through this one and should not reach past it. Without it a
        # consumer spells `eval.many.scheme`, and the chain becomes
        # load-bearing from outside.
        self.scheme = scheme
        self.lam = lam
        self.ell = self.garbage.ell
        # How much of the Ajtai half the caller's statement is about. It is
        # all of it for every consumer up to Fig. 9, and a prefix for Fig.
        # 10, which appends the binary-decomposition vector `x` to the half
        # and writes its functions against `s1` alone.
        if s1_take is None:
            s1_take = scheme.s1_cols
        elif not 0 <= s1_take <= scheme.s1_cols:
            raise ValueError(
                f"eval: the statement cannot be written against {s1_take} of "
                f"the scheme's {scheme.s1_cols} Ajtai columns"
            )
        self.s1_take = s1_take
        # The width the caller's two families are written against: the lift
        # of `(s1, m)` alone. The inner protocol's own width is wider — it
        # lifts `m‖g` — and `_embed` is the map between them.
        self.width = SIGMA_ORDER * (s1_take + self.ell)

        # Both halves carve: `s1_take` of the Ajtai columns, `ell` of the
        # `ell + lam` each message copy carries. `lift_positions` owns the
        # rule; this is the only place that needs to know both numbers.
        self._positions = lift_positions(
            s1_take, scheme.s1_cols, self.ell, self.ell + lam
        )
        # `x^{(g)}_{2,1,i}` of eq. 38 — the garbage of the *first*
        # automorphism copy, which is the only copy the equation reads.
        # Off `lift_slots` rather than `SIGMA_ORDER * s1_cols` spelled here:
        # that helper owns where each copy of each half starts, and this was
        # one of the places that had derived it.
        self._garbage_slots = lift_slots(scheme.s1_cols, self.ell + lam).message[
            self.ell :
        ]

    def prove(
        self,
        publics: Publics,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        e2: np.ndarray,
        e1: np.ndarray,
        e0: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
        rng: np.random.Generator,
        transcript: ByteTranscript,
    ) -> tuple[QuadraticEvalProof, ByteTranscript]:
        """One non-interactive proof that every `f_j(s)` is zero and every
        `F̃_j(s)` is zero.

        `s1`/`s2` are signed integer `(m_i, d)` arrays as in `opening.py`;
        `message` is the ring stack `m` that `commit` was called with —
        without the garbage, which this layer appends itself. `publics
        .blocks` is the whole BDLOP matrix, garbage rows included, so it
        goes down to Fig. 7 untouched."""
        ring = self.many.scheme.ring
        publics.require(self.scheme)
        self._require_functions(r2, r1, r0, e2, e1, e0)
        s2_ring = ring.from_signed_stack(s2)

        # (1) λ garbage terms with zero constant coefficient, committed
        #     under B_g beside the message.
        g, t_g = self.garbage.commit(self.garbage.rows(publics.blocks), s2_ring, rng)

        # (2) Γ, once t_g is bound.
        t, gamma = self.garbage.gamma(transcript, t_g, _count(e2))

        # (3) h_i = g_i + Σ_j γ_{i,j}·F_j(s)  (eq. 37). Aggregating the M
        #     functions *before* evaluating them is the same value by
        #     linearity and costs λ evaluations rather than λ·M, over a ring
        #     whose mul is a deliberate O(d²) host oracle. The aggregates
        #     are the eq.-38 relations' own, so they are built once.
        # The caller's aggregates are written against its own lift, which
        # covers `s1_take` of the Ajtai half — the carve applies to the
        # witness here exactly as it does to `_positions`, or the two are
        # about different widths.
        s = lift(ring, ring.from_signed_stack(s1)[: self.s1_take], message)
        aggregates = self._aggregates(gamma, e2, e1, e0)
        # Named apart from this method's `a1`/`a2` — those are the Ajtai
        # matrices, and shadowing them here sends the aggregates down to
        # Fig. 7 in their place.
        agg2, agg1, agg0 = aggregates
        h = ring.add(
            g,
            np.concatenate(
                [evaluate(ring, agg2[i], agg1[i], agg0[i], s) for i in range(self.lam)]
            ),
        )

        # (4) the N carried relations and the λ new ones, in one Fig. 7 run
        #     over the message `m‖g`.
        t = self.garbage.observe(t, h)
        f2, f1, f0 = self._relations(aggregates, r2, r1, r0, h)
        proof, t = self.many.prove(
            publics,
            f2,
            f1,
            f0,
            s1,
            s2,
            np.concatenate([message, g]),
            rng,
            t,
        )
        return QuadraticEvalProof(t_g=t_g, h=h, quadratic=proof), t

    def verify(
        self,
        publics: Publics,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        e2: np.ndarray,
        e1: np.ndarray,
        e0: np.ndarray,
        t_a: np.ndarray,
        t_b: np.ndarray,
        proof: QuadraticEvalProof,
        transcript: ByteTranscript,
    ) -> tuple[bool, ByteTranscript]:
        """Fig. 8's two checks: every `h_i` has a zero constant coefficient,
        and the Π_many^(2) proof of the `N + λ` relations verifies."""
        # The statement is the caller's and raises; the proof is the
        # prover's and is a verdict. See `zorch/lnp/wire.py`.
        publics.require(self.scheme)
        self._require_functions(r2, r1, r0, e2, e1, e0)
        if not self._is_well_formed(proof):
            return False, transcript

        t, gamma = self.garbage.gamma(transcript, proof.t_g, _count(e2))
        t = self.garbage.observe(t, proof.h)
        if not self.garbage.vanishes(proof.h):
            return False, t
        f2, f1, f0 = self._relations(
            self._aggregates(gamma, e2, e1, e0), r2, r1, r0, proof.h
        )
        return self.many.verify(
            publics,
            f2,
            f1,
            f0,
            t_a,
            self.garbage.commitment(t_b, proof.t_g),
            proof.quadratic,
            t,
        )

    def require_witness(self, name: str, s1: np.ndarray, s2: np.ndarray) -> None:
        """The witness gate of the masking this protocol ultimately proves
        against, deferred down the chain the way `_is_well_formed` is.

        Same reason: a layer above should not have to know that the masking
        sits three constructors down, and the tunnel `eval.many.quadratic
        .masking` would break on any re-parenting."""
        self.many.require_witness(name, s1, s2)

    def _is_well_formed(self, proof: QuadraticEvalProof) -> bool:
        """Whether `proof` is structurally usable — every field of it, in
        one place, per `zorch/lnp/wire.py`. The composite field defers to
        the layer that owns its wire."""
        scheme = self.many.scheme
        return (
            isinstance(proof, QuadraticEvalProof)
            and wire.is_stack(scheme, proof.t_g, self.lam)
            and wire.is_stack(scheme, proof.h, self.lam)
            and self.many._is_well_formed(proof.quadratic)
        )

    def _aggregates(
        self, gamma: np.ndarray, e2: np.ndarray, e1: np.ndarray, e0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`Σ_j γ_{i,j}·F_j` for each `i ∈ [λ]`, still over the caller's
        width — three block-major stacks, the shape `_embed` wants."""
        return self.garbage.aggregate(gamma, e2, e1, e0)

    def _relations(
        self,
        aggregates: tuple[np.ndarray, np.ndarray, np.ndarray],
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The `N + λ` quadratic functions the inner protocol is run on.

        The caller's `N` relations pass through unchanged in meaning but
        re-indexed (eq. 39). Each of the λ new ones is eq. 38,

            f_{N+i}(x) = x^{(g)}_{2,1,i} + Σ_j γ_{i,j}·F_j(x) − h_i,

        which is the aggregate this layer already needed for `h`, plus a
        selector for one garbage term and the revealed `h_i` as a constant.
        Building them here rather than in `prove` is what lets the verifier
        rebuild the identical statement from `Γ` and `h` alone."""
        ring = self.scheme.ring
        new2, new1, new0 = self._embed(*aggregates)
        # `_embed` writes only the caller's positions, so the garbage slots
        # it leaves zero take eq. 38's coefficient by assignment.
        new1[np.arange(self.lam), self._garbage_slots] = ring.one()
        new0 = ring.sub(new0, h[:, None])
        old2, old1, old0 = self._embed(r2, r1, r0)
        return (
            np.concatenate([old2, new2]),
            np.concatenate([old1, new1]),
            np.concatenate([old0, new0]),
        )

    def _embed(
        self, r2: np.ndarray, r1: np.ndarray, r0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A stack of quadratic functions over the caller's lift, re-indexed
        into the inner protocol's wider one.

        The two lifts differ only in what the message half carries: the
        caller writes against `[s1, σ(s1), m, σ(m)]`, the inner protocol
        against `[s1, σ(s1), m, g, σ(m), σ(g)]` — the garbage interleaved
        per automorphism copy, because `lift` orbits the message stack as a
        whole. So `s1`'s images keep their positions, each copy of `m`
        shifts right by the garbage the copies before it now carry, and
        `r0` is not indexed at all."""
        ring = self.many.scheme.ring
        n = self.many.width
        positions = self._positions
        wide2 = ring.zeros(len(r2), n, n)
        wide2[:, positions[:, None], positions[None, :]] = r2
        wide1 = ring.zeros(len(r1), n)
        wide1[:, positions] = r1
        return wide2, wide1, r0

    def _require_functions(
        self,
        r2: np.ndarray,
        r1: np.ndarray,
        r0: np.ndarray,
        e2: np.ndarray,
        e1: np.ndarray,
        e0: np.ndarray,
    ) -> None:
        """Both families are three aligned pieces over `self.width`.

        `N` may be zero — Fig. 8 carrying no relations is the direct
        generalization of Fig. 5, and the module convention's own answer for
        an empty stack makes it fall out. `M` may not: with nothing to
        aggregate, the λ garbage terms buy nothing and Fig. 7 is the
        protocol for that statement."""
        scheme = self.many.scheme
        n = self.width
        evaluations = _count(e2)
        if evaluations < 1:
            raise ValueError("eval: need at least one evaluation function")
        relations = _count(r2)
        for name, arr, lead in (
            ("r2", r2, (relations, n, n)),
            ("r1", r1, (relations, n)),
            ("r0", r0, (relations, 1)),
            ("e2", e2, (evaluations, n, n)),
            ("e1", e1, (evaluations, n)),
            ("e0", e0, (evaluations, 1)),
        ):
            scheme.require_stack(f"eval: {name}", arr, *lead)


def _count(arr: np.ndarray) -> int:
    """The leading-axis length of a function-family block, and zero for
    anything with no axes at all — so a malformed family reaches the shape
    gate below rather than an `IndexError` above it."""
    shape = getattr(arr, "shape", ())
    return int(shape[0]) if shape else 0
