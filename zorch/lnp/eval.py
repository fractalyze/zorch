# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π_eval — linear relations over Z_q, as constant coefficients over R_q.

The second protocol layer of the LNP framework (eprint 2022/284, Fig. 5).
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
from zorch.lnp import wire
from zorch.lnp.opening import AbdlopOpening, OpeningProof
from zorch.lnp.transcript import absorb_stacks

_LABEL_COMMIT = b"lnp/eval/garbage"
_LABEL_MASK = b"lnp/eval/masked"

# `γ` is drawn as a Z_q scalar off the transcript, and the byte sampler
# builds its draws from little-endian u64 chunks — so q must fit one.
_MAX_MODULUS = 1 << 64


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
        if lam < 1:
            raise ValueError(f"eval: lam must be positive, got {lam!r}")
        scheme = opening.scheme
        ell = scheme.messages - lam
        if ell < 0:
            raise ValueError(
                f"eval: the opening's BDLOP half carries {scheme.messages} "
                f"messages, too few for lam={lam} garbage terms on top of a "
                f"message vector — build it over the extended scheme"
            )
        ring = scheme.ring
        modulus = math.prod(ring.q_moduli)
        if modulus >= _MAX_MODULUS:
            raise ValueError(
                f"eval: γ is a Z_q scalar drawn from u64 chunks, so q must be "
                f"below 2^64; this ring's q has {modulus.bit_length()} bits. "
                f"An RNS chain this wide needs a wider γ sampler first."
            )
        self.opening = opening
        self.lam = lam
        self.ell = ell
        self.modulus = modulus

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
        ring = self.opening.scheme.ring
        self._require_functions(fs1, fm, target)
        s1_ring = ring.from_signed_stack(s1)
        s2_ring = ring.from_signed_stack(s2)

        # (1) garbage with a zero constant coefficient, committed under B_g.
        g = self._sample_garbage(rng)
        t_g = ring.add(ring.matvec(bg, s2_ring), g)

        # (2) γ, once t_g is bound.
        t, gamma = self._gamma(transcript, t_g, fs1.shape[0])

        # (3) the masked aggregates. Aggregating the M functions *before*
        #     applying them, rather than evaluating all M and summing, is
        #     the same value by linearity —
        #       h_j = g_j + Σ_u γ_{j,u}·(Fs1_u·s1 + Fm_u·m − target_u)
        #           = g_j + (Σ_u γ_{j,u}Fs1_u)·s1 + (Σ_u γ_{j,u}Fm_u)·m − …
        #     — and costs λ matvecs instead of M, over a ring whose mul is
        #     a deliberate O(d²) host oracle. The blocks are the eq.-28
        #     relation's own, so they are computed once and used twice.
        aggregated = self._blocks(fs1, fm, target, gamma)
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
        t = absorb_stacks(t.observe_label(_LABEL_MASK), h)
        # `u` is the verifier's side of eq. 28; the prover only supplies
        # the matrices its opening call takes.
        r1, rm, _ = self._relation(aggregated, h)
        opening, t = self.opening.prove(
            a1, a2, np.concatenate([b, bg]), r1, rm, s1, s2, rng, t
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
        ring = self.opening.scheme.ring
        # The statement is the caller's and raises; the proof is the
        # prover's and is a verdict. See `zorch/lnp/wire.py`.
        self._require_functions(fs1, fm, target)
        if not self._is_well_formed(proof):
            return False, transcript

        t, gamma = self._gamma(transcript, proof.t_g, fs1.shape[0])
        t = absorb_stacks(t.observe_label(_LABEL_MASK), proof.h)
        # Zero mod each q_i is zero mod q, by CRT — so the ring's own
        # constant-coefficient reading answers Fig. 5's `h̃_j = 0` check
        # directly, over every limb at once.
        if ring.constant_coeff(proof.h).any():
            return False, t
        r1, rm, u = self._relation(self._blocks(fs1, fm, target, gamma), proof.h)
        return self.opening.verify(
            a1,
            a2,
            np.concatenate([b, bg]),
            r1,
            rm,
            t_a,
            np.concatenate([t_b, proof.t_g]),
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
        scheme = self.opening.scheme
        return (
            isinstance(proof, EvalProof)
            and wire.is_stack(scheme, proof.t_g, self.lam)
            and wire.is_stack(scheme, proof.h, self.lam)
            and self.opening._is_well_formed(proof.opening)
        )

    def _sample_garbage(self, rng: np.random.Generator) -> np.ndarray:
        """`g ← {x ∈ R_q : x̃ = 0}^λ` — the ring's uniform stack with the
        constant coefficient forced to zero. Private coins: this is masking,
        like the Gaussian `y` of the layer below, so it comes off the
        caller's generator and never off the transcript.

        The zeroing is this protocol's own — `uniform_stack` is the module
        convention's uniform constructor, and `x̃ = 0` is Fig. 5's condition
        on the garbage, not a ring-level shape."""
        g = self.opening.scheme.ring.uniform_stack(rng, self.lam)
        g[..., 0] = 0
        return g

    def _gamma(
        self, transcript: ByteTranscript, t_g: np.ndarray, relations: int
    ) -> tuple[ByteTranscript, np.ndarray]:
        """Absorb the garbage commitment and squeeze `γ ∈ Z_q^{λ×M}` — the
        one derivation both sides replay."""
        count = self.lam * relations
        t = absorb_stacks(transcript.observe_label(_LABEL_COMMIT), t_g)
        t, raw = t.sample_scalar(uniform_bytes_needed(self.modulus, count))
        draws = uniform_from_bytes(raw, self.modulus, count)
        return t, draws.reshape(self.lam, relations)

    def _blocks(
        self,
        fs1: np.ndarray,
        fm: np.ndarray,
        target: np.ndarray,
        gamma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The γ-aggregated blocks `(Σ_u γ_{j,u}Fs1_u, Σ_u γ_{j,u}Fm_u,
        Σ_u γ_{j,u}target_u)`, one row per j.

        Both the prover's `h` and the eq.-28 relation are built from these,
        so aggregating is done once per proof rather than once per use —
        and the prover never materializes the M individual `F_u(s1, m)`.

        `combine` contracts the leading (per-relation) axis and carries
        whatever the block holds past it, so one call serves `target`'s
        stack of elements and `fs1`/`fm`'s stacks of whole matrix rows."""
        ring = self.opening.scheme.ring
        return tuple(  # type: ignore[return-value]
            np.stack([ring.combine(row, block) for row in gamma])
            for block in (fs1, fm, target)
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
        ring = self.opening.scheme.ring
        weighted_s1, weighted_m, weighted_target = aggregated
        eye = ring.zeros(self.lam, self.lam)
        diagonal = np.arange(self.lam)
        eye[diagonal, diagonal] = ring.one()
        return (
            weighted_s1,
            np.concatenate([weighted_m, eye], axis=1),
            ring.add(h, weighted_target),
        )

    def _require_functions(
        self, fs1: np.ndarray, fm: np.ndarray, target: np.ndarray
    ) -> None:
        """The M linear functions are three aligned pieces; a mismatch here
        would otherwise surface as a γ of the wrong width."""
        scheme = self.opening.scheme
        relations = fs1.shape[0]
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
