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

from dataclasses import dataclass

import numpy as np
from lattice_frx.sampler import uniform_bytes_needed, uniform_from_bytes

from zorch.byte_transcript import ByteTranscript
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
        modulus = 1
        for q in ring.q_moduli:
            modulus *= q
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

        # (3) the masked aggregates.
        values = self._values(fs1, fm, target, s1_ring, message)
        h = np.stack([self._aggregate(g[j], values, gamma[j]) for j in range(self.lam)])

        # (4) the same aggregation as a linear relation over (s1, m‖g),
        #     proved by the layer below.
        t = absorb_stacks(t.observe_label(_LABEL_MASK), h)
        r1, rm, u = self._relation(fs1, fm, target, gamma, h)
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
        self._require_functions(fs1, fm, target)
        self._require_stack("verify: t_g", proof.t_g, self.lam)
        self._require_stack("verify: h", proof.h, self.lam)

        t, gamma = self._gamma(transcript, proof.t_g, fs1.shape[0])
        t = absorb_stacks(t.observe_label(_LABEL_MASK), proof.h)
        # The constant coefficient is coefficient 0 of every limb: zero mod
        # each q_i is zero mod q, by CRT.
        if proof.h[..., 0].any():
            return False, t
        r1, rm, u = self._relation(fs1, fm, target, gamma, proof.h)
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

    def _sample_garbage(self, rng: np.random.Generator) -> np.ndarray:
        """`g ← {x ∈ R_q : x̃ = 0}^λ` — uniform per limb (uniform over R_q by
        CRT), with the constant coefficient forced to zero. Private coins:
        this is masking, like the Gaussian `y` of the layer below, so it
        comes off the caller's generator and never off the transcript."""
        ring = self.opening.scheme.ring
        g = np.stack(
            [
                rng.integers(0, q, size=(self.lam, ring.d), dtype=np.uint64)
                for q in ring.q_moduli
            ],
            axis=-2,
        )
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

    def _values(
        self,
        fs1: np.ndarray,
        fm: np.ndarray,
        target: np.ndarray,
        s1_ring: np.ndarray,
        message: np.ndarray,
    ) -> np.ndarray:
        """`F_u(s1, m) = Fs1_u·s1 + Fm_u·m − target_u`, all M at once."""
        ring = self.opening.scheme.ring
        return ring.sub(
            ring.add(ring.matvec(fs1, s1_ring), ring.matvec(fm, message)), target
        )

    def _aggregate(
        self, base: np.ndarray, values: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """`base + Σ_u weights_u · values_u`, the row shape of both `h` and
        the relation matrices."""
        ring = self.opening.scheme.ring
        for value, weight in zip(values, weights):
            base = ring.add(base, ring.mul_scalar(value, int(weight)))
        return base

    def _relation(
        self,
        fs1: np.ndarray,
        fm: np.ndarray,
        target: np.ndarray,
        gamma: np.ndarray,
        h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Eq. 28 in Π_many's `(R1, Rm, u)` shape over the message `m‖g`.

        Row j of `f_j(s1, m‖g) = g_j + Σ_u γ_{j,u}·F_u(s1, m) − h_j = 0`
        rearranges to `R1_j·s1 + Rm_j·(m‖g) = u_j` with

            R1_j  = Σ_u γ_{j,u}·Fs1_u
            Rm_j  = [ Σ_u γ_{j,u}·Fm_u | e_j ]     (`e_j` selects `g_j`)
            u_j   = h_j + Σ_u γ_{j,u}·target_u

        — the `e_j` block being why the inner opening has to be built over
        the extended scheme."""
        ring = self.opening.scheme.ring
        zero = self._zeros(self.lam)
        one = ring.from_signed([1] + [0] * (ring.d - 1))
        r1, rm, u = [], [], []
        for j in range(self.lam):
            selector = zero.copy()
            selector[j] = one
            r1.append(self._aggregate(self._zeros(fs1.shape[1]), fs1, gamma[j]))
            rm.append(
                np.concatenate(
                    [
                        self._aggregate(self._zeros(fm.shape[1]), fm, gamma[j]),
                        selector,
                    ]
                )
            )
            u.append(self._aggregate(h[j], target, gamma[j]))
        return np.stack(r1), np.stack(rm), np.stack(u)

    def _zeros(self, rows: int) -> np.ndarray:
        ring = self.opening.scheme.ring
        return np.zeros((rows, len(ring.q_moduli), ring.d), dtype=np.uint64)

    def _require_functions(
        self, fs1: np.ndarray, fm: np.ndarray, target: np.ndarray
    ) -> None:
        """The M linear functions are three aligned pieces; a mismatch here
        would otherwise surface as a γ of the wrong width."""
        scheme = self.opening.scheme
        relations = fs1.shape[0]
        if relations < 1:
            raise ValueError("eval: need at least one linear function")
        for name, arr, want in (
            ("fs1", fs1, (relations, scheme.s1_cols)),
            ("fm", fm, (relations, self.ell)),
        ):
            if arr.shape[:2] != want:
                raise ValueError(
                    f"eval: {name} must lead with {want}, got {arr.shape[:2]}"
                )
        self._require_stack("target", target, relations)

    def _require_stack(self, name: str, arr: np.ndarray, rows: int) -> None:
        ring = self.opening.scheme.ring
        want = (rows, len(ring.q_moduli), ring.d)
        if not isinstance(arr, np.ndarray) or arr.shape != want:
            raise ValueError(
                f"eval: {name} must be a ring stack of shape {want}, got "
                f"{getattr(arr, 'shape', type(arr).__name__)}"
            )
