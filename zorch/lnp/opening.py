# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π_many — ZK opening of an ABDLOP commitment with N linear relations.

The first protocol layer of the LNP framework (eprint 2022/284, Fig. 4):
prove knowledge of `(s1, s2)` opening `t_A = A1·s1 + A2·s2` — with the
message implicitly `m = t_B − B·s2` — such that `‖z_i‖` stays small and the
N relations `R1·s1 + Rm·m = u` over R_q hold. One masked response pair
carries all N relations, so the proof size is independent of N.

The interactive shape, made non-interactive against the `ByteTranscript`
seam: the prover masks with Gaussians `y_i ~ D_{s_i}`, absorbs
`w = A1·y1 + A2·y2` and `v = R1·y1 − Rm·B·y2` into the transcript, squeezes
the challenge `c ∈ C` (`challenge.py`), answers `z_i = c·s_i + y_i` over the
*integers*, and rejection-samples both responses (Rej1, Lemma 2.14-1 —
leakage-free, so plain MLWE; Rej2/Rej0 are a recorded later optimization).
The wire is `(c, z1, z2)`: the verifier recomputes `w = A1·z1 + A2·z2 − c·t_A`
and `v = R1·z1 + Rm·(c·t_B − B·z2) − c·u` from the verification equations,
replays the absorb/squeeze, and accepts iff the recomputed challenge equals
`c` and both `‖z_i‖₂ ≤ s_i·√(2·m_i·d)` — hashing (w, v) instead of sending
them is what makes the proof N-independent, and it is the paper's own
Fiat-Shamir shape.

Rejection is paid with a precomputed budget, the sampler discipline lifted
to the protocol loop: one attempt accepts with probability ≈ 1/(M1·M2)
(Lemma 2.14-1: `M = exp(14/γ + 1/(2γ²))` at `s = γ·T`, `T ≥ ‖c·s_i‖`), so
`prove` runs at most `ceil(log(fail_prob)/log(1 − 1/(M1·M2)))` attempts and
raises rather than looping open-endedly. Each attempt restarts from the
caller's transcript value — `ByteTranscript` is functional, so a rejected
attempt leaves no trace, which is exactly the Fiat-Shamir-with-aborts
convention. The transcript arrives already bound to the statement (the
caller absorbed the commitment); this protocol absorbs only its own
messages.

Host/device boundary per `docs/reference/conventions.md`: the ring algebra
is ring ops only (`matvec`/`add`/`sub`/`mul` — traced the moment the split
ring grows its traced leg), while the responses live on the host by
necessity — `z = c·s + y` must be computed over unreduced ℤ (the norm
statement is about magnitudes a field dtype would fold away), and the
rejection inner products and norm checks run over exact Python ints
(`lattice_frx.norms`). The host touches verdicts and integer vectors, not
ring arrays.

Prover randomness: masking uses lattice-frx's Gaussian tier with an
injected `np.random.Generator` — private coins, not transcript-derived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from lattice_frx import norms, sampler
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.challenge import (
    challenge_bytes_needed,
    challenge_from_bytes,
    negacyclic_mul,
)

_LABEL = b"lnp/open"


@dataclass(frozen=True)
class OpeningProof:
    """The non-interactive Π_many wire: the challenge and the two masked
    responses, as signed integer coefficient vectors (`int64`; `z_i` is
    `(m_i, d)`, `c` is `(d,)`). `w` and `v` are deliberately absent — the
    verifier recomputes them, which is what keeps the proof independent
    of the relation count."""

    c: np.ndarray
    z1: np.ndarray
    z2: np.ndarray


class AbdlopOpening:
    """Π_many prove/verify over an `AbdlopCommitment` (Fig. 4).

    The scheme object carries the ring and the module shape; this class
    adds the proof parameters: the masking standard deviations `s?_std`,
    the Lemma 2.14-1 repetition rates `rep?` they were derived with, the
    challenge-space point `(kappa, eta, k)`, and the budget's `fail_prob`.
    Parameter *derivation* (the γ-factors of §2.6/§6.1) stays with the
    consumer — this seam takes the derived numbers."""

    def __init__(
        self,
        scheme: AbdlopCommitment,
        relations: int,
        s1_std: float,
        s2_std: float,
        rep1: float,
        rep2: float,
        kappa: int,
        eta: int,
        k: int,
        fail_prob: float = 2.0**-128,
    ) -> None:
        if relations < 0:
            raise ValueError(f"opening: relations must be >= 0, got {relations!r}")
        for name, value in (("s1_std", s1_std), ("s2_std", s2_std)):
            if value <= 0.0:
                raise ValueError(f"opening: {name} must be positive, got {value!r}")
        for name, value in (("rep1", rep1), ("rep2", rep2)):
            if value <= 1.0:
                raise ValueError(f"opening: {name} must exceed 1, got {value!r}")
        if not 0.0 < fail_prob < 1.0:
            raise ValueError(f"opening: fail_prob must be in (0, 1), got {fail_prob!r}")
        self.scheme = scheme
        self.relations = relations
        self.s1_std = s1_std
        self.s2_std = s2_std
        self.rep1 = rep1
        self.rep2 = rep2
        self.kappa = kappa
        self.eta = eta
        self.k = k
        self.fail_prob = fail_prob
        ring = scheme.ring
        # ‖z_i‖₂ ≤ s_i·√(2·m_i·d) [Ban93], compared squared over exact ints;
        # the floor only tightens the bound.
        self._bound1_sq = math.floor(s1_std**2 * 2 * scheme.s1_cols * ring.d)
        self._bound2_sq = math.floor(s2_std**2 * 2 * scheme.randomness_cols * ring.d)
        self._challenge_bytes = challenge_bytes_needed(ring.d, kappa, eta, k)
        # One resolved sampler per std: `sampler_for`'s tier decision is a
        # function of σ (its admissibility gate), so a callable resolved at
        # one std must not be invoked at the other.
        count1 = scheme.s1_cols * ring.d
        count2 = scheme.randomness_cols * ring.d
        # One attempt accepts both Rej1 gates with probability ≈ 1/(rep1·rep2).
        accept = 1.0 / (rep1 * rep2)
        self._attempts = math.ceil(math.log(fail_prob) / math.log1p(-accept))
        # The budget grows as rep1·rep2·log(1/fail_prob): well-derived
        # repetition rates (M ≈ 3–6 each) land in the hundreds, and a huge
        # budget means a mis-derived rate — surface it here rather than as
        # a prove() that loops for hours.
        if self._attempts > 100_000:
            raise ValueError(
                f"opening: rep1·rep2 = {rep1 * rep2!r} implies an attempt "
                f"budget of {self._attempts} — repetition rates this far "
                f"from their Lemma 2.14-1 values are a parameter bug"
            )
        self._draw1 = sampler.sampler_for(s1_std, self._attempts * count1)
        self._draw2 = sampler.sampler_for(s2_std, self._attempts * count2)

    def prove(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        r1: np.ndarray,
        rm: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        rng: np.random.Generator,
        transcript: ByteTranscript,
    ) -> tuple[OpeningProof, ByteTranscript]:
        """One non-interactive proof, and the transcript advanced past it.

        `s1`/`s2` arrive as signed integer `(m_i, d)` arrays — the raw
        witness form the samplers emit; `ring.from_signed` of their rows
        are the columns `commit` was called with. The commitment
        `(t_a, t_b)` and target `u` are deliberately absent: the prover's
        messages never read them — the transcript arrived bound to the
        statement, and the asymmetry with `verify` documents exactly
        that convention."""
        ring = self.scheme.ring
        self._require_signed("prove: s1", s1, self.scheme.s1_cols)
        self._require_signed("prove: s2", s2, self.scheme.randomness_cols)
        centers1 = np.zeros_like(s1, dtype=np.float64)
        centers2 = np.zeros_like(s2, dtype=np.float64)
        for _ in range(self._attempts):
            y1 = self._draw1(rng, centers1, self.s1_std)
            y2 = self._draw2(rng, centers2, self.s2_std)
            y1_ring = ring.from_signed_stack(y1)
            y2_ring = ring.from_signed_stack(y2)
            w = ring.add(ring.matvec(a1, y1_ring), ring.matvec(a2, y2_ring))
            if self.relations:
                v = ring.sub(
                    ring.matvec(r1, y1_ring),
                    ring.matvec(rm, ring.matvec(b, y2_ring)),
                )
            else:
                v = _empty_stack(ring)
            advanced, c = self._challenge(transcript, w, v)
            cs1 = _challenge_times(c, s1)
            cs2 = _challenge_times(c, s2)
            z1 = cs1 + y1
            z2 = cs2 + y2
            # Both gates draw their coin every attempt, accepted or not, so
            # the rng stream a given seed produces is a function of the
            # attempt count alone.
            ok1 = _rej1(rng, z1, cs1, self.s1_std, self.rep1)
            ok2 = _rej1(rng, z2, cs2, self.s2_std, self.rep2)
            if ok1 and ok2:
                return OpeningProof(c=c, z1=z1, z2=z2), advanced
        raise RuntimeError(
            f"opening.prove: every attempt was rejected — {self._attempts} "
            f"attempts at acceptance ≈ 1/(rep1·rep2) fail together with "
            f"probability <= {self.fail_prob!r}, so suspect the parameters "
            f"(a repetition rate far from its Lemma 2.14-1 value), not bad "
            f"luck."
        )

    def verify(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        r1: np.ndarray,
        rm: np.ndarray,
        t_a: np.ndarray,
        t_b: np.ndarray,
        u: np.ndarray,
        proof: OpeningProof,
        transcript: ByteTranscript,
    ) -> tuple[bool, ByteTranscript]:
        """Fig. 4's three checks in their non-interactive shape: both norm
        bounds, then the recomputed `(w, v)` must replay to the proof's
        challenge — which is checks 2 and 3 folded into the hash."""
        ring = self.scheme.ring
        self._require_signed("verify: c", proof.c)
        self._require_signed("verify: z1", proof.z1, self.scheme.s1_cols)
        self._require_signed("verify: z2", proof.z2, self.scheme.randomness_cols)
        if norms.l2_squared(proof.z1.reshape(-1)) > self._bound1_sq:
            return False, transcript
        if norms.l2_squared(proof.z2.reshape(-1)) > self._bound2_sq:
            return False, transcript
        c_elem = ring.from_signed(proof.c)
        z1_ring = ring.from_signed_stack(proof.z1)
        z2_ring = ring.from_signed_stack(proof.z2)
        w = ring.sub(
            ring.add(ring.matvec(a1, z1_ring), ring.matvec(a2, z2_ring)),
            ring.scale(c_elem, t_a),
        )
        if self.relations:
            v = ring.sub(
                ring.add(
                    ring.matvec(r1, z1_ring),
                    ring.matvec(
                        rm,
                        ring.sub(ring.scale(c_elem, t_b), ring.matvec(b, z2_ring)),
                    ),
                ),
                ring.scale(c_elem, u),
            )
        else:
            v = _empty_stack(ring)
        advanced, c = self._challenge(transcript, w, v)
        return bool(np.array_equal(c, proof.c)), advanced

    def _challenge(
        self, transcript: ByteTranscript, w: np.ndarray, v: np.ndarray
    ) -> tuple[ByteTranscript, np.ndarray]:
        """Absorb this protocol's messages and squeeze the challenge — the
        one definition both sides replay. Serialization is the canonical
        residues as C-order little-endian u64."""
        t = transcript.observe_label(_LABEL)
        t = t.observe_slice(_stack_bytes(w), w.shape[0])
        t = t.observe_slice(_stack_bytes(v), v.shape[0])
        t, raw = t.sample_scalar(self._challenge_bytes)
        ring = self.scheme.ring
        return t, challenge_from_bytes(raw, ring.d, self.kappa, self.eta, self.k)

    def _require_signed(self, name: str, arr: np.ndarray, *lead: int) -> None:
        """The signed-integer twin of the ring-side shape gates: witnesses
        and responses are `(lead, d)` integer arrays, the challenge the
        rank-one case `(d,)`. Every field of the wire `OpeningProof` passes
        through here, so a malformed proof fails in this module's
        vocabulary instead of deep inside a ring op — and a `c` that is a
        list, or float coefficients silently truncated by `from_signed`,
        is refused rather than quietly scored against the real challenge."""
        want = (*lead, self.scheme.ring.d)
        if not isinstance(arr, np.ndarray) or not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(
                f"opening: {name} must be a signed integer ndarray, got "
                f"{type(arr).__name__} dtype {getattr(arr, 'dtype', None)!r}"
            )
        if arr.shape != want:
            raise ValueError(f"opening: {name} must have shape {want}, got {arr.shape}")


def _empty_stack(ring: HostSplitRing) -> np.ndarray:
    return np.empty((0, len(ring.q_moduli), ring.d), dtype=np.uint64)


def _stack_bytes(stack: np.ndarray) -> bytes:
    """Canonical residues as C-order little-endian u64 — the deterministic
    wire form both sides absorb. This body, together with `_challenge`'s
    `observe_slice` count convention (ring *elements*, not words), is the
    lnp transcript wire for ring stacks: a sibling protocol module must
    share this definition, never respell it."""
    return np.ascontiguousarray(stack).astype("<u8", copy=False).tobytes()


def _challenge_times(c: np.ndarray, signed: np.ndarray) -> np.ndarray:
    """`c·s` over the *integers*, one negacyclic product per row: the
    responses' norm statement lives in unreduced ℤ, so no ring (mod-q)
    product may touch them. int64 is exact here — coefficients are bounded
    by `d·κ·‖s‖∞` plus a Gaussian tail, orders of magnitude under 2^63."""
    return np.stack(
        [negacyclic_mul(c.astype(np.int64), row.astype(np.int64)) for row in signed]
    )


def _rej1(
    rng: np.random.Generator,
    z: np.ndarray,
    v: np.ndarray,
    std: float,
    rep: float,
) -> bool:
    """Rej1 (Fig. 1): accept iff `u < (1/M)·exp((−2⟨z,v⟩ + ‖v‖²)/(2s²))`,
    taken in log space so a large positive exponent cannot overflow, with
    the inner product and norm over exact Python ints (a fixed-width dot
    would wrap at real parameter scales)."""
    dot = int(z.reshape(-1).astype(object) @ v.reshape(-1).astype(object))
    v_sq = norms.l2_squared(v.reshape(-1))
    threshold = (-2 * dot + v_sq) / (2.0 * std**2) - math.log(rep)
    coin = max(float(rng.random()), np.finfo(np.float64).tiny)
    return math.log(coin) < threshold
