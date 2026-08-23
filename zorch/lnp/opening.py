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

The masking, the rejection budget and the `[Ban93]` norm bounds are not
this protocol's own — Fig. 6 masks against exactly the same ones, and
Fig. 8 runs both protocols against a single commitment. They live on the
`Masking` this is built over (`masking.py`), together with the Ajtai
commitment algebra both protocols mask against and the host/device
boundary they imply. What is this module's own is `v` — the relation
message — and the equation that recomputes it.

The transcript arrives already bound to the statement (the caller absorbed
the commitment); this protocol absorbs only its own messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zorch.byte_transcript import ByteTranscript
from zorch.lnp.masking import Masking

_LABEL = b"lnp/open"


@dataclass(frozen=True)
class OpeningProof:
    """The non-interactive Π_many wire: the challenge and the two masked
    responses, as signed integer coefficient vectors (`int64`; `z_i` is
    `(m_i, d)`, `c` is `(d,)`). `w` and `v` are absent by design — see the
    module docstring on why the verifier recomputes them."""

    c: np.ndarray
    z1: np.ndarray
    z2: np.ndarray


class AbdlopOpening:
    """Π_many prove/verify over an `AbdlopCommitment` (Fig. 4).

    The proof parameters — masking deviations, repetition rates, the
    challenge point, the rejection budget — live on the `Masking` this is
    built over, because Fig. 6 masks against the same ones and Fig. 8 runs
    both protocols against a single commitment. See `masking.py`.

    The relation count is not among them: it is `r1.shape[0]`, which both
    `prove` and `verify` already receive, so storing it would be a second
    representation of one number with nothing gating the two against each
    other."""

    def __init__(self, masking: Masking) -> None:
        self.masking = masking
        self.scheme = masking.scheme

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
        masking = self.masking
        masking.require_witness("opening.prove", s1, s2)
        for _ in range(masking.attempts):
            y1, y2 = masking.draw(rng)
            y1_ring = ring.from_signed_stack(y1)
            y2_ring = ring.from_signed_stack(y2)
            w = masking.ajtai_mask(a1, a2, y1_ring, y2_ring)
            v = ring.sub(
                ring.matvec(r1, y1_ring),
                ring.matvec(rm, ring.matvec(b, y2_ring)),
            )
            advanced, c = masking.challenge_from(transcript, _LABEL, w, v)
            cs1, cs2 = masking.respond(c, s1, s2)
            z1 = cs1 + y1
            z2 = cs2 + y2
            if masking.accepts(rng, z1, cs1, z2, cs2):
                return OpeningProof(c=c, z1=z1, z2=z2), advanced
        raise masking.exhausted("opening.prove")

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
        # `proof` is the prover's, so malformed is a verdict; the publics
        # `t_a`, `t_b` and `u` are the caller's and keep raising, from the
        # ring ops they reach. See `zorch/lnp/wire.py`.
        if not self._is_well_formed(proof):
            return False, transcript
        if not self.masking.within_bounds(proof.z1, proof.z2):
            return False, transcript
        c_elem = ring.from_signed(proof.c)
        z1_ring = ring.from_signed_stack(proof.z1)
        z2_ring = ring.from_signed_stack(proof.z2)
        w = self.masking.recomputed_ajtai_mask(a1, a2, z1_ring, z2_ring, c_elem, t_a)
        # `c·m` for the implicit message `m = t_B − B·s2`: the same masking
        # the responses carry, applied to the quantity the BDLOP half commits
        # to but never sends.
        z_m = self.masking.masked_message(c_elem, t_b, b, z2_ring)
        v = ring.sub(
            ring.add(ring.matvec(r1, z1_ring), ring.matvec(rm, z_m)),
            ring.scale(c_elem, u),
        )
        advanced, c = self.masking.challenge_from(transcript, _LABEL, w, v)
        return bool(np.array_equal(c, proof.c)), advanced

    def _is_well_formed(self, proof: OpeningProof) -> bool:
        """Whether `proof` is structurally usable — every field of it, in
        one place.

        One gate over the whole dataclass rather than a check at each point
        of use: per-field gating leaves whichever field nobody remembered
        ungated, and `verify` then crashes on exactly the message this is
        meant to reject. `Π_eval` calls this for the opening it nests, so
        the nested proof meets the same gate as a top-level one."""
        return isinstance(proof, OpeningProof) and self.masking.is_response(
            proof.c, proof.z1, proof.z2
        )
