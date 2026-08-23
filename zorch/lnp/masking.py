# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The masking and rejection machinery every ABDLOP proof shares.

Fig. 4 (`opening.py`) and Fig. 6 (`quadratic.py`) are siblings, not
consumers of one another: both mask the witness with Gaussians `y_i ~
D_{s_i}`, absorb their own first-round messages, squeeze one challenge
`c ∈ C`, answer `z_i = c·s_i + y_i` over the *integers*, and rejection-
sample both responses. What differs between them is only which messages
go into the transcript and which equation the verifier checks — the
parameter point, the draw, the response, the Rej1 gates and the `[Ban93]`
norm bounds are one thing, held here.

They also have to be *the same* thing rather than merely alike. Fig. 8
runs a Π_eval-shaped layer over Π_many^(2), so a single proof carries
both protocols against one commitment; two parameter objects that drifted
apart would be a soundness bug no single-protocol suite could see.

Rejection is paid with a precomputed budget, the sampler discipline
lifted to the protocol loop: one attempt accepts with probability
≈ 1/(rep1·rep2) (Lemma 2.14-1: `M = exp(14/γ + 1/(2γ²))` at `s = γ·T`,
`T ≥ ‖c·s_i‖`), so a prover runs at most
`ceil(log(fail_prob)/log(1 − 1/(rep1·rep2)))` attempts and raises rather
than looping open-endedly. Each attempt restarts from the caller's
transcript value — `ByteTranscript` is functional, so a rejected attempt
leaves no trace, which is exactly the Fiat-Shamir-with-aborts convention.

Host/device boundary per `docs/reference/conventions.md`: the responses
live on the host by necessity — `z = c·s + y` must be computed over
unreduced ℤ (the norm statement is about magnitudes a field dtype would
fold away), and the rejection inner products and norm checks run over
exact Python ints (`lattice_frx.norms`). The host touches verdicts and
integer vectors, not ring arrays.

Prover randomness is the caller's `np.random.Generator` — private coins,
not transcript-derived.
"""

from __future__ import annotations

import math

import numpy as np
from lattice_frx import norms, sampler

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp import wire
from zorch.lnp.challenge import ChallengeParams, attempt_budget, negacyclic_mul
from zorch.lnp.transcript import absorb_stacks

# A budget past this means a mis-derived repetition rate, not a demanding
# `fail_prob`: well-derived rates (M ≈ 3–6 each) land in the hundreds.
_MAX_ATTEMPTS = 100_000


class Masking:
    """The parameter point a Fig.-4/Fig.-6 proof masks and rejects against.

    Takes the *derived* numbers: the masking standard deviations `s?_std`,
    the Lemma 2.14-1 repetition rates `rep?` they were derived with, the
    challenge point (`ChallengeParams`, carrying its own budget), and the
    `fail_prob` pricing the rejection loop. Parameter *derivation* (the
    γ-factors of §2.6/§6.1) stays with the consumer — this seam takes the
    derived numbers, like every other seam in this package.

    The relation count is deliberately not among them: it is a property of
    the statement each protocol is given, so storing it here would be a
    second representation of one number with nothing gating the two.
    """

    def __init__(
        self,
        scheme: AbdlopCommitment,
        s1_std: float,
        s2_std: float,
        rep1: float,
        rep2: float,
        challenge: ChallengeParams,
        fail_prob: float = 2.0**-128,
    ) -> None:
        for name, value in (("s1_std", s1_std), ("s2_std", s2_std)):
            if value <= 0.0:
                raise ValueError(f"masking: {name} must be positive, got {value!r}")
        for name, value in (("rep1", rep1), ("rep2", rep2)):
            if value <= 1.0:
                raise ValueError(f"masking: {name} must exceed 1, got {value!r}")
        # Rates past ~1e154 overflow their product to inf, which collapses
        # the acceptance to exactly 0.0 — and then `attempt_budget` refuses
        # `accept_prob`, a parameter this caller never passed, instead of
        # the budget guard below naming the rates that are actually wrong.
        if not math.isfinite(rep1 * rep2):
            raise ValueError(
                f"masking: rep1·rep2 overflows to infinity at rep1={rep1!r}, "
                f"rep2={rep2!r} — repetition rates this far from their Lemma "
                f"2.14-1 values are a parameter bug"
            )
        if not 0.0 < fail_prob < 1.0:
            raise ValueError(f"masking: fail_prob must be in (0, 1), got {fail_prob!r}")
        ring = scheme.ring
        if challenge.d != ring.d:
            raise ValueError(
                f"masking: challenge degree {challenge.d} does not match the "
                f"ring's {ring.d}"
            )
        self.scheme = scheme
        self.s1_std = s1_std
        self.s2_std = s2_std
        self.rep1 = rep1
        self.rep2 = rep2
        self.challenge = challenge
        self.fail_prob = fail_prob
        # Squeezed once per attempt, so it is resolved once here.
        self.challenge_bytes = challenge.bytes_needed
        # ‖z_i‖₂ ≤ s_i·√(2·m_i·d) [Ban93], compared squared over exact ints;
        # the floor only tightens the bound.
        self._bound1_sq = math.floor(s1_std**2 * 2 * scheme.s1_cols * ring.d)
        self._bound2_sq = math.floor(s2_std**2 * 2 * scheme.randomness_cols * ring.d)
        count1 = scheme.s1_cols * ring.d
        count2 = scheme.randomness_cols * ring.d
        # One attempt accepts both Rej1 gates with probability ≈ 1/(rep1·rep2).
        self.attempts = attempt_budget(fail_prob, 1.0 / (rep1 * rep2))
        # The budget grows as rep1·rep2·log(1/fail_prob), so a huge one means
        # a mis-derived rate — surface it here rather than as a prove() that
        # loops for hours.
        if self.attempts > _MAX_ATTEMPTS:
            raise ValueError(
                f"masking: rep1·rep2 = {rep1 * rep2!r} implies an attempt "
                f"budget of {self.attempts} — repetition rates this far "
                f"from their Lemma 2.14-1 values are a parameter bug"
            )
        # One resolved sampler per std: `sampler_for`'s tier decision is a
        # function of σ (its admissibility gate), so a callable resolved at
        # one std must not be invoked at the other.
        self._draw1 = sampler.sampler_for(s1_std, self.attempts * count1)
        self._draw2 = sampler.sampler_for(s2_std, self.attempts * count2)
        self._centers1 = np.zeros((scheme.s1_cols, ring.d), dtype=np.float64)
        self._centers2 = np.zeros((scheme.randomness_cols, ring.d), dtype=np.float64)

    def draw(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One masking pair `(y1, y2) ~ (D_{s1}, D_{s2})`, as signed integer
        `(m_i, d)` arrays."""
        return (
            self._draw1(rng, self._centers1, self.s1_std),
            self._draw2(rng, self._centers2, self.s2_std),
        )

    def challenge_from(
        self, transcript: ByteTranscript, label: bytes, *stacks: np.ndarray
    ) -> tuple[ByteTranscript, np.ndarray]:
        """Absorb a protocol's first-round messages under its own `label` and
        squeeze the challenge — the one derivation both sides replay.

        The label is the caller's because it is what separates two protocols
        that would otherwise hash the same stacks to the same challenge."""
        t = absorb_stacks(transcript.observe_label(label), *stacks)
        t, raw = t.sample_scalar(self.challenge_bytes)
        return t, self.challenge.from_bytes(raw)

    def respond(
        self, c: np.ndarray, s1: np.ndarray, s2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """`c·s_i` over the *integers*, one per witness half — the term both
        the response `z_i = c·s_i + y_i` and its Rej1 gate are stated in."""
        return _challenge_times(c, s1), _challenge_times(c, s2)

    def accepts(
        self,
        rng: np.random.Generator,
        z1: np.ndarray,
        cs1: np.ndarray,
        z2: np.ndarray,
        cs2: np.ndarray,
    ) -> bool:
        """Both Rej1 gates (Lemma 2.14-1) on one attempt's responses.

        Both coins are drawn before either gate is evaluated, so the rng
        stream a given seed produces stays a function of the attempt count
        alone — and the gates are then free to short-circuit, which matters
        because the exact-integer arithmetic is the expensive half of an
        attempt and a `1 − 1/rep1` share of attempts are already lost at the
        first gate."""
        coin1, coin2 = _coin(rng), _coin(rng)
        return _rej1(coin1, z1, cs1, self.s1_std, self.rep1) and _rej1(
            coin2, z2, cs2, self.s2_std, self.rep2
        )

    def within_bounds(self, z1: np.ndarray, z2: np.ndarray) -> bool:
        """`‖z_i‖₂ ≤ s_i·√(2·m_i·d)` [Ban93], the verifier's norm checks."""
        return (
            norms.l2_squared(z1) <= self._bound1_sq
            and norms.l2_squared(z2) <= self._bound2_sq
        )

    def exhausted(self, protocol: str) -> RuntimeError:
        """The error a prover raises when its whole budget was rejected —
        one message, because the diagnosis is the same for every protocol
        that masks this way."""
        return RuntimeError(
            f"{protocol}: every attempt was rejected — {self.attempts} "
            f"attempts at acceptance ≈ 1/(rep1·rep2) fail together with "
            f"probability <= {self.fail_prob!r}, so suspect the parameters "
            f"(a repetition rate far from its Lemma 2.14-1 value), not bad "
            f"luck."
        )

    def require_witness(self, name: str, s1: np.ndarray, s2: np.ndarray) -> None:
        """The witness halves are the caller's, so a malformed one raises —
        `zorch/lnp/wire.py`'s statement side."""
        ring = self.scheme.ring
        wire.require_signed(ring, f"{name}: s1", s1, self.scheme.s1_cols)
        wire.require_signed(ring, f"{name}: s2", s2, self.scheme.randomness_cols)

    def is_response(self, c: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> bool:
        """Whether an untrusted `(c, z1, z2)` triple is structurally usable —
        the shared half of every masked protocol's `_is_well_formed`."""
        ring = self.scheme.ring
        return (
            wire.is_signed(ring, c)
            and wire.is_signed(ring, z1, self.scheme.s1_cols)
            and wire.is_signed(ring, z2, self.scheme.randomness_cols)
        )


def _challenge_times(c: np.ndarray, signed: np.ndarray) -> np.ndarray:
    """`c·s` over the *integers*, one negacyclic product per row: the
    responses' norm statement lives in unreduced ℤ, so no ring (mod-q)
    product may touch them. int64 is exact here — coefficients are bounded
    by `d·κ·‖s‖∞` plus a Gaussian tail, orders of magnitude under 2^63."""
    return np.stack(
        [negacyclic_mul(c.astype(np.int64), row.astype(np.int64)) for row in signed]
    )


def _coin(rng: np.random.Generator) -> float:
    """One Rej1 coin, floored off zero so its `log` stays finite."""
    return max(float(rng.random()), np.finfo(np.float64).tiny)


def _rej1(coin: float, z: np.ndarray, v: np.ndarray, std: float, rep: float) -> bool:
    """Rej1 (Fig. 1): accept iff `coin < (1/M)·exp((−2⟨z,v⟩ + ‖v‖²)/(2s²))`,
    taken in log space so a large positive exponent cannot overflow, with
    the inner product and norm over exact Python ints (a fixed-width dot
    would wrap at real parameter scales).

    `‖v‖²` is `vo @ vo` rather than `norms.l2_squared` to reuse the
    object-dtype view the inner product already needs: the substrate's
    norm is now the *same* self-dot, so the two are identical arithmetic,
    but calling it here would convert `v` a second time — measured ~1.14x
    slower at `(4, 128)` and `(16, 256)`. `within_bounds` has no such view
    to reuse and calls the substrate. This stays hand-spelled only while
    the inner product is: a substrate op that returns `⟨z,v⟩` and `‖v‖²`
    off one conversion would take both lines."""
    zo = z.reshape(-1).astype(object)
    vo = v.reshape(-1).astype(object)
    threshold = (-2 * int(zo @ vo) + int(vo @ vo)) / (2.0 * std**2) - math.log(rep)
    return math.log(coin) < threshold
