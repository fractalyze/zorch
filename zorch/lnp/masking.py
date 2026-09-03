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

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from lattice_frx import norms, sampler
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp import wire
from zorch.lnp.challenge import ChallengeParams, attempt_budget, negacyclic_mul
from zorch.lnp.transcript import absorb_stacks

# The paper's projection dimension: Lemma 2.8/2.9 are stated at 256 rows and
# their 2^-128 tail bounds are what fix it, so the constants 26, 337 and 41 in
# the proven bound are all 256's. A default rather than a hard-coding so a test
# can shrink it, never so a consumer can tune it.
PROJECTION = 256

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

    def ajtai_image(
        self, a1: np.ndarray, a2: np.ndarray, x1: np.ndarray, x2: np.ndarray
    ) -> np.ndarray:
        """`A1·x1 + A2·x2`, the Ajtai half of the commitment equation.

        Both siblings send it as their first message at `x = y` and both
        rebuild it at `x = z` to check it, so it is one expression at four
        sites, not two protocols that happen to agree. Here rather than on
        `AbdlopCommitment` because what the layers pass is a *masking* or a
        *response*, not a witness — the scheme's own `commit` is the
        witness-shaped caller and keeps its bound checks."""
        ring = self.scheme.ring
        return ring.add(ring.matvec(a1, x1), ring.matvec(a2, x2))

    def masked_message(
        self, c: np.ndarray, b: np.ndarray, t_b: np.ndarray, z2: np.ndarray
    ) -> np.ndarray:
        """`c·t_B − B·z2` — `c·m` for the message the BDLOP half commits to
        but never sends.

        The only route either verifier has to `m`: Fig. 4 feeds it to the
        linear check, Fig. 6 lifts it as eq. 30's message half. Named once
        because the two must agree on it and a suite of either alone cannot
        see them drift."""
        ring = self.scheme.ring
        return ring.sub(ring.scale(c, t_b), ring.matvec(b, z2))

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
    return max(float(rng.random()), float(np.finfo(np.float64).tiny))


def _rej1(coin: float, z: np.ndarray, v: np.ndarray, std: float, rep: float) -> bool:
    """Rej1 (Fig. 1): accept iff `coin < (1/M)·exp((−2⟨z,v⟩ + ‖v‖²)/(2s²))`,
    taken in log space so a large positive exponent cannot overflow, with
    the inner product and norm over exact Python ints (a fixed-width dot
    would wrap at real parameter scales).

    Both quantities come off `_exact_dots`, which is the one-conversion
    seam this docstring used to ask for."""
    zv, vv = _exact_dots(z, v)
    threshold = (-2 * zv + vv) / (2.0 * std**2) - math.log(rep)
    return math.log(coin) < threshold


def _exact_dots(z: np.ndarray, v: np.ndarray) -> tuple[int, int]:
    """`(⟨z, v⟩, ‖v‖²)` over exact Python ints, off one object-dtype view.

    Both rejection gates need exactly this pair, and both need it in
    unreduced ℤ — a fixed-width dot wraps at real parameter scales, which
    is a silently wrong verdict rather than an error.

    `‖v‖²` is `vo @ vo` rather than `norms.l2_squared` to reuse the view
    the inner product already built: the substrate's norm is now the same
    self-dot, so the arithmetic is identical, but calling it would convert
    `v` a second time — measured ~1.14x slower at `(4, 128)` and
    `(16, 256)`. `within_bounds` has no view to reuse and calls the
    substrate."""
    zo = z.reshape(-1).astype(object)
    vo = v.reshape(-1).astype(object)
    return int(zo @ vo), int(vo @ vo)


# §2.6's tail bound, and the reason the ℓ∞ gate needs no coefficient of its
# own: `|⟨z, v⟩| < 14·s·‖v‖` holds with probability `1 − 2^-128` for
# `z ← D_s^ℓ` [Ban93, Lyu12], and a unit `v` reads that one coordinate at a
# time. The same 14 the Rej1 repetition rate `exp(14/γ + 1/(2γ²))` is built
# from, so it is one constant in two places rather than two constants.
_LINF_TAIL = 14

# Whichever norm a gate reads, and the exact integer it compares against.
_Gate = tuple[Callable[[np.ndarray], int], int]


@dataclass(frozen=True)
class L2Bound:
    """Prop. 5.1's `‖⃗z‖₂ ≤ t·√256·s` — what Fig. 9, and Fig. 10's exact-ℓ2
    leg, verify the revealed projection against.

    `accept_t` is the paper's `t ≥ 1.64`, and it lives on the gate rather
    than on the masking because it is *this* gate's slack and means nothing
    to the other one. A masking carrying an `accept_t` its gate ignored
    would be a parameter nobody could act on."""

    accept_t: float = 1.64

    def __post_init__(self) -> None:
        if self.accept_t <= 0.0:
            raise ValueError(
                f"masking: accept_t must be positive, got {self.accept_t!r}"
            )

    def resolve(self, projection: int, mask_std: float) -> _Gate:
        # Compared squared over exact ints so the reveal never round-trips
        # through a float; the floor only tightens the bound, as in
        # `Masking`.
        return norms.l2_squared, math.floor(
            (self.accept_t**2) * projection * mask_std**2
        )

    def proven_norm(self, projection: int, mask_std: float) -> int:
        """`B = 2·√(256/26)·t·s` — what a verifying proof actually says about
        the *projected vector*, as opposed to about `⃗z`.

        `resolve` above gives the gate on `⃗z`; this is the conclusion Lemma
        2.9 draws from it, `‖⃗s‖ ≤ B`, and it is a different number. A
        consumer that needs the bound rather than the gate — §5.2's
        wraparound conditions are the caller — had no way to ask for it, and
        `range.py` stated the formula in prose while nothing computed it.
        The first caller to need it passed the projection *dimension*, a
        count, and the conditions were slack enough not to notice.

        Rounded up, since it is an upper bound and a floor here would claim
        something the proof does not."""
        return math.ceil(2.0 * math.sqrt(projection / 26.0) * self.accept_t * mask_std)


@dataclass(frozen=True)
class LinfBound:
    """Fig. 10's `‖⃗z‖_∞ ≤ 14·s` — what its approximate-ℓ∞ leg verifies the
    revealed projection against.

    A *tail* bound rather than a slack: Rej0's accepted output really is
    `D_s^{256}` centred at zero (Lemma 2.14-3), not the bimodal mixture the
    prover drew from, so what the verifier is checking is that a Gaussian
    stayed inside its own tail. That is why the paper fixes 14 here where it
    lets the ℓ2 gate carry a tunable `t` — there is nothing to trade against
    the repetition rate.

    `projection` is unread: `14·s` bounds one coordinate, so the dimension
    that makes `√256` appear in the ℓ2 bound does not appear here.

    Note what this does *not* change. The leg still draws `256/d` mask
    elements at `s = γ·√337·α` for an **ℓ2** bound `α` on the projected
    vector — Fig. 10 derives both legs' deviations that way, since `√337` is
    Lemma 2.8-2's ℓ2 projection growth either way. Only the verifier's gate
    differs."""

    def resolve(self, projection: int, mask_std: float) -> _Gate:
        return norms.linf, math.floor(_LINF_TAIL * mask_std)

    def proven_norm(self, projection: int, mask_std: float) -> int:
        """Refused, and the refusal is Theorem 5.3's own scoping rather than
        a gap waiting to be filled.

        Fig. 10 runs two legs. This one is `(d)`: it establishes eq. 52's
        approximate ℓ∞ statement and reaches `‖e⃗^(d)‖_∞ ≤ 24·s^(d)` through
        **Lemma 2.7**, which carries no precondition. The wraparound
        conditions — `B < q/(41·c)` and the two that keep the exact
        identities from wrapping — are stated over `B^(e)` alone, the ℓ2
        bound of the *other* leg, because they exist to stop an integer
        identity proved mod `q` from wrapping and eq. 52 is not one. So
        there is no condition here for an ℓ∞ bound to be converted for.

        Converting anyway does not work numerically either, which is worth
        recording because `√n` is the obvious move: `‖·‖₂ ≤ √n·‖·‖_∞` over
        the 256-coordinate projection costs a factor `16·14 = 224` against
        the ℓ2 leg's `2·√(256/26)·t`, some `21.8×` — while the suite's own
        parameter point clears Lemma 2.9's precondition by only `9.7×`. It
        would refuse honest configurations to check a condition they do not
        owe.

        An exact statement therefore has to be carried by an ℓ2 leg. That is
        a composition error, and raising here is what makes it one rather
        than a proof of a bound nobody established."""
        raise ValueError(
            "masking: this is Fig. 10's ℓ∞ leg, which proves eq. 52 via "
            "Lemma 2.7 — Theorem 5.3 states the wraparound conditions over "
            "the ℓ2 leg's bound alone, so an exact statement must ride a "
            "leg carrying `L2Bound`"
        )


class BimodalMasking:
    """The parameter point Fig. 9's projection masks and rejects against.

    `Masking` above is the masking of a *witness*: two Gaussians, one per
    ABDLOP half, gated by Rej1. This is the masking of a *projection* — a
    single Gaussian `y ~ D_{s3}^{256/d}` over the 256 integers `R⃗s` shrinks
    the witness to, gated by the bimodal Rej0 of Fig. 2. The two are not
    the same object and must not be one: they mask different vectors, at
    different standard deviations, under different rejection algorithms.

    **Why bimodal here and not there.** The projection is masked as
    `z = b·R⃗s + y` for a secret sign `b ∈ {−1, 1}`, which makes `z`'s
    distribution the average of two Gaussians centred at `±R⃗s`. Rej0
    (Lemma 2.14-3) accepts that average against `M = exp(1/(2γ²))` rather
    than Rej1's `exp(14/γ + 1/(2γ²))`, so the same repetition rate is
    reached at a much smaller `s3` — and `s3` is what the revealed `z`'s bit
    length, and therefore the proof size, is set by. The price is proving
    `b` really is a sign, which is why Fig. 9 hands the layer above a
    quadratic relation and `d − 1` evaluations it would not otherwise need.

    Rej1 cannot be swapped in as a "safer default": it is stated for a
    unimodal `z = v + y` and says nothing about this distribution.

    The witness masking is *not* absorbed here even though both are
    rejection loops, because a rejected attempt at this layer redraws
    `(b, y)` and re-derives `R`, while a rejected attempt inside the proof
    below redraws that proof's own `(y1, y2)`. One loop around both would
    throw away work the inner loop had already accepted.

    Parameters arrive derived, as everywhere in this package: `mask_std` is
    the paper's `s3 = γ·√337·β` (the `√337` is Lemma 2.8's projection
    growth, the `γ` the usual `s = γ·T`), `rep0` its Lemma 2.14-3 rate, and
    `bound` the norm the verifier gates the reveal in. Deriving them from a
    witness bound is the consumer's, since `β` is a property of the
    statement rather than of this seam.

    `bound` is a parameter and not a second class because it is the *only*
    thing that differs between Fig. 10's two legs: both draw `256/d`
    elements at a deviation derived the same way and both reject through the
    same Rej0, and only the verifier's acceptance norm splits — `LinfBound`
    for the approximate ℓ∞ leg, `L2Bound` for the exact ℓ2 one.

    `attempts` is derived here from `rep0` and `fail_prob`, for this point
    used alone. A composition that draws from it more often re-resolves it
    through `for_attempts`.
    """

    def __init__(
        self,
        ring: HostSplitRing,
        mask_std: float,
        rep0: float,
        projection: int = PROJECTION,
        bound: L2Bound | LinfBound = L2Bound(),
        fail_prob: float = 2.0**-128,
    ) -> None:
        if projection < 1:
            raise ValueError(
                f"masking: projection must be positive, got {projection!r}"
            )
        if projection % ring.d:
            raise ValueError(
                f"masking: the mask is a stack of ring elements, so the "
                f"projection dimension must be a multiple of the ring degree "
                f"{ring.d}; got {projection}"
            )
        if mask_std <= 0.0:
            raise ValueError(f"masking: mask_std must be positive, got {mask_std!r}")
        if rep0 <= 1.0:
            raise ValueError(f"masking: rep0 must exceed 1, got {rep0!r}")
        if not 0.0 < fail_prob < 1.0:
            raise ValueError(f"masking: fail_prob must be in (0, 1), got {fail_prob!r}")
        self.ring = ring
        self.projection = projection
        self.mask_cols = projection // ring.d
        self.mask_std = mask_std
        self.rep0 = rep0
        self.bound = bound
        self.fail_prob = fail_prob
        self.attempts = attempt_budget(fail_prob, 1.0 / rep0)
        if self.attempts > _MAX_ATTEMPTS:
            raise ValueError(
                f"masking: rep0 = {rep0!r} implies an attempt budget of "
                f"{self.attempts} — a repetition rate this far from its "
                f"Lemma 2.14-3 value is a parameter bug"
            )
        self._norm, self._limit = bound.resolve(projection, mask_std)
        self._draw = sampler.sampler_for(mask_std, self.attempts * projection)
        self._centers = np.zeros((self.mask_cols, ring.d), dtype=np.float64)

    def for_attempts(self, attempts: int) -> BimodalMasking:
        """This same parameter point, re-resolved for a loop that may run
        `attempts` times.

        A leg composed with others is redrawn whenever *any* leg rejects,
        so it is drawn from more often than its own rate implies, and
        `sampler_for`'s `sample_count` is contractually "the number of
        scalar Gaussian draws the caller's whole protocol run makes". Only
        the composition knows that number, and only after it has seen every
        leg, so a leg cannot be *built* with it. That circle is why this is a
        method and not a constructor argument: the caller states a parameter
        point and `range.ApproximateRange` sizes it for the loop it is about
        to run.

        **The count is currently inert, and this is still not optional.**
        `sampler_for` picks its tier by `sample_count` only through a
        `distance(σ) · count < 2^-64` gate, and `distance` falls as ~1/σ²,
        so the middle tier is unreachable below σ ≈ 10^11 — at every
        parameter point this package can reach, σ alone decides. Nothing
        observable therefore separates a re-resolved sampler from a stale
        one today. It is honoured anyway because under-counting is the
        direction `sampler_for` says is unsafe, and a tightened distance
        bound upstream would make it bite silently.

        Returns `self` when the number is already right, which is every
        single-leg composition — Fig. 9 included."""
        if attempts < self.attempts:
            raise ValueError(
                f"masking: {attempts} attempts is below the {self.attempts} "
                f"this rep0 = {self.rep0!r} and fail_prob = {self.fail_prob!r} "
                f"imply — a loop that gives up sooner never reaches the "
                f"failure probability this point was built for"
            )
        if attempts > _MAX_ATTEMPTS:
            # The same gate the constructor applies to its own derived
            # budget, because this is the other way one is set and the limit
            # is a property of the class rather than of one code path.
            # Composition is what makes it reachable: legs that each clear
            # the constructor can multiply past it, since the joint budget
            # is `∏ 1/rep0_i` and three legs at rep0 = 100 need ~8.9e7
            # attempts while each alone needs ~8.8e3.
            raise ValueError(
                f"masking: a composition of this point needs {attempts} "
                f"attempts — repetition rates this far from their Lemma "
                f"2.14-3 values are a parameter bug, and a loop this long is "
                f"a hang rather than a rejection budget"
            )
        if attempts == self.attempts:
            return self
        resized = copy.copy(self)
        resized.attempts = attempts
        resized._draw = sampler.sampler_for(self.mask_std, attempts * self.projection)
        return resized

    def draw(self, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        """One attempt's `(b, y)`: a sign and a `(256/d, d)` Gaussian mask.

        Both are private coins off the caller's generator, never off the
        transcript — the sign especially, since a transcript-derived `b`
        would be public and the bimodal trick buys nothing.

        Drawn together because they are one attempt: a loop that redrew the
        mask while holding the sign would be sampling neither Rej0's
        distribution nor Rej1's."""
        sign = 1 if rng.integers(0, 2) else -1
        return sign, self._draw(rng, self._centers, self.mask_std)

    def accepts(self, rng: np.random.Generator, z: np.ndarray, v: np.ndarray) -> bool:
        """Rej0 (Fig. 2) on one attempt's revealed projection.

        `v` is the *signed* centre `b·R⃗s`, not `R⃗s` — Lemma 2.14-3 is
        stated for `z = y + (−1)^β v` and the gate reads `⟨z, v⟩` at that
        same `v`. `cosh` is even, so the two spellings agree here by luck
        rather than by contract; passing the centre the response was built
        from is what stays true when a later leg is not symmetric."""
        return _rej0(_coin(rng), z, v, self.mask_std, self.rep0)

    def proven_norm(self) -> int:
        """The bound a verifying proof establishes about the *projected
        vector*, in whichever norm `bound` names — Lemma 2.9's conclusion,
        not the gate on `⃗z` that `within_bounds` applies."""
        return self.bound.proven_norm(self.projection, self.mask_std)

    def within_bounds(self, z: np.ndarray) -> bool:
        """The verifier's gate on the revealed projection, in whichever norm
        `bound` names — and the only place the range statement's slack is
        actually enforced.

        Flattened here because the two norms disagree about rank:
        `norms.l2_squared` flattens any input itself and says so, while
        `norms.linf` walks its argument and raises on the `(256/d, d)` stack
        a reveal actually is. Doing it once, on the way in, keeps that from
        being a property of which gate a leg happens to carry."""
        return self._norm(np.asarray(z).reshape(-1)) <= self._limit


def _rej0(coin: float, z: np.ndarray, v: np.ndarray, std: float, rep: float) -> bool:
    """Rej0 (Fig. 2): accept iff
    `coin < 1/(M·exp(−‖v‖²/(2s²))·cosh(⟨z,v⟩/s²))`, in log space.

    Taken in logs for the reason `_rej1` is, and then some: `cosh` of a
    real-parameter argument overflows to `inf` long before the acceptance
    itself underflows, which would reject every attempt rather than the
    intended `1 − 1/M` share of them. `log cosh x = |x| + log1p(e^{−2|x|})
    − log 2` is the overflow-free form.

    """
    zv, vv = _exact_dots(z, v)
    scaled = abs(zv) / std**2
    log_cosh = scaled + math.log1p(math.exp(-2.0 * scaled)) - math.log(2.0)
    threshold = vv / (2.0 * std**2) - math.log(rep) - log_cosh
    return math.log(coin) < threshold
