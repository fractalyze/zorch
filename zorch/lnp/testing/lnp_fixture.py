# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The LNP test parameter point, and the fixtures every protocol suite
builds on it.

One definition, because the numbers below are *derived* — from Lemma
2.14-1, from the Figure-3 challenge point, from a rejection budget — and a
suite that copies the values without the derivations inherits four magic
constants. The second protocol suite is where that stops being a copy and
starts being a divergence, so the point lives here.
"""

from __future__ import annotations

import numpy as np
from hash_frx.sha256 import HostSha256
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteHashTranscript, ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp.challenge import ChallengeParams
from zorch.lnp.masking import BimodalMasking, Masking

# One ~32-bit split prime (≡ 5 mod 8; `find_nearest_split_primes(32, 1)`)
# and a small degree keep the schoolbook ring affordable; the challenge
# keeps the paper's (κ, η, k) since the gate is degree-agnostic.
SPLIT_Q = (4294967197,)
D = 64
KAPPA, ETA, K = 2, 59, 32

# The whole candidate budget is squeezed on every challenge derivation,
# decided or not, so it is the dominant cost of a test proof. 2^-40 keeps a
# margin far past anything a seeded suite can reach (the measured gate
# rejection is ~1%, so the first block almost always decides) at a third of
# the hashing; production keeps the library's 2^-128 default.
FAIL_PROB = 2.0**-40
CHALLENGE = ChallengeParams(d=D, kappa=KAPPA, eta=ETA, k=K, fail_prob=FAIL_PROB)

# Lemma 2.14-1 at γ = 14: M = exp(14/γ + 1/(2γ²)) ≈ e ≈ 2.72 per response,
# s_i = γ·T_i with T_1 = η·α (α = ‖s1‖ ≤ √(m1·d) for ternary s1) and
# T_2 = η·ν·√(m2·d) at ν = 1. M1 below is the s1 column count both suites
# share; a suite with a different shape re-derives STD itself.
GAMMA = 14.0
M1 = 2
T = ETA * float(np.sqrt(M1 * D))
STD = GAMMA * T
REP = float(np.exp(14.0 / GAMMA + 1.0 / (2.0 * GAMMA**2)))

# The masking/rejection parameters every protocol at this point masks
# against. One `Masking` per scheme, shared by the protocols built over it —
# which is the object's own reason for existing (see `masking.py`).
MASKING_PARAMS: dict[str, object] = dict(
    s1_std=STD, s2_std=STD, rep1=REP, rep2=REP, challenge=CHALLENGE
)


def masking(scheme: AbdlopCommitment, **overrides: object) -> Masking:
    """The test masking point over `scheme`, with one kwarg moved per call
    so a test that varies `fail_prob` states only that."""
    return Masking(scheme, **(MASKING_PARAMS | overrides))  # type: ignore[arg-type]


def ring() -> HostSplitRing:
    return HostSplitRing(SPLIT_Q, D)


def transcript(domain: bytes, tag: bytes = b"") -> ByteTranscript:
    """A fresh transcript in `domain`, with `tag` absorbed — the per-test
    separator that makes two proofs of one statement distinguishable."""
    return ByteHashTranscript.new(domain, HostSha256()).observe_bytes(tag)


def constant(ring: HostSplitRing, value: np.ndarray) -> np.ndarray:
    """The one-element stack holding `value` in its constant coefficient and
    nothing else.

    Written through the array layout the way `GarbageMasking.sample` zeroes
    that slot, and for the same reason: `constant_coeff` reads it, and the
    module convention has no constructor that writes it. Here rather than in
    a suite because *which slot* `constant_coeff` reads is one convention,
    and two suites spelling it is how a ring-layout change breaks one and
    not the other."""
    out = ring.zeros(1)
    out[0, :, 0] = value
    return out


def vanishing_constant(ring: HostSplitRing, value: np.ndarray) -> np.ndarray:
    """The `e0` that makes `F̃(s) = 0` for a function whose value on the
    witness is `value` — while `F(s)` itself stays nonzero, since only the
    constant coefficient is negated and not the whole thing.

    The construction every suite needs to state an *evaluation* that is true
    on its witness, as opposed to a relation: `quadratic_eval_test` builds
    Fig. 8's `F_j` with it and `range_test` a caller's `F_i`.

    Takes the already-evaluated value rather than `(e2, e1, s)` so this
    module keeps needing nothing but a ring. It is the shared *parameter
    point*, and every suite depends on it — pulling `quadratic.evaluate` in
    here would put the quadratic protocol behind `opening_test`, which is
    about Fig. 4 and has no quadratic anything."""
    return ring.neg(constant(ring, ring.constant_coeff(value)[0]))


def bump(stack: np.ndarray, *index: int) -> np.ndarray:
    """A copy with one residue moved by one — the tamper idiom, so a
    soundness test states only *what* it tampered."""
    out = stack.copy()
    out[index] = (int(out[index]) + 1) % SPLIT_Q[0]
    return out


# The Fig.-9 masking point, derived at the same γ. Lemma 2.14-3's bimodal
# rate is M = exp(1/(2γ²)) ≈ 1.003 — far under Rej1's ≈ 2.72, which is the
# whole reason Fig. 9 pays for a sign and its `d − 1` coefficient proofs.
# The projection dimension is `BimodalMasking`'s own default, which is the
# object that validates it against the ring degree.
REP0 = float(np.exp(1.0 / (2.0 * GAMMA**2)))


def bimodal(
    ring: HostSplitRing, witness_cols: int, **overrides: object
) -> BimodalMasking:
    """The bimodal point over `ring` for a *ternary* `(s1, m)` of
    `witness_cols` ring elements.

    `s3 = γ·√337·β`: the `√337` is Lemma 2.8-2's projection growth at κ = 1,
    and `β = ‖(s1, m)‖₂ ≤ √(witness_cols·d)` is the ternary witness bound
    this package's suites all build. Derived here rather than in each suite
    for the reason `MASKING_PARAMS` is — Fig. 10 needs two of these, one per
    leg, and two spellings of one derivation is the divergence.

    The deviation is the same either way: Fig. 10 derives both legs' `s`
    from an **ℓ2** bound on the projected vector, and only the verifier's
    gate splits. Pass `bound=LinfBound()` for the ℓ∞ leg."""
    beta = float(np.sqrt(witness_cols * ring.d))
    params: dict[str, object] = dict(
        mask_std=GAMMA * float(np.sqrt(337.0)) * beta,
        rep0=REP0,
        fail_prob=FAIL_PROB,
    )
    return BimodalMasking(ring, **(params | overrides))  # type: ignore[arg-type]
