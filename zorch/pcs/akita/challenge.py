# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ring challenges as a policy the consumer picks, not a set this scheme fixes.

A module-lattice argument multiplies its witness by a challenge from a subset
of `Z[X]/(X^d+1)`, and *which* subset is a soundness decision the surrounding
protocol owns: it fixes the challenge-set size (knowledge error), the operator
norm (how far the response's norm grows), and — in schemes that extract by
dividing — whether differences are invertible. Those three pull against each
other, so no single set serves every consumer, and a scheme that hardcodes one
has quietly answered its consumer's soundness question.

So the seam is a `ChallengePolicy`: a byte count and a parser over that many
bytes. The pair is the contract, because a policy whose parser reads a
different number of bytes than its companion quotes desynchronises the
transcript — the same pairing discipline lattice-frx's `*_bytes_needed`
sampler family and `zorch/lnp/challenge.py` both keep, named here as a type so
a consumer can pass its own.

`FixedWeightTernary` is the sparse/short instance: exactly `weight` nonzero
coefficients in `{-1, +1}`, so `‖c‖₁ = weight` bounds the norm growth of `c·s`
directly and the challenge multiply is `weight` sign-flipped shifts rather
than a full ring product. `zorch/lnp/challenge.py`'s `ChallengeParams` — the
σ₋₁-invariant LNP set — satisfies the same protocol without knowing about it,
which is what makes this a seam rather than a wrapper around one sampler.

Host by construction, like the samplers underneath: the draw is a rejection
walk over a byte stream, and where those bytes come from is the consumer's
transcript, not this module's business.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from lattice_frx.sampler import fixed_weight_ternary, fixed_weight_ternary_bytes_needed

from zorch.byte_transcript import ByteTranscript


@runtime_checkable
class ChallengePolicy(Protocol):
    """A challenge set, as the two things a Fiat-Shamir caller needs of one."""

    @property
    def bytes_needed(self) -> int:
        """Exactly how many transcript bytes `from_bytes` consumes."""
        ...

    def from_bytes(self, data: bytes | bytearray | np.ndarray) -> np.ndarray:
        """One challenge as a length-`d` signed coefficient vector.

        Raw host integers ready for a ring's `from_signed`, rather than a ring
        element: the policy does not know which ring the consumer commits in,
        and the same challenge is often needed unreduced (a response's norm is
        a statement about `c·s` over ℤ, which a residue has none of).
        """
        ...


@runtime_checkable
class BoundedChallengePolicy(ChallengePolicy, Protocol):
    """A challenge set that also bounds every draw's ℓ1 norm.

    What a fold needs before any challenge is drawn: each folded term grows the
    response by at most `max_l1` times the witness bound, and whether the ring
    modulus lifts that exactly is a property of the parameter point, so it is
    settled where the point is set rather than by the first unlucky transcript.
    """

    @property
    def max_l1(self) -> int:
        """An upper bound on `‖c‖₁` for every challenge `from_bytes` returns."""
        ...


@dataclass(frozen=True)
class FixedWeightTernary:
    """`weight` nonzero coefficients in `{-1, +1}`, the rest zero.

    The set size is `C(d, weight)·2^weight`, and both knobs move it, so a
    consumer targeting a knowledge error picks `weight` against its own
    degree. `fail_prob` prices the sampler's rejection budget — the stream
    length is fixed ahead of the draw, so an unlucky one fails loudly instead
    of consuming more bytes and desynchronising the transcript.
    """

    degree: int
    weight: int
    fail_prob: float = 2.0**-128

    def __post_init__(self) -> None:
        # Validated at construction rather than at the first draw: a policy is
        # built once from configuration and used every round, so a bad weight
        # should fail where the parameter point is written down.
        fixed_weight_ternary_bytes_needed(self.weight, self.degree, self.fail_prob)

    @property
    def bytes_needed(self) -> int:
        return fixed_weight_ternary_bytes_needed(
            self.weight, self.degree, self.fail_prob
        )

    @property
    def max_l1(self) -> int:
        """Exactly `weight`: every nonzero coefficient is ±1."""
        return self.weight

    def from_bytes(self, data: bytes | bytearray | np.ndarray) -> np.ndarray:
        return fixed_weight_ternary(data, self.weight, self.degree, self.fail_prob)


def squeeze_challenge(
    transcript: ByteTranscript, label: bytes, policy: ChallengePolicy
) -> tuple[ByteTranscript, np.ndarray]:
    """Absorb `label`, then draw one challenge under `policy`.

    The count and the parse travel together here so no call site can pair a
    byte count with a different policy's parser — the failure that pairing
    prevents is silent, since a prover and verifier that make the same mistake
    still agree with each other and only disagree with every other consumer.

    `ByteTranscript` is functional, so the advanced transcript comes back and
    the caller's value is untouched.
    """
    advanced = transcript.observe_label(label)
    advanced, raw = advanced.sample_scalar(policy.bytes_needed)
    return advanced, policy.from_bytes(raw)


if TYPE_CHECKING:
    _: type[ChallengePolicy] = FixedWeightTernary
    _bounded: type[BoundedChallengePolicy] = FixedWeightTernary
