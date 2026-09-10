# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Virtual-to-committed reduction: a claim on a product of committed
polynomials, reduced to claims on the factors an opening can serve.

**What a virtual polynomial is here.** `V(x_1, …, x_m, y) = Π_i P_i(x_i, y)`:
each committed factor `P_i` owns a slice `x_i` of the leading variables and all
of them share the trailing ones, `y`. Nothing of `V` is committed — its table is
the product of theirs, which is what makes committing the factors the cheap
side when the slices are wide: a selector over a wide index is exactly this
product over narrow chunks of it.

**Why a sumcheck.** The shared `y` couples the factors, so at a point
`(r_1, …, r_m, r_y)` the extension is `V(r) = Σ_y eq(r_y, y)·Π_i P_i(r_i, y)`,
not a product of evaluations. A sumcheck over `y` — degree `m + 1`, bound
MSB-first like the dense rounds it reuses — binds `y` to challenges `ρ`; the
prover then names each factor's value `a_i = P_i(r_i, ρ)`, and the verifier
checks `eq(r_y, ρ)·Π_i a_i` against the last round's claim. What remains is one
committed claim per factor, at `(r_i ‖ ρ)`: distinct points, which the packed
opening groups apart. With no shared variables no round runs, and the check is
the product of the factor values itself.

**Transcript order**: the label `akita/virtual`, the payload, per factor its
message index and slice width, the point, the value; then per round its
polynomial followed by one field challenge; then the factor values. The
opening continues on the same transcript, so its challenges depend on the
reduction's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array

from zorch.byte_transcript import ByteTranscript
from zorch.pcs.akita.commit import AkitaCommitment
from zorch.pcs.akita.config import AkitaConfig
from zorch.pcs.akita.wire import AkitaOpeningClaim, field_bytes, ring_bytes

_VIRTUAL_LABEL = b"akita/virtual"


@dataclass(frozen=True)
class VirtualClaim:
    """`V(point) = value` for `V = Π_i P_i(x_i, y)` over the committed
    messages `messages`, where factor `i` owns `widths[i]` leading variables.

    The point is MSB-first and laid out as the variables are:
    `x_1 ‖ … ‖ x_m ‖ y`.
    """

    commitment: AkitaCommitment
    messages: tuple[int, ...]
    widths: tuple[int, ...]
    point: Array
    value: Array


@dataclass(frozen=True)
class VirtualProof:
    """The round polynomials `[shared_vars, m + 2]`, each sampled on
    `{0, …, m + 1}`, and the factor values `[m]` at the bound point."""

    rounds: Array
    factors: Array


def shared_vars(config: AkitaConfig, claim: VirtualClaim) -> int:
    """How many trailing variables the factors share, with every factor's
    length checked against its slice.

    A factor whose length disagrees with `2^(width + shared)` is a claim about
    some other product, so it is refused rather than padded.
    """
    if not claim.messages:
        raise ValueError("shared_vars: a virtual product needs at least one factor")
    if len(claim.widths) != len(claim.messages):
        raise ValueError(
            f"shared_vars: {len(claim.widths)} widths for "
            f"{len(claim.messages)} factors"
        )
    if claim.point.ndim != 1 or claim.value.shape != ():
        raise ValueError("shared_vars: the point is a vector and the value a scalar")
    if claim.value.dtype != claim.point.dtype:
        raise ValueError(
            f"shared_vars: value in {claim.value.dtype}, point in {claim.point.dtype}"
        )
    shared = claim.point.shape[0] - sum(claim.widths)
    if shared < 0 or min(claim.widths) < 0:
        raise ValueError(
            f"shared_vars: widths {claim.widths} do not fit a point of "
            f"{claim.point.shape[0]} coordinates"
        )
    for message, width in zip(claim.messages, claim.widths):
        if not 0 <= message < len(config.message_lens):
            raise ValueError(
                f"shared_vars: message {message} outside "
                f"[0, {len(config.message_lens)})"
            )
        want = 1 << (width + shared)
        if config.message_lens[message] != want:
            raise ValueError(
                f"shared_vars: factor message {message} has "
                f"{config.message_lens[message]} coefficients, but a slice of "
                f"{width} variables over {shared} shared ones needs {want}"
            )
    return shared


def factor_slices(claim: VirtualClaim) -> tuple[tuple[int, int], ...]:
    """Each factor's `[start, stop)` slice of the point; the shared variables
    follow the last one."""
    slices = []
    start = 0
    for width in claim.widths:
        slices.append((start, start + width))
        start += width
    return tuple(slices)


def reduced_claim(
    claim: VirtualClaim, challenges: Sequence[Array]
) -> AkitaOpeningClaim:
    """One committed claim per factor, at its own slice followed by `ρ`."""
    bound = (
        fnp.stack(list(challenges))
        if challenges
        else fnp.zeros((0,), claim.point.dtype)
    )
    points = tuple(
        fnp.concatenate([claim.point[start:stop], bound])
        for start, stop in factor_slices(claim)
    )
    return AkitaOpeningClaim(claim.commitment, points, claim.messages)


def observe_virtual_claim(
    transcript: ByteTranscript, claim: VirtualClaim
) -> ByteTranscript:
    """Everything before the first round, in the transcript order above."""
    transcript = transcript.observe_label(_VIRTUAL_LABEL)
    transcript = transcript.observe_bytes(ring_bytes(claim.commitment))
    for message, width in zip(claim.messages, claim.widths):
        transcript = transcript.observe_bytes(
            message.to_bytes(8, "little") + width.to_bytes(8, "little")
        )
    transcript = transcript.observe_bytes(field_bytes(claim.point))
    return transcript.observe_bytes(field_bytes(claim.value))


def observe_round(transcript: ByteTranscript, round_poly: Array) -> ByteTranscript:
    return transcript.observe_bytes(field_bytes(round_poly))


def observe_factors(transcript: ByteTranscript, factors: Array) -> ByteTranscript:
    return transcript.observe_bytes(field_bytes(factors))


def squeeze_field(
    transcript: ByteTranscript, dtype: Any
) -> tuple[ByteTranscript, Array]:
    """One uniform element of the prime field `dtype`, by rejection over its
    bit width.

    A draw at or past `p` is discarded and the next one taken, so no residue is
    favoured and nothing is reduced by hand. Both roles replay the same draws,
    so a rejection costs bytes, never agreement.

    Its own loop rather than lattice-frx's `uniform_from_bytes`, the draw
    `zorch/lnp` uses: that sampler caps the modulus at 2^64, and this field is
    the consumer's — a pairing-friendly scalar field is 254 bits.
    """
    modulus = int(zk_dtypes.pfinfo(dtype).modulus)
    bits = modulus.bit_length()
    width = (bits + 7) // 8
    while True:
        transcript, raw = transcript.sample_scalar(width)
        candidate = int.from_bytes(raw, "little") & ((1 << bits) - 1)
        if candidate < modulus:
            element = np.array([candidate], dtype=object).astype(dtype)
            return transcript, fnp.asarray(element)[0]
