# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The Akita opening's shared half: its claim and proof, the packed layout both
roles derive from them, and the transcript order both replay.

**One fold, stated per block.** An evaluation of a committed polynomial is a
public linear form in its coefficients, and the block layout makes it factor.
Read a coefficient index as super-block `s`, position `p` inside it, and
coefficient `ℓ` inside one ring element; its eq weight is then `B_s·Q_p·I_ℓ`.
The prover sends each super-block's ring partial `E_s = Σ_p Q_p·F_{s,p}` over
the field, so the claimed value is `Σ_s B_s·⟨I, E_s⟩`. It then folds the digit
witness over super-blocks with sparse ring challenges, `z_p = Σ_s c_s·s_{s,p}`,
and the verifier checks the three relations that make the partials honest —
upstream Akita's fold relations ("Semantic relations in an Akita fold",
eqs. 12a and 13):

- `‖z‖∞ ≤ β·Σ_s ‖c_s‖₁`, the norm an honest fold of `β`-short digits reaches;
- `A·z_p = Σ_s c_s·t_{s,p}`, against inner images the outer payload binds;
- `Σ_p Q_p·Σ_h 2^{w·h}·z_{p,h} = Σ_s c_s·E_s` over the field: the challenge
  folds super-blocks while `Q` contracts positions, so the two commute.

Nothing recurses. The images travel in the proof and are checked against the
payload directly, where upstream commits them into the next fold's witness, so
the proof grows with the block count: a verified opening, not a succinct one.

**Positions are the opening's choice, not the commitment's.** Every block has
its own inner image, so any power-of-two run of consecutive blocks is a valid
super-block. The layout takes the balanced split, `2^⌊log₂(blocks)/2⌋`
positions, which evens the partials (one ring element per super-block per
claim) against the response (one per position per digit).

**Packed claims.** Claims at one point share every weight, so they share one
response: the fold runs over each member's super-blocks with independent
challenges. Claims at different points cannot, since `Q` differs, so the layout
groups claims by point, in order of first appearance, and each group carries
its own partials and response under one set of images.

**Transcript order**, replayed by both roles through the functions below:

1. the label `akita/open`, then the payload;
2. per claim, its message index, then its point;
3. the claimed values, in claim order;
4. the inner images;
5. each group's partials, in group order;
6. one fold challenge per (group, member, super-block), each under the label
   `akita/fold`;
7. each group's response.

The responses go last so a protocol continuing on this transcript is bound to
the whole proof.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from lattice_frx.ring import Coeff, Eval, RnsRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import centered_lift
from zorch.pcs.akita.challenge import ChallengePolicy, squeeze_challenge
from zorch.pcs.akita.commit import AkitaCommitment
from zorch.pcs.akita.config import AkitaConfig
from zorch.pcs.stage import OpeningClaim
from zorch.poly.eq import expand_eq_to_hypercube

_OPEN_LABEL = b"akita/open"
_FOLD_LABEL = b"akita/fold"

# One signed challenge vector per super-block, per member of a group.
GroupChallenges = tuple[tuple[np.ndarray, ...], ...]
FoldChallenges = tuple[GroupChallenges, ...]


@dataclass(frozen=True)
class AkitaOpeningClaim(OpeningClaim[AkitaCommitment]):
    """Committed message `messages[i]` evaluates at `points[i]`.

    A subset of the batch rather than all of it: the payload binds every
    message, but a consumer opens the few its protocol reduced to, and each
    unopened one would cost partials nobody reads.
    """

    messages: tuple[int, ...]


@dataclass(frozen=True)
class AkitaOpeningProof:
    """The inner images `[blocks, inner_rows]`, then per group its field
    partials `[members, super_blocks, d]` and its response
    `[positions, inner_cols]`."""

    images: Coeff
    partials: tuple[Array, ...]
    responses: tuple[Coeff, ...]


@dataclass(frozen=True)
class ClaimGroup:
    """The claims opened at one point, and how that point splits over the
    block layout."""

    point: Array
    members: tuple[int, ...]  # indices into the claim, ascending
    messages: tuple[int, ...]
    positions: int
    super_blocks: int

    def weights(self, degree: int) -> tuple[Array, Array, Array]:
        """The eq tables `(I, Q, B)` over coefficients `[degree]`, positions
        and super-blocks.

        MSB-first like `eval_mle`, so the point's leading coordinates address
        super-blocks and its trailing ones a coefficient inside a ring element.
        A message shorter than a ring element leaves the tail of `I` zero — its
        padding, which the commitment bound as zero digits.
        """
        point = self.point
        one = fnp.ones((), point.dtype)
        split = self.super_blocks.bit_length() - 1
        block_vars = split + self.positions.bit_length() - 1
        inner = expand_eq_to_hypercube(point[block_vars:], one)
        if inner.shape[0] < degree:
            padding = fnp.zeros((degree - inner.shape[0],), point.dtype)
            inner = fnp.concatenate([inner, padding])
        return (
            inner,
            expand_eq_to_hypercube(point[split:block_vars], one),
            expand_eq_to_hypercube(point[:split], one),
        )

    def view(self, config: AkitaConfig, message: int, table: Array) -> Array:
        """One member's rows of a per-block table `[blocks, ...]`, as
        `[super_blocks, positions, ...]`; position is the fast index inside a
        super-block."""
        return message_rows(config, message, table).reshape(
            self.super_blocks, self.positions, *table.shape[1:]
        )


def message_rows(config: AkitaConfig, message: int, table: Array) -> Array:
    """One message's rows of a per-block table `[blocks, ...]`.

    A slice, not a gather: the commitment pads each message to whole blocks on
    its own, so a message's blocks are contiguous.
    """
    offset = sum(config.blocks_per_message[:message])
    return table[offset : offset + config.blocks_per_message[message]]


def claimed_value(inner: Array, block: Array, partial: Array) -> Array:
    """`Σ_s B_s·⟨I, E_s⟩`, the value one member's partials `[super_blocks, d]`
    claim — what the prover sends and the verifier checks it against."""
    return (block[:, None] * inner[None, :] * partial).sum()


def packed_layout(
    config: AkitaConfig, claim: AkitaOpeningClaim
) -> tuple[ClaimGroup, ...]:
    """The claim grouped by point, each point split over the block layout.

    Refuses what no split can serve: a message whose length is not a power of
    two has no multilinear point, a point of the wrong length evaluates a
    different polynomial, and points over two fields cannot both be the field
    the batch was committed in.
    """
    if not claim.messages:
        raise ValueError("packed_layout: an opening needs at least one claim")
    if len(claim.points) != len(claim.messages):
        raise ValueError(
            f"packed_layout: {len(claim.points)} points for "
            f"{len(claim.messages)} messages"
        )
    dtype = claim.points[0].dtype
    by_point: dict[bytes, list[int]] = {}
    for index, (message, point) in enumerate(zip(claim.messages, claim.points)):
        if not 0 <= message < len(config.message_lens):
            raise ValueError(
                f"packed_layout: message {message} outside "
                f"[0, {len(config.message_lens)})"
            )
        length = config.message_lens[message]
        if length & (length - 1):
            raise ValueError(
                f"packed_layout: message {message} has {length} coefficients, "
                "not a power of two, so no multilinear point addresses it"
            )
        variables = length.bit_length() - 1
        if point.shape != (variables,):
            raise ValueError(
                f"packed_layout: message {message} takes a point of {variables} "
                f"coordinates, got shape {tuple(point.shape)}"
            )
        if point.dtype != dtype:
            raise ValueError(
                f"packed_layout: points mix {dtype} and {point.dtype}; one batch "
                "is committed in one field"
            )
        by_point.setdefault(field_bytes(point), []).append(index)

    groups = []
    for members in by_point.values():
        messages = tuple(claim.messages[index] for index in members)
        # Equal points have equal lengths, so every member has these blocks.
        blocks = config.blocks_per_message[messages[0]]
        positions = 1 << ((blocks.bit_length() - 1) // 2)
        groups.append(
            ClaimGroup(
                claim.points[members[0]],
                tuple(members),
                messages,
                positions,
                blocks // positions,
            )
        )
    return tuple(groups)


def ring_bytes(element: Coeff | Eval) -> bytes:
    """Every limb's residues as little-endian u64, limb after limb.

    Exact because each limb's dtype converts to its residue canonically, the
    conversion `centered_lift` relies on. The domain is not tagged, so both
    roles absorb an element in the domain the wire fixes for it.
    """
    return b"".join(
        np.asarray(limb).astype(np.uint64).astype("<u8").tobytes()
        for limb in element.limbs
    )


def field_bytes(values: Array) -> bytes:
    """Canonical residues at the field's own byte width, little-endian — wide
    enough for whichever field the consumer commits."""
    modulus = int(zk_dtypes.pfinfo(values.dtype).modulus)
    width = (modulus.bit_length() + 7) // 8
    return b"".join(
        int(value).to_bytes(width, "little")
        for value in np.asarray(values).astype(object).reshape(-1)
    )


def observe_statement(
    transcript: ByteTranscript,
    claim: AkitaOpeningClaim,
    values: Array,
    images: Coeff,
) -> ByteTranscript:
    """Steps 1-4 of the transcript order."""
    transcript = transcript.observe_label(_OPEN_LABEL)
    transcript = transcript.observe_bytes(ring_bytes(claim.commitment))
    for message, point in zip(claim.messages, claim.points):
        transcript = transcript.observe_bytes(message.to_bytes(8, "little"))
        transcript = transcript.observe_bytes(field_bytes(point))
    transcript = transcript.observe_bytes(field_bytes(values))
    return transcript.observe_bytes(ring_bytes(images))


def observe_partials(
    transcript: ByteTranscript, partials: Sequence[Array]
) -> ByteTranscript:
    """Step 5 of the transcript order."""
    for partial in partials:
        transcript = transcript.observe_bytes(field_bytes(partial))
    return transcript


def draw_fold_challenges(
    transcript: ByteTranscript,
    layout: Sequence[ClaimGroup],
    policy: ChallengePolicy,
    degree: int,
) -> tuple[ByteTranscript, FoldChallenges]:
    """Step 6: one challenge per member super-block, in group, then member,
    then super-block order."""
    groups = []
    for group in layout:
        members = []
        for _ in group.messages:
            drawn = []
            for _ in range(group.super_blocks):
                transcript, challenge = squeeze_challenge(
                    transcript, _FOLD_LABEL, policy
                )
                if challenge.shape != (degree,):
                    raise ValueError(
                        f"draw_fold_challenges: the policy draws "
                        f"{challenge.shape[0]} coefficients, the ring has "
                        f"degree {degree}"
                    )
                drawn.append(challenge)
            members.append(tuple(drawn))
        groups.append(tuple(members))
    return transcript, tuple(groups)


def observe_responses(
    transcript: ByteTranscript, responses: Sequence[Coeff]
) -> ByteTranscript:
    """Step 7 of the transcript order."""
    for response in responses:
        transcript = transcript.observe_bytes(ring_bytes(response))
    return transcript


def fold_blocks(
    ring: RnsRing,
    config: AkitaConfig,
    table: Eval,
    group: ClaimGroup,
    challenges: GroupChallenges,
) -> Eval:
    """`Σ_member Σ_s c_s·x_{s,p}` for every position `p`, over a per-block
    table `[blocks, ...]`: the witness on the prover's side, the images on the
    verifier's — the two sides of `A·z_p = Σ_s c_s·t_{s,p}`.

    One broadcast product per member over its `[super_blocks, positions, ...]`
    view, rather than a product per block.
    """
    trailing = (1,) * (table.limbs[0].ndim - 2)
    folded = []
    for message, drawn in zip(group.messages, challenges):
        weights = ring.ntt(ring.stack([ring.from_signed(c) for c in drawn]))
        limbs = []
        for weight, limb in zip(weights.limbs, table.limbs):
            # `[super_blocks, d]` against the view: over positions and any
            # module axes between them and the coefficients.
            shaped = weight.reshape(weight.shape[0], 1, *trailing, weight.shape[-1])
            limbs.append((shaped * group.view(config, message, limb)).sum(axis=0))
        folded.append(Eval(tuple(limbs)))
    total = folded[0]
    for term in folded[1:]:
        total = ring.add(total, term)
    return total


def digits_to_field(ring: RnsRing, digits: Coeff, log_base: int, dtype: Any) -> Array:
    """Short digit planes `[..., k]` recomposed in the field, `[..., d]`.

    In the field rather than over the integers: the digits are short, but
    their recomposition is not, and no integer past `p` converts to it. Plane
    `h` weighs `2^{w·h}`, the gadget's order, least significant first.
    """
    lead = tuple(digits.limbs[0].shape[:-1])
    lifted = np.array(centered_lift("digits_to_field", ring, digits), dtype=object)
    planes = fnp.asarray(lifted.astype(dtype)).reshape(*lead, ring.d)
    base = fnp.asarray(np.array([1 << log_base], dtype=object).astype(dtype))[0]
    weight = fnp.ones((), dtype)
    total = planes[..., 0, :]
    for plane in range(1, lead[-1]):
        weight = weight * base
        total = total + weight * planes[..., plane, :]
    return total
