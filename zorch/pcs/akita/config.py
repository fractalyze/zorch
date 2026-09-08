# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The module-lattice PCS parameter point: SIS modulus profile, digit
decomposition, and the module shape the two of them force.

Every number here is a value a consumer chooses, not a constant this package
picked. That is the whole reason the file exists: a commitment scheme whose
ring degree, modulus chain and digit base are literals is one deployment's
scheme wearing a general name, and the catalog of parameter points that a
downstream prover ships is *its* data — read from its own artifacts, handed
in here.

Three objects, because they are chosen against three different pressures:

- `SisProfile` — the ring `Z_q[X]/(X^d+1)` the commitment lives in. Its
  degree and modulus chain are picked against MSIS hardness at a target
  security level.
- `Decomposition` — base-`2^w` balanced digits. Picked against the *field*
  being committed: enough digits to represent every coefficient exactly,
  and a base small enough that the digit norm stays far under `q`.
- `AkitaConfig` — the two together plus the module height, which is where
  the derived shapes live: how many ring elements the decomposed witness
  occupies, and hence how wide the public matrix must be.

Binding strength is *not* checked here, matching the stance of the
commitment algebra below it (`zorch/commit/ajtai.py`): MSIS hardness at a
parameter point is the consumer's analysis, and a gate that pretended to
verify it would be a security claim this package cannot make. What is
checked is exactness — a decomposition that cannot represent the values it
is handed is a bug, not a parameter choice, and `gadget.decompose` already
refuses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from lattice_frx.primes import find_nearest_ntt_primes
from lattice_frx.ring import RnsRing


@dataclass(frozen=True)
class SisProfile:
    """The commitment ring: degree `d` and the RNS chain `q` splits over.

    Constructed from an explicit chain, or from a target bit width through
    `nearest` — the two ways a parameter point arrives. An explicit chain is
    what a consumer replaying a pinned reference needs (the moduli are part
    of what the two sides agree on); `nearest` is what sizing a fresh point
    needs.
    """

    degree: int
    moduli: tuple[int, ...]

    def __post_init__(self) -> None:
        # `RnsRing` enforces the same two predicates, but only once a ring is
        # built. A profile is passed around, serialized and compared long
        # before that, so an unusable one fails where it was written down.
        if self.degree < 2 or self.degree & (self.degree - 1):
            raise ValueError(
                f"SisProfile: degree must be a power of two, got {self.degree}"
            )
        if not self.moduli:
            raise ValueError("SisProfile: the modulus chain must be non-empty")
        for q in self.moduli:
            if (q - 1) % (2 * self.degree):
                raise ValueError(
                    f"SisProfile: {q} is not NTT-friendly at degree {self.degree} "
                    f"(needs q = 1 mod {2 * self.degree})"
                )

    @classmethod
    def nearest(cls, degree: int, limb_bits: float, limb_count: int) -> SisProfile:
        """The `limb_count` NTT-friendly primes nearest `2**limb_bits`.

        Sizing a chain by bit width rather than by listing primes is what
        lets a consumer move its modulus up or down against a security
        estimate without hand-picking primes; the walk itself is
        lattice-frx's, so a chain built here and one built for a ring
        elsewhere are the same numbers.
        """
        return cls(
            degree, tuple(find_nearest_ntt_primes(degree, limb_bits, limb_count))
        )

    @cached_property
    def modulus(self) -> int:
        """`Q = prod(q_i)`, the modulus the balanced lift is taken against."""
        return math.prod(self.moduli)

    @cached_property
    def ring(self) -> RnsRing:
        """The traced ring at this point. Cached because building one
        precomputes per-limb field dtypes and a bit-reversal table, and every
        layer above holds the profile rather than passing the ring down."""
        return RnsRing(self.moduli, self.degree)


@dataclass(frozen=True)
class Decomposition:
    """Balanced base-`2^log_base` digits, `num_digits` of them.

    Balanced rather than unsigned because the digit *norm* is what the
    binding statement bounds, and `[-B/2, B/2)` halves the norm the same
    digit count reaches — see lattice-frx's `gadget`, whose conventions
    (including the asymmetric endpoint) this only names at the scheme level.
    """

    log_base: int
    num_digits: int

    def __post_init__(self) -> None:
        if self.log_base < 1:
            raise ValueError(
                f"Decomposition: log_base must be >= 1, got {self.log_base}"
            )
        if self.num_digits < 1:
            raise ValueError(
                f"Decomposition: num_digits must be >= 1, got {self.num_digits}"
            )

    @property
    def base(self) -> int:
        return 1 << self.log_base

    @property
    def beta_inf(self) -> int:
        """`‖digits‖∞ ≤ B/2`, the opening bound the commitment enforces.

        The bound is `B/2` and not `B/2 - 1` even though `+B/2` is the
        excluded endpoint: `-B/2` is attained, and the norm is of the
        balanced lift.
        """
        return self.base >> 1

    @property
    def representable(self) -> tuple[int, int]:
        """The exact closed interval `t` balanced digits cover.

        Asymmetric — `[-(B/2)·S, (B/2-1)·S]` for `S = (B^t - 1)/(B - 1)` —
        because the digit interval is. A consumer sizing `num_digits` against
        a prime field wants this, not the symmetric `[-B^t/2, B^t/2)` the
        unsigned intuition suggests.
        """
        scale = (self.base**self.num_digits - 1) // (self.base - 1)
        return (-self.beta_inf * scale, (self.beta_inf - 1) * scale)

    @classmethod
    def covering(cls, log_base: int, magnitude: int) -> Decomposition:
        """The fewest base-`2^log_base` digits whose range covers
        `[-magnitude, magnitude]`.

        The bridge from a field to a parameter point: a consumer committing
        balanced lifts modulo `p` passes `p // 2` and gets the digit count
        that represents every one of them, instead of picking a number and
        discovering the overflow on an unlucky witness.
        """
        if magnitude < 0:
            raise ValueError(
                f"Decomposition.covering: magnitude must be >= 0, got {magnitude}"
            )
        if log_base == 1 and magnitude > 0:
            # The balanced digit set at `B = 2` is `{-1, 0}`, so the upper end
            # of `representable` is 0 at every digit count and the search below
            # would climb forever rather than converge.
            raise ValueError(
                "Decomposition.covering: balanced base-2 digits reach no "
                f"positive value, so magnitude {magnitude} is unreachable; "
                "log_base must be >= 2"
            )
        num_digits = 1
        while True:
            candidate = cls(log_base, num_digits)
            low, high = candidate.representable
            if low <= -magnitude and magnitude <= high:
                return candidate
            num_digits += 1


@dataclass(frozen=True)
class AkitaConfig:
    """A parameter point plus the batch it commits, and the module shape both
    force.

    `message_lens` carries one length per committed polynomial rather than a
    single total, because each is padded to a whole number of ring elements
    *individually*. Concatenating first would let one polynomial's tail share
    a ring element with the next one's head, and an opening that has to name
    a polynomial then names a fraction of a ring element.
    """

    profile: SisProfile
    decomposition: Decomposition
    rows: int
    message_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError(f"AkitaConfig: rows must be >= 1, got {self.rows}")
        if not self.message_lens:
            raise ValueError("AkitaConfig: at least one message length is required")
        for length in self.message_lens:
            if length < 1:
                raise ValueError(
                    f"AkitaConfig: message lengths must be >= 1, got {length}"
                )

    @property
    def blocks_per_message(self) -> tuple[int, ...]:
        """Ring elements each message occupies, one entry per message."""
        degree = self.profile.degree
        return tuple(-(-length // degree) for length in self.message_lens)

    @property
    def blocks(self) -> int:
        """Ring elements one digit plane occupies across the whole batch."""
        return sum(self.blocks_per_message)

    @property
    def cols(self) -> int:
        """Module width: one column per (digit, block) pair.

        The pairing is **digit-major** — column `digit * blocks + block` —
        following the orientation `gadget.decompose_vector` already returns,
        so the layout is read off the substrate rather than transposed into a
        second convention. Both sides of an opening index columns by this
        formula, so it is pinned here and nowhere else.
        """
        return self.decomposition.num_digits * self.blocks

    @property
    def beta_inf(self) -> int:
        """The opening bound: the digit bound, since the witness *is* digits."""
        return self.decomposition.beta_inf

    def column(self, digit: int, block: int) -> int:
        """The column index of one digit plane's block — the `cols` formula
        as the accessor callers use, so no call site re-derives it."""
        if not 0 <= digit < self.decomposition.num_digits:
            raise ValueError(
                f"column: digit {digit} outside [0, {self.decomposition.num_digits})"
            )
        if not 0 <= block < self.blocks:
            raise ValueError(f"column: block {block} outside [0, {self.blocks})")
        return digit * self.blocks + block
