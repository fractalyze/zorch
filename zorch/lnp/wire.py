# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""What a verifier may believe about a proof it did not produce.

The sibling of `transcript.py`: that module owns how a ring stack becomes
transcript bytes, this one owns what `verify` is allowed to assume about
the values arriving on that wire. Both exist for the same reason — a
protocol module that respelled either would fork a convention its own
tests cannot see disagreeing.

The rule, one answer across the proof surface rather than per module:

- **Proof fields are a verdict.** The prover chose them, so "malformed"
  is the ordinary case of an adversary sending something, and the honest
  answer is `False`. A verifier that raises instead is a total predicate
  only over *well-formed* messages, which is not the property its caller
  needs — anywhere the proof arrives from a network, it is the difference
  between rejecting and crashing.
- **Statement fields raise.** The caller supplied the public parameters,
  so a malformed one is that caller's own bug. Reporting it as a failed
  proof is how a parameter mistake becomes a silently always-false
  verifier, so those keep reaching the raising gates.

Both halves route through a raising gate rather than restating its
predicate, so the two ways of asking about one array cannot drift apart.

Each protocol layer applies this through a single `_is_well_formed` over
its whole proof dataclass, rather than per field at the point of use —
a per-field habit leaves whichever field nobody remembered ungated, and
the field it left was the composite one.
"""

from __future__ import annotations

import numpy as np
from lattice_frx.canonical import is_canonical
from lattice_frx.split_ring import HostSplitRing

from zorch.commit.ajtai import AbdlopCommitment


def require_signed(ring: HostSplitRing, name: str, arr: np.ndarray, *lead: int) -> None:
    """The signed-integer twin of the scheme's ring-stack shape gate.

    Witnesses and responses are `(lead, d)` integer arrays, the challenge
    the rank-one case `(d,)`. A malformed one fails in this package's
    vocabulary instead of deep inside a ring op — and a `c` that is a
    list, or float coefficients silently truncated by `from_signed`, is
    refused rather than quietly scored against the real challenge.

    The two failure modes stay distinct, per lattice-frx's own contract
    rule: `TypeError` for dtype, `ValueError` for shape.
    """
    want = (*lead, ring.d)
    if not isinstance(arr, np.ndarray) or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            f"{name} must be a signed integer ndarray, got "
            f"{type(arr).__name__} dtype {getattr(arr, 'dtype', None)!r}"
        )
    if arr.shape != want:
        raise ValueError(f"{name} must have shape {want}, got {arr.shape}")


def is_signed(ring: HostSplitRing, arr: np.ndarray, *lead: int) -> bool:
    """`require_signed` asked rather than enforced."""
    try:
        require_signed(ring, "wire", arr, *lead)
    except (TypeError, ValueError):
        return False
    return True


def is_stack(scheme: AbdlopCommitment, arr: np.ndarray, *lead: int) -> bool:
    """Whether an untrusted `lead + (limbs, d)` ring stack is usable at all.

    The scheme's `require_stack` asked rather than enforced, extended to
    the array contract. The range check belongs to this side and not to
    `require_stack` because it is exactly what nothing else performs in
    time: the ring ops downstream *do* reject a non-canonical operand,
    and that rejection is the raise this answers first.
    """
    try:
        scheme.require_stack("wire", arr, *lead)
    except ValueError:
        return False
    return is_canonical(arr, scheme.ring.q_moduli)
