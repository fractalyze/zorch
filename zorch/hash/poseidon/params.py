"""PoseidonParams — the classic-Poseidon parameter surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from jax import Array


@dataclass(frozen=True)
class PoseidonParams:
    """Fully-free parameter surface of a classic Poseidon permutation.

    The core treats `dtype` as opaque and names no field/scheme/zkVM. Classic
    Poseidon (ark-sponge / HorizenLabs reference style) is the symmetric round
    function: every round is `ARC -> S-box -> dense MDS`, with the S-box applied
    to all lanes in a *full* round and to the last lane only in a *partial*
    round. The rounds split full/partial/full — `full_rounds/2` full, then
    `partial_rounds` partial, then `full_rounds/2` full — and the dense MDS runs
    on *every* round.

    Contract (validated in __post_init__):
      mds : (width, width) over dtype, small canonical ints; applied as
          `mds @ state` every round (so any matrix works). The dedicated
          emitter carries it as a marker attribute, so it is held as small
          canonical ints, not an opaque field array.
      round_constants : (full_rounds + partial_rounds, width) over dtype —
          one full-width ARC vector per round, full and partial rounds alike.
      full_rounds : even, positive; split half-before / half-after the partials.
      partial_rounds : non-negative.
      alpha : positive S-box exponent `x^alpha`; caller guarantees
          gcd(alpha, p-1) == 1 (the core does not know p, so it cannot check).
    """

    width: int
    dtype: Any
    alpha: int
    full_rounds: int
    partial_rounds: int
    round_constants: Array
    mds: Array

    def __post_init__(self) -> None:
        if self.alpha < 1:
            raise ValueError(f"alpha must be a positive int, got {self.alpha}")
        if self.full_rounds < 1 or self.full_rounds % 2 != 0:
            raise ValueError(
                f"full_rounds must be a positive even int, got {self.full_rounds}"
            )
        if self.partial_rounds < 0:
            raise ValueError(
                f"partial_rounds must be non-negative, got {self.partial_rounds}"
            )
        w = self.width
        total_rounds = self.full_rounds + self.partial_rounds
        checks = {
            "mds": ((w, w), self.mds),
            "round_constants": ((total_rounds, w), self.round_constants),
        }
        for name, (want, arr) in checks.items():
            got = tuple(np.shape(arr))
            if got != want:
                raise ValueError(f"{name}: expected shape {want}, got {got}")
            if arr.dtype != self.dtype:
                raise ValueError(
                    f"{name}: expected dtype {self.dtype}, got {arr.dtype}"
                )

    # Value equality/hash: a permutation rides pytree aux (`DuplexTranscript`
    # meta_fields), which must compare by value — identity equality turns every
    # freshly built transcript into a new jit cache key, re-tracing the whole
    # enclosing zone per call (issue #163; docs/conventions.md "Pytree
    # registration"). The dataclass-derived __eq__ is unusable here anyway:
    # `==` on the Array fields is elementwise. Both methods go through one
    # per-instance cached host-side key: jit dispatch calls __eq__ on the aux
    # per call, so comparing live device arrays there would cost a
    # device->host sync per dispatch.
    _ARRAY_FIELDS = ("round_constants", "mds")

    def _value_key(self) -> tuple:
        k = self.__dict__.get("_key")
        if k is None:
            k = (
                self.width,
                self.dtype,
                self.alpha,
                self.full_rounds,
                self.partial_rounds,
            ) + tuple(
                np.asarray(getattr(self, f)).tobytes() for f in self._ARRAY_FIELDS
            )
            object.__setattr__(self, "_key", k)
        return k

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, PoseidonParams):
            return NotImplemented
        return self._value_key() == other._value_key()

    def __hash__(self) -> int:
        # Memoized like `_key`: the permute jit zone hashes the params on
        # every dispatch, and CPython never caches tuple hashes (nor bytes
        # hashes from 3.13), so a bare hash(key) would re-SipHash the
        # constant-matrix bytes per permute call.
        h = self.__dict__.get("_hash")
        if h is None:
            h = hash(self._value_key())
            object.__setattr__(self, "_hash", h)
        return h

    @property
    def mds_rows(self) -> tuple[tuple[int, ...], ...]:
        """The `width × width` MDS as canonical ints (rows of ints) — the form
        the body applies via integer literals (no captured field array, which a
        name-routed `fused_region` would lift to a leading operand) and the
        dedicated emitter carries as a marker attribute (flattened row-major at
        the call-site). Canonical ints come from a numpy object cast, which
        Montgomery-decodes without needing jax x64."""
        w = self.width
        canon = np.asarray(self.mds).astype(object)
        return tuple(tuple(int(canon[i, j]) for j in range(w)) for i in range(w))
