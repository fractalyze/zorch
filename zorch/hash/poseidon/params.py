"""PoseidonParams — the classic-Poseidon parameter surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from frx import Array


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
    # enclosing zone per call (docs/reference/conventions.md "Pytree
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
        Montgomery-decodes without needing frx x64."""
        w = self.width
        canon = np.asarray(self.mds).astype(object)
        return tuple(tuple(int(canon[i, j]) for j in range(w)) for i in range(w))


@dataclass(frozen=True)
class SparsePoseidonParams:
    """Optimized-sparse Poseidon parameter surface — the same classic-Poseidon
    permutation as `PoseidonParams` but with the partial rounds re-factored for
    speed: the partial round applies a cheap rank-structured update instead of the
    dense MDS, a single transition matrix `P` follows the last pre-partial round,
    and each partial round's `width` constants are folded into one lane-0 constant.
    The folded constants cannot be re-expanded into `PoseidonParams`, so this is a
    separate surface (and permutation), not a `PoseidonParams` variant.

    The core treats `dtype` as opaque and names no field/scheme/zkVM. The S-box
    precedes the round constant (the constant seeds the next round's S-box); the
    initial ARC seeds the first. The schedule, with `H = half_full_rounds`,
    `W = width`, `NP = n_partial_rounds`:

        state += initial_arc
        (H - 1)× :  state = mds @ (state^alpha + rc)              # full
        1× (transition)     :  state = transition_matrix @ (state^alpha + rc)
        NP×      :  sparse partial round (S-box lane 0, then + rc)
        (H - 1)× :  state = mds @ (state^alpha + rc)              # full
        1× (final)          :  state = mds @ state^alpha          # no rc

    Contract (validated in __post_init__):

      initial_arc      : (W,)         seeds the first S-box (added before round 0).
      full_rc_pre      : (H-1, W)     post-S-box constant, each pre-partial full round.
      transition_rc    : (W,)         post-S-box constant of the transition round (P).
      partial_rc       : (NP,)        lane-0 post-S-box constant, each partial round.
      full_rc_post     : (H-1, W)     post-S-box constant, each post-partial full round.
      mds              : (W, W)       dense MDS `M`, applied as `M @ state`.
      transition_matrix: (W, W)       `P`, the transition round's linear layer.
      partial_dot      : (NP, W)      partial round r's lane-0 dot row.
      partial_col      : (NP, W-1)    partial round r's lane-t (t>=1) update column.

    Matrices apply as `M @ state` (`out[i] = sum_j M[i][j] * state[j]`); a
    reference that stores the transpose must transpose before constructing params.
    The final full round takes no ARC (folded into the preceding constants).
    """

    width: int
    dtype: Any
    alpha: int
    half_full_rounds: int
    n_partial_rounds: int
    initial_arc: Array
    full_rc_pre: Array
    transition_rc: Array
    partial_rc: Array
    full_rc_post: Array
    mds: Array
    transition_matrix: Array
    partial_dot: Array
    partial_col: Array

    def __post_init__(self) -> None:
        if self.alpha < 1:
            raise ValueError(f"alpha must be a positive int, got {self.alpha}")
        if self.half_full_rounds < 1:
            raise ValueError(
                f"half_full_rounds must be positive, got {self.half_full_rounds}"
            )
        if self.n_partial_rounds < 0:
            raise ValueError(
                f"n_partial_rounds must be non-negative, got {self.n_partial_rounds}"
            )
        if self.width < 1:
            raise ValueError(f"width must be positive, got {self.width}")
        w = self.width
        h = self.half_full_rounds
        npr = self.n_partial_rounds
        checks = {
            "initial_arc": ((w,), self.initial_arc),
            "full_rc_pre": ((h - 1, w), self.full_rc_pre),
            "transition_rc": ((w,), self.transition_rc),
            "partial_rc": ((npr,), self.partial_rc),
            "full_rc_post": ((h - 1, w), self.full_rc_post),
            "mds": ((w, w), self.mds),
            "transition_matrix": ((w, w), self.transition_matrix),
            "partial_dot": ((npr, w), self.partial_dot),
            "partial_col": ((npr, w - 1), self.partial_col),
        }
        for name, (want, arr) in checks.items():
            got = tuple(np.shape(arr))
            if got != want:
                raise ValueError(f"{name}: expected shape {want}, got {got}")
            if arr.dtype != self.dtype:
                raise ValueError(
                    f"{name}: expected dtype {self.dtype}, got {arr.dtype}"
                )

    # Value equality/hash: a permutation rides pytree aux, which must compare by
    # value — identity equality re-traces the enclosing jit zone on every freshly
    # built instance (issue #163). One per-instance cached host-side key, as
    # `PoseidonParams`.
    _ARRAY_FIELDS = (
        "initial_arc",
        "full_rc_pre",
        "transition_rc",
        "partial_rc",
        "full_rc_post",
        "mds",
        "transition_matrix",
        "partial_dot",
        "partial_col",
    )

    def _value_key(self) -> tuple:
        k = self.__dict__.get("_key")
        if k is None:
            k = (
                self.width,
                self.dtype,
                self.alpha,
                self.half_full_rounds,
                self.n_partial_rounds,
            ) + tuple(
                np.asarray(getattr(self, f)).tobytes() for f in self._ARRAY_FIELDS
            )
            object.__setattr__(self, "_key", k)
        return k

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, SparsePoseidonParams):
            return NotImplemented
        return self._value_key() == other._value_key()

    def __hash__(self) -> int:
        h = self.__dict__.get("_hash")
        if h is None:
            h = hash(self._value_key())
            object.__setattr__(self, "_hash", h)
        return h
