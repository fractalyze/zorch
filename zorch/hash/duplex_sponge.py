"""Add-absorb duplex sponge — scheme-agnostic over a Permutation.

A stateful sponge supporting interleaved absorb/squeeze (duplex): absorb ADDS
input into the rate lanes (`state[:rate] += block`, not overwrite — contrast the
one-shot overwrite `Sponge`), permuting when a rate block fills or the duplex
direction switches; squeeze reads the rate lanes, permuting when they drain.
This is the agnostic primitive a classic Fiat-Shamir prover (e.g. an
ark-sponge-faithful accumulation prover) drives; the scheme-specific challenge
packing, domain separation, and field conversions live in the consumer.

Kept separate from `DuplexTranscript` (the overwrite-mode Fiat-Shamir sponge),
not unified under an absorb-mode flag: the two implement different sponge
conventions and diverge on three independent axes — the absorb merge (add here
vs overwrite there), the squeeze read direction (this reads the rate low→high;
`DuplexTranscript` pops it high→low, so a partial squeeze returns different lanes,
not merely reversed ones), and the permute timing (this defers the permute on an
exactly-filled rate block and skips a spill permute when a squeeze request equals
the rate exactly). A shared core would have to parameterize all three — two
conventions in one config object, not real reuse — so the genuinely shared part
(a buffer, a position, a permute call) does not justify merging them.

Width comes from `permutation.width`; `rate` is the free parameter
(capacity = width - rate). The absorb/squeeze schedule is static (known element
counts), so the mode machine and permute triggers resolve at trace time —
mode/position are Python-level, only the field-element state is traced.

Unlike the one-shot `Sponge`/`Compression` (static configs used as jit-zone
keys), this carries traced per-step state and is threaded by return value, so it
deliberately omits the value-equality/hash those siblings define; pytree
registration and a static-key surface are left to the consumer that threads it
through `jit`, where the threading pattern can be validated rather than guessed.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.hash.permutation import Permutation

_ABSORBING = "absorbing"
_SQUEEZING = "squeezing"


class DuplexSponge:
    """Add-absorb duplex sponge over a fixed-width Permutation."""

    def __init__(self, permutation: Permutation, rate: int) -> None:
        if rate < 1:
            raise ValueError(f"rate ({rate}) must be >= 1")
        if rate >= permutation.width:
            raise ValueError(
                f"rate ({rate}) must be < permutation width ({permutation.width})"
            )
        self._permutation = permutation
        self.rate = rate
        self._state = jnp.zeros(permutation.width, dtype=permutation.dtype)
        self._mode = _ABSORBING
        self._pos = 0

    @property
    def has_dedicated_fusion(self) -> bool:
        """Whether the permutation lowers to a hash-dedicated fusion marker, so a
        consumer can wrap a whole region using this hash in an expandable
        composite. Delegates to the permutation; names no hash."""
        return self._permutation.has_dedicated_fusion

    def _with(self, *, state: Array, mode: str, pos: int) -> "DuplexSponge":
        new = object.__new__(DuplexSponge)
        new._permutation = self._permutation
        new.rate = self.rate
        new._state = state
        new._mode = mode
        new._pos = pos
        return new

    def absorb(self, elems: Array) -> "DuplexSponge":
        if elems.ndim != 1:
            raise ValueError(f"elems must be 1-D, got ndim={elems.ndim}")
        if elems.shape[0] == 0:
            return self  # empty input never touches the state (no direction switch)
        state, pos = self._state, self._pos
        if self._mode == _SQUEEZING:
            # Direction switch (squeeze -> absorb) permutes and resets to rate 0.
            state = self._permutation.permute(state)
            pos = 0
        state, pos = self._absorb_into_rate(state, pos, elems)
        return self._with(state=state, mode=_ABSORBING, pos=pos)

    def _absorb_into_rate(
        self, state: Array, start: int, elems: Array
    ) -> tuple[Array, int]:
        # Add elements into the rate lanes from `start`; when they spill past the
        # rate block, add what fits, permute, and recurse onto the fresh block.
        n = elems.shape[0]
        if start + n <= self.rate:
            return state.at[start : start + n].add(elems), start + n
        take = self.rate - start
        state = state.at[start : self.rate].add(elems[:take])
        state = self._permutation.permute(state)
        return self._absorb_into_rate(state, 0, elems[take:])

    def squeeze(self, n: int) -> tuple["DuplexSponge", Array]:
        if n < 0:
            raise ValueError(f"n ({n}) must be >= 0")
        state, pos = self._state, self._pos
        if self._mode == _ABSORBING:
            # Direction switch (absorb -> squeeze) permutes and resets to rate 0.
            state = self._permutation.permute(state)
            pos = 0
        elif pos == self.rate:
            # Rate fully drained: permute before reading the next block.
            state = self._permutation.permute(state)
            pos = 0
        state, pos, out = self._squeeze_from_rate(state, pos, n)
        return self._with(state=state, mode=_SQUEEZING, pos=pos), out

    def _squeeze_from_rate(
        self, state: Array, start: int, n: int
    ) -> tuple[Array, int, Array]:
        # Read n elements from the rate lanes starting at `start`; when the
        # request drains past the rate block, read what is there, permute, and
        # continue on the fresh block. The permute is skipped when the remaining
        # request length equals the rate exactly (ark-sponge's squeeze edge rule).
        # Block reads are collected and concatenated once, so the traced copy is
        # linear in the squeeze length rather than quadratic in the block count.
        chunks = []
        while start + n > self.rate:
            chunks.append(state[start : self.rate])
            take = self.rate - start
            if n != self.rate:
                state = self._permutation.permute(state)
            n -= take
            start = 0
        chunks.append(state[start : start + n])
        out = chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks)
        return state, start + n, out
