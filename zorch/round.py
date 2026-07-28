# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The composable IOP `Round` — zorch's nn.Module-style building block, and the
chains that compose Rounds like `nn.Sequential`.

Subclass `Round` and implement `__call__`, threading the transcript and calling
its `observe` / `sample` directly. A prover round maps
`(carry, transcript) -> (carry, transcript, msg)`; a verifier round maps
`(carry, msg, transcript) -> (carry, transcript, ok)`. The carry (sumcheck MLE
state, a GKR layer's running claim, ...) and the transcript thread functionally
— never hidden mutable state.

Rounds live at two levels. A *stage* round is one step of the heterogeneous
protocol sequence (a GKR layer), run by the chains here; an *inner* round binds
one variable of a stage's sumcheck — the homogeneous case, scanned by
`zorch.sumcheck`'s `prove` / `verify`, typically from inside a stage round. On
the prover side both levels share one shape, so `ProverRound` is the single
prover contract (`ProveChain`). On the verifier side the inner round must also
surface its sampled challenge for `verify` to collect into the evaluation point,
so the contracts split: `VerifierRound` (stage, `VerifyChain`) vs
`InnerVerifierRound` (per-variable). The Protocols are structural, so a
wrong-shaped — or wrong-level — round is a type error. `Round` stays the
nominal base subclasses inherit.

A composite protocol is itself a `Round`: `ProveChain` / `VerifyChain` sequence
sub-Rounds, threading the carry + transcript, so chains nest.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from jax import Array

from zorch.transcript import Transcript


class Round:
    def __call__(self, *args: Any) -> Any:
        """Run one round, threading the transcript — see the module docstring for
        the prover and verifier signatures. Implemented by subclasses."""
        raise NotImplementedError


class ProverRound(Protocol):
    """What `ProveChain` consumes. `carry` and `msg` are scheme-defined (`Any`);
    the checker enforces arity and the threaded `transcript` — what discriminates
    a prover round from a verifier one.

    Parameters are positional-only (`/`): rounds name their carry `state` /
    `claim` / `layer`, so a name-bound contract would reject every real round."""

    def __call__(
        self, carry: Any, transcript: Transcript, /
    ) -> tuple[Any, Transcript, Any]: ...


class VerifierRound(Protocol):
    """What `VerifyChain` consumes: a stage-level verifier round. Positional-only
    as on `ProverRound`."""

    def __call__(
        self, carry: Any, msg: Any, transcript: Transcript, /
    ) -> tuple[Any, Transcript, Array]: ...


class InnerVerifierRound(Protocol):
    """What the `zorch.sumcheck.verifier.verify` scan consumes: the per-variable
    verifier inside a stage's sumcheck. The trailing element is the bound
    coordinate `r`, collected by the driver into the evaluation point.
    Positional-only as on `ProverRound`."""

    def __call__(
        self, claim: Any, msg: Any, transcript: Transcript, /
    ) -> tuple[Any, Transcript, Array, Array]: ...


class ProveChain(Round):
    """Sequence prover rounds (nn.Sequential). Threads the carry + transcript
    through each round and collects their messages. Itself a `Round`, so chains
    nest.

    Distinct rounds (a GKR layer pyramid) pass a list of rounds; a folding open
    that repeats one round N times passes `[rnd] * n`.

    The rounds are consumed lazily on `__call__`: a generator that builds each
    round on demand lets a proved round (and the witness it holds) be released
    before the next is built, so at most one layer of a big-witness pyramid
    stays live. A chain over a one-shot iterable is single-use — build it from
    a list to call it more than once."""

    def __init__(self, rounds: Iterable[ProverRound]) -> None:
        self.rounds = rounds

    def __call__(
        self, carry: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, list[Any]]:
        msgs = []
        for rnd in self.rounds:
            carry, transcript, msg = rnd(carry, transcript)
            msgs.append(msg)
        return carry, transcript, msgs


class VerifyChain(Round):
    """Verifier dual of `ProveChain`: replays each round against its message,
    threading the carry, and ANDs every round's `ok`. `msgs` aligns with the
    rounds (one per round, in order). Unlike `ProveChain` it materializes its
    rounds — the len-vs-msgs fail-loud check needs them all up front."""

    def __init__(self, rounds: Iterable[VerifierRound]) -> None:
        self.rounds = list(rounds)

    def __call__(
        self, carry: Any, msgs: Sequence[Any], transcript: Transcript
    ) -> tuple[Any, Transcript, Array]:
        # Fail loud: a short msgs list would let zip skip rounds while ok stays
        # True -- a silent accept.
        if len(msgs) != len(self.rounds):
            raise ValueError(
                f"need one message per round: {len(self.rounds)} rounds, "
                f"got {len(msgs)} messages"
            )
        ok = True
        for rnd, msg in zip(self.rounds, msgs):
            carry, transcript, ok_round = rnd(carry, msg, transcript)
            ok = ok & ok_round
        return carry, transcript, ok
