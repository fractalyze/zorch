# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The composable IOP `Round` — zorch's nn.Module-style building block, and the
chains that compose Rounds like `nn.Sequential`.

Subclass `Round` and implement `__call__`, threading the transcript and calling
its `observe` / `sample` directly. A prover round maps
`(carry, transcript) -> (carry, transcript, msg)`; a chain's verifier round maps
`(carry, msg, transcript) -> (carry, transcript, ok)`. The carry (sumcheck MLE
state, a GKR layer's running claim, ...) and the transcript thread functionally
— never hidden mutable state. Those two shapes are the `ProverRound` and
`ChainVerifierRound` Protocols below — the typed contracts `ProveChain` /
`VerifyChain` (and `fold_rounds`) accept, so a wrong-shaped round is a type error
rather than a runtime surprise. `Round` stays the nominal base subclasses inherit;
the Protocols are what the drivers type against (structural, so a round need only
match the shape).

(The per-variable sumcheck verifier in `zorch.sumcheck.verifier` is a
specialized shape — it also returns the sampled challenge, which the
`zorch.verify` driver collects into the point — so it pairs with that driver,
not with `VerifyChain`.)

A composite protocol is itself a `Round`: `ProveChain` / `VerifyChain` sequence
sub-Rounds, threading the carry + transcript, so chains nest (a chain of layer
chains, each a chain of per-variable rounds). The per-variable sumcheck loop is
the homogeneous case (one round repeated) with its own runners in `zorch.prove`
/ `zorch.verify`; the chains here are the heterogeneous case (distinct rounds in
sequence, e.g. GKR layers).
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
    """The contract `ProveChain` / `fold_rounds` consume: a prover round maps
    `(carry, transcript) -> (carry, transcript, msg)`. `carry` and `msg` are
    scheme-defined (`Any`); what the checker enforces is the arity and the threaded
    `transcript` — a verifier round (extra `msg` arg) or a 4-tuple return does not
    satisfy it, so handing one to a prover driver is a type error.

    Parameters are positional-only (`/`): a round names its carry `state` /
    `claim` / `layer`, so binding the contract to a parameter *name* would reject
    every real round. Position + arity is the contract."""

    def __call__(
        self, carry: Any, transcript: Transcript, /
    ) -> tuple[Any, Transcript, Any]: ...


class ChainVerifierRound(Protocol):
    """The dual contract `VerifyChain` consumes: a verifier round maps
    `(carry, msg, transcript) -> (carry, transcript, ok)`. The per-variable sumcheck
    verifier returns the sampled challenge too (a 4-tuple) and so pairs with
    `zorch.verify`, not this — by design it does not satisfy this Protocol.

    Positional-only (`/`) for the same reason as `ProverRound`: rounds name the
    message `layer_proof` / `msg`, so the contract is position + arity, not names."""

    def __call__(
        self, carry: Any, msg: Any, transcript: Transcript, /
    ) -> tuple[Any, Transcript, Array]: ...


class ProveChain(Round):
    """Sequence prover rounds (nn.Sequential). Threads the carry + transcript
    through each round and collects their messages. Itself a `Round`, so chains
    nest."""

    def __init__(self, rounds: Iterable[ProverRound]) -> None:
        self.rounds = list(rounds)

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
    rounds (one per round, in order)."""

    def __init__(self, rounds: Iterable[ChainVerifierRound]) -> None:
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
