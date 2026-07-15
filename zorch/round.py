# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The composable IOP `Round` — zorch's nn.Module-style building block, and the
chains that compose Rounds like `nn.Sequential`.

Subclass `Round` and implement `__call__`, threading the transcript and calling
its `observe` / `sample` directly. A prover round maps
`(carry, transcript) -> (carry, transcript, msg)`; a verifier round maps
`(carry, msg, transcript) -> (carry, transcript, ok)`. The carry (sumcheck MLE
state, a GKR layer's running claim, ...) and the transcript thread functionally
— never hidden mutable state.

`Round` is the one recursive unit — a prover<->verifier IOP interaction that
nests: a single per-variable step is a `Round`, and a whole sumcheck (its
per-variable `Round`s bundled) is also a `Round`, since a chain is itself a
`Round`. Two *roles* a `Round` plays in a scheme's `prove_chain` get their own
nominal marker subclass, so a reader (and a chain) can tell them apart — the
markers add no behavior:

- A `Stage` is one phase of the top-level `prove_chain` — a zerocheck, a
  lincheck, a witness opening: the phases a consumer sequences into its
  `ProveChain`. It usually runs a sub-protocol — an inner sumcheck scanned by
  `zorch.prove` / `zorch.verify`, or a PCS open — and carries stage-local
  witness.
- A `Bridge` is a transcript-only `Round` for soundness or security — a grind, a
  sampled-and-discarded challenge, a framed observe, an RLC of claims under a
  fresh challenge. It touches only the transcript and the carry, and its verifier
  dual replays the same ops. It usually sits inside a stage; a scheme may also
  place one in the `prove_chain` between two stages (Spartan's RLC).

Every other `Round` is unmarked — a per-variable inner round, or a composite
that is neither a top-level phase nor a connective. On the prover side every
round shares one shape, so `ProverRound` is the single prover contract
(`ProveChain` / `fold_rounds`). On the verifier side the inner round must also
surface its sampled challenge for `zorch.verify` to collect into the evaluation
point, so the contracts split: `VerifierRound` (stage, `VerifyChain`) vs
`InnerVerifierRound` (per-variable). The Protocols are structural, so a
wrong-shaped — or wrong-level — round is a type error.

A composite protocol is itself a `Round`: `ProveChain` / `VerifyChain` sequence
sub-Rounds, threading the carry + transcript, so chains nest.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from frx import Array

from zorch.transcript import Transcript


class Round:
    def __call__(self, *args: Any) -> Any:
        """Run one round, threading the transcript — see the module docstring for
        the prover and verifier signatures. Implemented by subclasses."""
        raise NotImplementedError


class Stage(Round):
    """A `Round` that is one phase of a top-level prover chain — a zerocheck, a
    lincheck, a witness opening: the phases a consumer sequences into its
    `ProveChain`. A stage usually runs a sub-protocol (an inner sumcheck via
    `zorch.prove` / `fold_rounds`, or a PCS open) and carries stage-local
    witness on the instance. A semantic marker over `Round`: it adds no
    behavior, so a reader can tell a top-level phase from the inner rounds it
    drives. A round that is only a link in a stage's own sub-chain (a GKR layer
    round) is not a `Stage` — it stays a plain `Round`."""


class Bridge(Round):
    """A transcript-only `Round` for soundness or security — a grind, a
    sampled-and-discarded challenge, a framed observe, an RLC of claims under a
    fresh challenge. It touches only the transcript and the carry (no witness,
    no PCS work), and its verifier dual replays the same ops; a bridge that
    sends nothing emits `None` as its message. Usually inside a stage; a scheme
    may also place one in the `prove_chain` between two stages (Spartan's RLC).
    A semantic marker over `Round`, like `Stage`."""


class ProverRound(Protocol):
    """What `ProveChain` / `fold_rounds` consume. `carry` and `msg` are
    scheme-defined (`Any`); the checker enforces arity and the threaded
    `transcript` — what discriminates a prover round from a verifier one.

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
    """What the `zorch.verify` scan consumes: the per-variable verifier inside a
    stage's sumcheck. The trailing element is the bound coordinate `r`, collected
    by the driver into the evaluation point. Positional-only as on `ProverRound`."""

    def __call__(
        self, claim: Any, msg: Any, transcript: Transcript, /
    ) -> tuple[Any, Transcript, Array, Array]: ...


class ProveChain(Round):
    """Sequence prover rounds (nn.Sequential). Threads the carry + transcript
    through each round and collects their messages. Itself a `Round`, so chains
    nest.

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
