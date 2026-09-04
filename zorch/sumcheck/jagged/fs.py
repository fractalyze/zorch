# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The per-round Fiat-Shamir hop + scalar reduce: observe the round
poly, squeeze the challenge, and fold the claim/pad scalars -- traced
into the whole-layer jit, one fused region per round. (The round's
eval-point coordinate is sliced statically by the caller's loop index.)

The reduce (`eval_coeffs(poly, r)` + `mass * eq_factor(r, z)`) stays plain
device ops that consume the `zorch.duplex_fs` composite's challenge output --
zorch stays optimization-agnostic, leaving any FS-kernel absorption to the
consumer's xla layer rather than baking it in here."""

from __future__ import annotations

from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.sumcheck import gruen
from zorch.transcript import DuplexTranscript


def _reduce_body(
    r: Array,
    poly: Array,
    pad_adj: Array,
    z_cur: Array,
) -> tuple[Array, Array]:
    """Fold the round scalars by the round challenge `r`. The round's
    `eval_point` coordinate `z_cur` is sliced statically by the caller (the loop
    index is a compile-time constant), so no per-round gather rides here -- a
    device-resident index would cost a ~22us `fnp.take` dispatch every round.
    Returns the next `claim` and `pad_adj`; `r` stays the caller's, since the
    squeeze that produced it is now the caller's too. Plain (un-jitted) so it
    fuses into whichever kernel owns it -- the round loop's `_fs_reduce`."""
    return gruen.fold_round_scalars(poly, r, pad_adj, z_cur)


def _fs_reduce(
    poly: Array,
    transcript: DuplexTranscript,
    pad_adj: Array,
    z_cur: Array,
    challenges: ChallengePolicy,
) -> tuple[DuplexTranscript, Array, Array, Array]:
    """The per-round FS hop + reduce: observe `poly`, squeeze the challenge, then
    `_reduce_body`. Returns the advanced transcript and `(r, claim, pad_adj)`. No
    jit of its own -- it fuses into the round's compute under the whole-layer jit.

    The hop must reach an FS entry point rather than a backend body, so it lands
    on whichever backend the transcript carries: the device backend lowers it to
    the `zorch.duplex_fs` composite -- one register-resident kernel for the whole
    absorb+squeeze -- while `fs_on_host=True` runs the same call on the host
    sponge. This is the hottest FS call site in a jagged prove, ~78% of its hops.

    The reduce that consumes the challenge stays plain device ops --
    optimization-agnostic, left for the consumer's xla layer."""
    transcript, r = challenges.observe_and_sample(transcript, poly)
    claim, pad_adj = _reduce_body(r, poly, pad_adj, z_cur)
    return transcript, r, claim, pad_adj
