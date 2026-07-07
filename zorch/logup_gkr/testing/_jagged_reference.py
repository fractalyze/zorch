# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The unrolled byte-match oracle for the jagged round runner.

`_run_jagged_rounds_reference` writes the per-round jagged sumcheck out with an
explicit observe/sample per round; `jagged_round_runner_test` asserts the
production `_run_jagged_rounds` (one traced compute + FS hop per round, under the
whole-layer jit) reproduces it byte-for-byte. Test-only — nothing on the prover
path calls it, so it lives here rather than in `jagged_prover`.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from zorch.logup_gkr._jagged_rounds import _paired_sums
from zorch.logup_gkr._jagged_types import _JaggedState
from zorch.logup_gkr.circuit import _pad_neutral
from zorch.sumcheck.jagged.fs import _fold_scalars
from zorch.sumcheck.jagged.rounds import _bind_lsb, _round_coeffs
from zorch.sumcheck.jagged.types import _JaggedSchedule
from zorch.transcript import Transcript, sample_challenge


def _run_jagged_rounds_reference(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The unrolled oracle for `_run_jagged_rounds`: the per-round jagged sumcheck
    written out with an explicit observe/sample per round. Returns the bound point
    (challenges reversed), the advanced transcript, the stacked round polynomials,
    and the four folded pair openings. The round runner must match this byte-for-byte.
    """
    n0, n1, d0, d1 = state.planes.n0, state.planes.n1, state.planes.d0, state.planes.d1
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
    )
    if sched.meta is None:
        raise ValueError(
            "the reference oracle needs the schedule's host-built explicit "
            "meta (_round_metadata) — the round loop's derived-schedule "
            "fields do not carry it"
        )
    meta, nrv, niv = sched.meta, sched.nrv, sched.niv
    naturals, inv_vand = sched.consts.naturals, sched.consts.inv_vand
    challenge_limbs = sched.challenge_limbs
    one = jnp.ones((), eval_point.dtype)
    eq_adj = one
    pad_adj = one
    point = eval_point
    polys: list[Array] = []
    challenges: list[Array] = []
    for rnd in range(nrv + niv):
        in_rows = rnd < nrv
        if in_rows:
            # The oracle runs the exact layout; the schedule's `live` operand
            # (the fixed-width prefix marker) is the production loop's concern.
            gather, col_index, pair_index, _live = meta[rnd]
            n0, n1, d0, d1 = _pad_neutral(n0, n1, d0, d1, gather)
            w = eq_int[col_index]
            eval_zero, eval_half, eq_sum = _paired_sums(
                n0,
                n1,
                d0,
                d1,
                eq_row[pair_index * 2] * w,
                eq_row[pair_index * 2 + 1] * w,
                lam,
            )
        else:
            eval_zero, eval_half, eq_sum = _paired_sums(
                n0, n1, d0, d1, eq_int[0::2], eq_int[1::2], lam
            )
        poly = _round_coeffs(
            eval_zero,
            eval_half,
            eq_sum,
            eq_adj,
            pad_adj,
            point[-1],
            claim,
            naturals,
            inv_vand,
        )
        transcript = transcript.observe(poly)
        transcript, r = sample_challenge(transcript, claim.dtype, challenge_limbs)
        polys.append(poly)
        challenges.append(r)

        claim, pad_adj = _fold_scalars(poly, r, pad_adj, point[-1], one)
        n0, n1, d0, d1 = (_bind_lsb(a, r) for a in (n0, n1, d0, d1))
        if in_rows:
            eq_row = _bind_lsb(eq_row, r)
            if rnd == nrv - 1:
                # Rows exhausted: the accumulated row-eq product becomes the
                # scalar factor of every batch round; pad_adj restarts
                # to track the batch variables' own bound mass.
                eq_adj = pad_adj
                pad_adj = one
        else:
            eq_int = _bind_lsb(eq_int, r)
        point = point[:-1]

    return (
        jnp.stack(challenges[::-1]),
        transcript,
        jnp.stack(polys),
        n0[0],
        n1[0],
        d0[0],
        d1[0],
    )
