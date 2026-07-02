# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`_run_jagged_rounds` (the host loop running one fold-then-compute kernel per
round) is byte-identical to the unrolled `_run_jagged_rounds_reference` oracle
below, on every row/interaction/edge layout. The oracle -- the same sumcheck
math written inline, one helper call per step -- lives here as the differential
gate; the production driver regroups those calls (fold of round k + sum of round
k+1 per kernel) and must reproduce it byte-for-byte."""
from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.logup_gkr.circuit import JaggedGkrLayer, _pad_neutral
from zorch.logup_gkr.jagged_prover import (
    _DEGREE,
    _bind_lsb,
    _fold_scalars,
    _InterpConsts,
    _JaggedSchedule,
    _JaggedState,
    _paired_sums,
    _Planes,
    _round_coeffs,
    _round_metadata,
    _run_jagged_rounds,
)
from zorch.logup_gkr.prover import logup_combine
from zorch.logup_gkr.testing import random_jagged_layer, virtual_planes
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript, Transcript, sample_challenge

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


def _run_jagged_rounds_reference(
    state: _JaggedState,
    sched: _JaggedSchedule,
    transcript: Transcript,
) -> tuple[Array, Transcript, Array, Array, Array, Array, Array]:
    """The per-round jagged sumcheck written inline -- one `_paired_sums` /
    `_round_coeffs` / bind per round, no kernel regrouping. The readable oracle the
    production `_run_jagged_rounds` is byte-matched against. Returns the bound point
    (challenges reversed), the advanced transcript, the stacked round polynomials,
    and the four folded pair openings."""
    n0, n1, d0, d1 = state.planes.n0, state.planes.n1, state.planes.d0, state.planes.d1
    eq_row, eq_int, eval_point, lam, claim = (
        state.eq_row,
        state.eq_int,
        state.eval_point,
        state.lam,
        state.claim,
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
            gather, col_index, pair_index = meta[rnd]
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


class RoundRunnerMatchesReferenceTest(parameterized.TestCase):
    """`_run_jagged_rounds` — the host loop running one fold-then-compute kernel
    per round — is byte-identical to the unrolled `_run_jagged_rounds_reference`. Same
    arithmetic regrouped across the host Fiat-Shamir boundary, so the bound point,
    round polys, pair openings, AND the advanced transcript state must all match, on
    every row/interaction/edge layout."""

    def _setup(self, layer: JaggedGkrLayer, lam: Array, z: Array) -> tuple[
        _JaggedState,
        list[tuple[Array | None, Array, Array]],
        Array,
        Array,
        int,
        int,
    ]:
        niv = layer.num_interaction_variables
        nrv = z.shape[0] - niv
        one = jnp.ones((), z.dtype)
        eq_row = expand_eq_to_hypercube(z[niv:], one)
        eq_int = expand_eq_to_hypercube(z[:niv], one)
        n0, n1, d0, d1 = virtual_planes(layer, nrv)
        eq = expand_eq_to_hypercube(z, one)
        claim = jnp.sum(logup_combine(lam, eq, n0, d1, n1, d0))
        meta = _round_metadata(layer.row_counts, nrv)
        naturals = jnp.stack([jnp.array(j, z.dtype) for j in range(_DEGREE + 1)])
        inv_vand = compute_inv_vandermonde(_DEGREE, z.dtype)
        state = _JaggedState(
            _Planes(
                layer.numerator_0,
                layer.numerator_1,
                layer.denominator_0,
                layer.denominator_1,
            ),
            eq_row,
            eq_int,
            z,
            lam,
            claim,
        )
        return state, meta, naturals, inv_vand, nrv, niv

    def _check_round_runner(
        self, layer: JaggedGkrLayer, lam: Array, z: Array, challenge_limbs: int = 1
    ) -> None:
        state, meta, naturals, inv_vand, nrv, niv = self._setup(layer, lam, z)
        sched = _JaggedSchedule(
            meta, _InterpConsts(naturals, inv_vand), nrv, niv, challenge_limbs
        )
        ref = _run_jagged_rounds_reference(state, sched, cheap_transcript(KB))
        got = _run_jagged_rounds(state, sched, cheap_transcript(KB))
        self._assert_matches_reference(ref, got)

    def _assert_matches_reference(
        self,
        a: tuple[Array, Transcript, Array, Array, Array, Array, Array],
        b: tuple[Array, Transcript, Array, Array, Array, Array, Array],
    ) -> None:
        ach, at, apolys, *aopen = a
        bch, bt, bpolys, *bopen = b
        self.assertTrue(bool(jnp.all(ach == bch)), "challenges diverged")
        self.assertTrue(bool(jnp.all(apolys == bpolys)), "round polys diverged")
        for i, (x, y) in enumerate(zip(aopen, bopen, strict=True)):
            self.assertTrue(bool(jnp.all(x == y)), f"pair opening {i} diverged")
        if not isinstance(at, DuplexTranscript) or not isinstance(bt, DuplexTranscript):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        for f in ("input_buffer", "output_buffer", "sponge_state"):
            self.assertTrue(
                bool(jnp.all(getattr(at.state, f) == getattr(bt.state, f))),
                f"transcript {f} diverged",
            )
        self.assertEqual(int(at.state.in_pos), int(bt.state.in_pos))
        self.assertEqual(int(at.state.out_pos), int(bt.state.out_pos))

    @parameterized.named_parameters(
        # Row + interaction phases, odd/saturated/even segments, the nrv==1 and
        # niv==0 edges (no fix_and_sum_row chain / no boundary+interaction).
        ("std", (3, 1, 5, 2), 5),
        ("small", (3, 1), 3),
        ("nrv1", (1, 1), 2),
        ("saturated", (1, 1, 1, 1), 5),
        ("niv0", (2,), 2),
    )
    def test_matches_reference_base_field(
        self, row_counts: tuple[int, ...], z_len: int
    ) -> None:
        layer = random_jagged_layer(7, row_counts)
        self._check_round_runner(
            layer, rand_field(17, (), KB), rand_field(18, (z_len,), KB)
        )

    def test_matches_reference_multi_limb_ef(self) -> None:
        # koalabearx4 challenges (four squeezes reinterpreted) through the loop.
        layer = random_jagged_layer(41, (3, 1, 5, 2))
        self._check_round_runner(
            layer,
            rand_ext_field(51, (), KB, EF),
            rand_ext_field(52, (5,), KB, EF),
            challenge_limbs=4,
        )


if __name__ == "__main__":
    absltest.main()
