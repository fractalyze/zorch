# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`_run_jagged_rounds` (the host loop threading one compute + FS hop per round,
traced into the whole-layer jit) is byte-identical to the unrolled
`_run_jagged_rounds_reference` oracle, on every row/interaction/edge layout. The
oracle is kept in-tree precisely for this gate."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.logup_gkr.circuit import JaggedGkrLayer
from zorch.logup_gkr.jagged_prover import (
    _DEGREE,
    _InterpConsts,
    _JaggedSchedule,
    _JaggedState,
    _Planes,
    _round_metadata,
    _run_jagged_rounds,
    _run_jagged_rounds_reference,
)
from zorch.logup_gkr.prover import logup_combine
from zorch.logup_gkr.testing import random_jagged_layer, virtual_planes
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript, Transcript

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


class RoundRunnerMatchesReferenceTest(parameterized.TestCase):
    """`_run_jagged_rounds` — the host loop dispatching one fold-then-compute kernel
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
        # `_run_jagged_rounds` runs under the consumer's whole-layer jit (its FS hop
        # traces into the layer kernel); under jit it must reproduce the unrolled
        # eager reference byte-for-byte.
        got = jax.jit(lambda tr: _run_jagged_rounds(state, sched, tr))(
            cheap_transcript(KB)
        )
        self._assert_matches_reference(ref, got, "jit")

    def _assert_matches_reference(
        self,
        a: tuple[Array, Transcript, Array, Array, Array, Array, Array],
        b: tuple[Array, Transcript, Array, Array, Array, Array, Array],
        label: str,
    ) -> None:
        ach, at, apolys, *aopen = a
        bch, bt, bpolys, *bopen = b
        self.assertTrue(bool(jnp.all(ach == bch)), f"challenges diverged ({label})")
        self.assertTrue(
            bool(jnp.all(apolys == bpolys)), f"round polys diverged ({label})"
        )
        for i, (x, y) in enumerate(zip(aopen, bopen, strict=True)):
            self.assertTrue(
                bool(jnp.all(x == y)), f"pair opening {i} diverged ({label})"
            )
        if not isinstance(at, DuplexTranscript) or not isinstance(bt, DuplexTranscript):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        for f in ("input_buffer", "output_buffer", "sponge_state"):
            self.assertTrue(
                bool(jnp.all(getattr(at.state, f) == getattr(bt.state, f))),
                f"transcript {f} diverged ({label})",
            )
        self.assertEqual(int(at.state.in_pos), int(bt.state.in_pos), label)
        self.assertEqual(int(at.state.out_pos), int(bt.state.out_pos), label)

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
