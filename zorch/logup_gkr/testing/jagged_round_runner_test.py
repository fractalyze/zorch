# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`_run_jagged_rounds` (the host loop threading one compute + FS hop per round,
traced into the whole-layer jit) is byte-identical to the unrolled
`_run_jagged_rounds_reference` oracle, on every row/interaction/edge layout and
under the fixed-width caps. The oracle is kept in-tree precisely for this
gate."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.logup_gkr._jagged_types import _DEGREE, _JaggedState, _Planes
from zorch.logup_gkr.circuit import JaggedGkrLayer
from zorch.logup_gkr.jagged_prover import _run_jagged_rounds
from zorch.logup_gkr.prover import logup_combine
from zorch.logup_gkr.testing import random_jagged_layer, virtual_planes
from zorch.logup_gkr.testing._jagged_reference import _run_jagged_rounds_reference
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde
from zorch.sumcheck.jagged.buffers import _LAYER_BUF_POOL, _pool_lay_batch
from zorch.sumcheck.jagged.schedule import (
    _round_live_meta,
    _round_metadata,
    _round_out_pairs,
    _row_counts_operand,
)
from zorch.sumcheck.jagged.types import (
    RoundWidthCaps,
    _InterpConsts,
    _JaggedSchedule,
)
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
        list[tuple[Array | None, Array, Array, Array]],
        Array,
        Array,
        int,
        int,
    ]:
        niv = layer.num_batch_variables
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
        # The runner's schedule carries the derived-schedule fields
        # (row_counts operand + live triples + the exact layout's static
        # padded widths); `meta` — the host-built explicit schedule — feeds
        # only the reference oracle, keeping the derivation cross-checked
        # against the original construction.
        sched = _JaggedSchedule(
            _row_counts_operand(layer.row_counts),
            _round_live_meta(layer.row_counts, nrv),
            _round_out_pairs(layer.row_counts, nrv),
            _InterpConsts(naturals, inv_vand),
            nrv,
            niv,
            challenge_limbs,
            meta=meta,
        )
        ref = _run_jagged_rounds_reference(state, sched, cheap_transcript(KB))
        # `_run_jagged_rounds` runs under the consumer's whole-layer jit (its FS hop
        # traces into the layer kernel); under jit it must reproduce the unrolled
        # eager reference byte-for-byte.
        got = jax.jit(lambda tr: _run_jagged_rounds(state, sched, tr))(
            cheap_transcript(KB)
        )
        self._assert_matches_reference(ref, got, "jit")
        # The fixed-width route (xla#179 size-invariance): every round runs at
        # the capped widths with the live prefix tracked by the `live` operand,
        # so one kernel shape serves all rounds/layers/shards. Both a slack cap
        # (real padding on every buffer) and the exact-fit cap (empty pads) must
        # reproduce the reference bit-for-bit under the whole-layer jit.
        for slack in (True, False):
            caps = self._caps(layer, nrv, niv, slack=slack)
            fixed_sched = _JaggedSchedule(
                _row_counts_operand(layer.row_counts),
                _round_live_meta(layer.row_counts, nrv),
                None,  # capped = width-preserving; no exact padded widths
                _InterpConsts(naturals, inv_vand),
                nrv,
                niv,
                challenge_limbs,
                caps,
            )
            label = "fixed-slack" if slack else "fixed-tight"
            got_fixed = jax.jit(lambda tr: _run_jagged_rounds(state, fixed_sched, tr))(
                cheap_transcript(KB)
            )
            self._assert_matches_reference(ref, got_fixed, label)
            # Wide-shard memory fix (#254): rolling the uniform row phase into a
            # lax.scan (bounded loop carry vs co-resident unrolled rounds) must be
            # byte-identical to the unrolled capped path.
            got_scan = jax.jit(
                lambda tr: _run_jagged_rounds(
                    state, fixed_sched, tr, force_row_scan=True
                )
            )(cheap_transcript(KB))
            self._assert_matches_reference(ref, got_scan, label + "-rowscan")

    @staticmethod
    def _caps(
        layer: JaggedGkrLayer, nrv: int, niv: int, *, slack: bool
    ) -> RoundWidthCaps:
        """Width caps for the fixed-width legs: `slack` pads every buffer well
        past its natural width (the cross-layer/shard reuse shape); the tight
        variant pins the caps to the layer's own widths (rounded up only where
        the ABI demands -- row to a multiple of 4, interaction to >= 4)."""
        round0_width = sum(rc + rc % 2 for rc in layer.row_counts)
        if not slack:
            return RoundWidthCaps(
                row=round0_width + (-round0_width % 4),
                eq_row=1 << nrv,
                interaction=max(4, 1 << niv),
            )
        return RoundWidthCaps(
            row=2 * round0_width + 8,
            eq_row=(1 << nrv) * 2,
            interaction=max(4, (1 << niv) * 2),
        )

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

    def test_multi_round_layout_matches_reference(self) -> None:
        # nrv=3 gives a 2-round row stretch and niv=3 a 2-round dense stretch,
        # exercising the row / boundary-handoff / dense phases of the
        # single-round export legs in one byte-gated call against the
        # reference oracle.
        layer = random_jagged_layer(23, (3, 1, 5, 2, 1, 1, 2, 4))
        self._check_round_runner(
            layer, rand_field(27, (), KB), rand_field(28, (6,), KB)
        )

    def test_layer_pool_donates_in_place(self) -> None:
        # The layer-entry lay-in (`_pool_lay_batch`, which the zone runs to
        # pre-lay each layer's planes) writes into pooled, DONATED cap-wide
        # buffers. Across two lay-ins the pool entry must keep the same device
        # allocation -- a silent donation failure would copy per layer,
        # reintroducing the cap-wide materialization the pool removes, and the
        # byte gates cannot see that regression (the values match either way).
        width = 36
        src = rand_field(31, (12,), KB)
        key = ("n0", width, src.dtype)
        _pool_lay_batch([("n0", src, width)])
        ptr = _LAYER_BUF_POOL[key].unsafe_buffer_pointer()
        _pool_lay_batch([("n0", src, width)])
        self.assertEqual(ptr, _LAYER_BUF_POOL[key].unsafe_buffer_pointer())

    def test_matches_reference_multi_limb_ef(self) -> None:
        # koalabearx4 challenges (four squeezes reinterpreted) through the loop.
        layer = random_jagged_layer(41, (3, 1, 5, 2))
        self._check_round_runner(
            layer,
            rand_ext_field(51, (), KB, EF),
            rand_ext_field(52, (5,), KB, EF),
            challenge_limbs=4,
        )

    def test_matches_reference_mixed_plane_ef(self) -> None:
        # The PRODUCTION plane mix: base-field numerators, extension-field
        # denominators (x - alpha). The first round carries the planes through
        # un-promoted, so its claimed kernel must handle per-plane fields
        # (limb strides, pad neutrals) within one round -- the uniform-plane
        # cases above cannot see a regression there.
        base = random_jagged_layer(41, (3, 1, 5, 2))
        layer = JaggedGkrLayer(
            base.numerator_0,
            base.numerator_1,
            rand_ext_field(61, (base.height,), KB, EF),
            rand_ext_field(62, (base.height,), KB, EF),
            base.row_counts,
        )
        self._check_round_runner(
            layer,
            rand_ext_field(51, (), KB, EF),
            rand_ext_field(52, (5,), KB, EF),
            challenge_limbs=4,
        )


if __name__ == "__main__":
    absltest.main()
