# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`_run_jagged_rounds_relaunch` (the host loop relaunching one shape-polymorphic
round kernel per round) is byte-identical to the unrolled `_run_jagged_rounds`
oracle -- both the eager-kernel path and the production export_dispatch=True path,
on every row/interaction/edge layout. The oracle is kept in-tree precisely for
this gate."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.logup_gkr.circuit import JaggedGkrLayer
from zorch.logup_gkr.jagged_prover import (
    _DEGREE,
    _round_metadata,
    _run_jagged_rounds,
    _run_jagged_rounds_relaunch,
    prove_jagged_layer,
)
from zorch.logup_gkr.prover import logup_combine
from zorch.logup_gkr.testing import random_jagged_layer, virtual_planes
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.testkit.transcript import CheapPermutation
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont


class RelaunchEqualsRunJaggedRoundsTest(parameterized.TestCase):
    """`_run_jagged_rounds_relaunch` — the host loop relaunching one
    fold-then-compute kernel per round — is byte-identical to the unrolled
    `_run_jagged_rounds`. Same arithmetic regrouped across the host Fiat-Shamir
    boundary, so the bound point, round polys, pair openings, AND the advanced
    transcript state must all match, on every row/interaction/edge layout."""

    def _setup(self, layer: JaggedGkrLayer, lam: Array, z: Array):
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
        head = (
            layer.numerator_0, layer.numerator_1,
            layer.denominator_0, layer.denominator_1,
            eq_row, eq_int, z, lam, claim,
        )
        return head, (meta, naturals, inv_vand, nrv, niv)

    def _assert_relaunch_equals(
        self, layer: JaggedGkrLayer, lam: Array, z: Array, challenge_limbs: int = 1
    ) -> None:
        head, (meta, naturals, inv_vand, nrv, niv) = self._setup(layer, lam, z)
        tail = (meta, naturals, inv_vand, nrv, niv, challenge_limbs)
        ref = _run_jagged_rounds(*head, cheap_transcript(KB), *tail)
        # Both the eager-kernel relaunch AND the production export-dispatch relaunch
        # (the cached jax.export binaries, the 4*g / 2*pp brackets, the dtype-mix
        # key, the gather=None -> identity substitution) must reproduce the unrolled
        # reference byte-for-byte -- prove_jagged_layer ships export_dispatch=True.
        for export_dispatch in (False, True):
            got = _run_jagged_rounds_relaunch(
                *head, cheap_transcript(KB), *tail, export_dispatch=export_dispatch
            )
            self._assert_matches_reference(ref, got, f"export_dispatch={export_dispatch}")

    def _assert_matches_reference(self, a, b, label: str) -> None:
        ach, at, apolys, *aopen = a
        bch, bt, bpolys, *bopen = b
        self.assertTrue(bool(jnp.all(ach == bch)), f"challenges diverged ({label})")
        self.assertTrue(bool(jnp.all(apolys == bpolys)), f"round polys diverged ({label})")
        for i, (x, y) in enumerate(zip(aopen, bopen)):
            self.assertTrue(bool(jnp.all(x == y)), f"pair opening {i} diverged ({label})")
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
    def test_relaunch_equals_base_field(self, row_counts, z_len) -> None:
        layer = random_jagged_layer(7, row_counts)
        self._assert_relaunch_equals(
            layer, rand_field(17, (), KB), rand_field(18, (z_len,), KB)
        )

    def test_relaunch_equals_multi_limb_ef(self) -> None:
        # koalabearx4 challenges (four squeezes reinterpreted) through the loop.
        layer = random_jagged_layer(41, (3, 1, 5, 2))
        self._assert_relaunch_equals(
            layer,
            rand_ext_field(51, (), KB, EF),
            rand_ext_field(52, (5,), KB, EF),
            challenge_limbs=4,
        )



@absltest.skipIf(
    jax.default_backend() == "cpu",
    "host-FS vs device is only meaningful off CPU (zkx#500)",
)
class RelaunchHostFsEqualsDeviceTest(parameterized.TestCase):
    """A full jagged-layer prove through the relaunch with `fs_on_host=True` (the
    host sponge) is byte-identical to the same prove on the device sponge -- the
    integration of the host-FS transcript and the relaunch round chain that the
    SP1-shaped consumer runs in production. A `CheapPermutation` keeps both off the
    marked path; only the Fiat-Shamir backend differs."""

    def _claim(self, layer: JaggedGkrLayer, lam: Array, z: Array) -> Array:
        niv = layer.num_interaction_variables
        n0, n1, d0, d1 = virtual_planes(layer, z.shape[0] - niv)
        eq = expand_eq_to_hypercube(z, jnp.ones((), z.dtype))
        return jnp.sum(logup_combine(lam, eq, n0, d1, n1, d0))

    def _prove(self, layer, lam, claim, z, limbs, fs_on_host):
        t = DuplexTranscript.new(
            CheapPermutation(width=8, dtype=KB), rate=4, fs_on_host=fs_on_host
        )
        return prove_jagged_layer(layer, lam, claim, z, t, challenge_limbs=limbs)

    @parameterized.named_parameters(("base", (3, 1, 5, 2), 5, 1), ("ef", (3, 1, 5, 2), 5, 4))
    def test_host_fs_relaunch_equals_device(self, row_counts, z_len, limbs) -> None:
        layer = random_jagged_layer(7, row_counts)
        if limbs == 1:
            lam, z = rand_field(17, (), KB), rand_field(18, (z_len,), KB)
        else:
            lam, z = rand_ext_field(51, (), KB, EF), rand_ext_field(52, (z_len,), KB, EF)
        claim = self._claim(layer, lam, z)
        pd, td, prd = self._prove(layer, lam, claim, z, limbs, False)
        ph, th, prh = self._prove(layer, lam, claim, z, limbs, True)
        self.assertTrue(bool(jnp.all(pd == ph)), "bound point diverged")
        for a, b in zip(jax.tree_util.tree_leaves(prd),
                        jax.tree_util.tree_leaves(prh)):
            self.assertTrue(bool(jnp.all(a == b)), "proof diverged")
        for f in ("input_buffer", "output_buffer", "sponge_state"):
            self.assertTrue(
                bool(jnp.all(getattr(td.state, f) == getattr(th.state, f))),
                f"transcript {f} diverged",
            )


if __name__ == "__main__":
    absltest.main()
