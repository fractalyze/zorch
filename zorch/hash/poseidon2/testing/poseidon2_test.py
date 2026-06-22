"""Poseidon2 koalabear-16 byte-matches the honest Plonky3 reference."""

from __future__ import annotations

import dataclasses
import functools
import re

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

import zorch._composite as _composite
from zorch.hash.poseidon2.poseidon2 import (
    POSEIDON2_MARKER,
    POSEIDON2_MARKER_VERSION,
    Poseidon2,
    _permute_body,
    _permute_body_batched,
)
from zorch.hash.poseidon2.testing.koalabear16 import (
    KOALABEAR16_EXPECTED,
    KOALABEAR16_POSEIDON2_ATTRS,
    koalabear16_params,
    koalabear16_perm,
)
from zorch.testkit.jit_cache import assert_single_trace


class Poseidon2Koalabear16Test(absltest.TestCase):
    def test_permute_byte_matches_plonky3(self) -> None:
        p = koalabear16_perm()
        out = p.permute(jnp.arange(16, dtype=F))
        self.assertTrue(bool(jnp.array_equal(out, KOALABEAR16_EXPECTED)))

    def test_vmap_batch_matches(self) -> None:
        p = koalabear16_perm()
        x = jnp.arange(16, dtype=F)
        batch = jax.vmap(p.permute)(jnp.stack([x, x]))
        self.assertTrue(bool(jnp.array_equal(batch[0], KOALABEAR16_EXPECTED)))
        self.assertTrue(bool(jnp.array_equal(batch[1], KOALABEAR16_EXPECTED)))

    def test_permute_batched_matches_vmap(self) -> None:
        # permute_batched is numerically a vmap(permute) over the leading axis; it
        # differs only in how the region lowers (one shared body, not one per
        # batch shape — see test_permute_batched_shares_one_lowered_body).
        p = koalabear16_perm()
        states = jnp.arange(5 * 16, dtype=F).reshape(5, 16)
        self.assertTrue(
            bool(
                jnp.array_equal(p.permute_batched(states), jax.vmap(p.permute)(states))
            )
        )

    @absltest.skipUnless(
        _composite._HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp"
    )
    def test_permute_batched_shares_one_lowered_body(self) -> None:
        # zisk-zorch#36: ragged Merkle-fold rounds must reference ONE shared
        # permute body, not re-emit the ~width-sized s-box/MDS body per batch
        # shape. Lower two ragged batches in one module: there are two
        # zorch.poseidon2 ops (one per shape), but exactly one decomposition func
        # carries the real body (the rest are thin lax.map wrappers over the
        # shared _single_permute). A revert to vmap(permute) re-bakes the batch
        # into a fresh per-shape body and fails this.
        p = koalabear16_perm()
        a = jnp.arange(8 * 16, dtype=F).reshape(8, 16)
        b = jnp.arange(4 * 16, dtype=F).reshape(4, 16)
        txt = (
            jax.jit(
                lambda x, y: p.permute_batched(x).sum() + p.permute_batched(y).sum()
            )
            .lower(a, b)
            .as_text()
        )
        self.assertEqual(txt.count('stablehlo.composite "zorch.poseidon2"'), 2, txt)
        funcs = re.findall(r"func\.func.*?@[\w.$]+\([^)]*\)(.*?)\n  \}", txt, re.S)
        big = [body for body in funcs if body.count("stablehlo.multiply") > 50]
        self.assertLen(big, 1)

    def test_permute_batched_inline_fallback_matches(self) -> None:
        # Published-wheel path (no CompositeOp): the batched decomposition runs
        # inline as lax.map(_single_permute) over the batch. Must equal the
        # composite/vmap result — the shared body computes the real permutation.
        p = koalabear16_perm()
        states = jnp.arange(3 * 16, dtype=F).reshape(3, 16)
        ref = jax.vmap(p.permute)(states)
        orig = _composite._HAS_COMPOSITE_OP
        try:
            _composite._HAS_COMPOSITE_OP = False
            out = p.permute_batched(states)
        finally:
            _composite._HAS_COMPOSITE_OP = orig
        self.assertTrue(bool(jnp.array_equal(out, ref)))

    def test_permute_batched_reuses_one_trace_across_instances(self) -> None:
        # The batched jit zone must also share one trace across freshly built
        # same-params permutations (#214/#216) — the dedup is pointless if the
        # batched body re-traces per instance.
        states = jnp.arange(4 * 16, dtype=F).reshape(4, 16)
        calls = [
            functools.partial(koalabear16_perm().permute_batched, states)
            for _ in (0, 1)
        ]
        assert_single_trace(self, _permute_body_batched, calls)

    def test_permute_reuses_one_trace_across_instances(self) -> None:
        # Freshly built same-params permutations must share one module-level
        # permute trace — the static key compares by value (#214). Without the
        # zone, every composite emission re-traced the permutation body, which
        # dominated the PCS first-trace-per-config cost (#216).
        x = jnp.arange(16, dtype=F)
        calls = [functools.partial(koalabear16_perm().permute, x) for _ in (0, 1)]
        assert_single_trace(self, _permute_body, calls)

    def test_inline_fallback_byte_matches(self) -> None:
        # The published-wheel path (no CompositeOp): fused_region runs the
        # 6-operand decomposition inline. Must still byte-match the golden.
        orig = _composite._HAS_COMPOSITE_OP
        try:
            _composite._HAS_COMPOSITE_OP = False
            out = koalabear16_perm().permute(jnp.arange(16, dtype=F))
            self.assertTrue(bool(jnp.array_equal(out, KOALABEAR16_EXPECTED)))
        finally:
            _composite._HAS_COMPOSITE_OP = orig

    @absltest.skipUnless(
        _composite._HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp"
    )
    def test_permute_emits_poseidon2_named_composite(self) -> None:
        # The standard-MDS permute marks its region "zorch.poseidon2" so zkx
        # routes it to the dedicated Poseidon2Fusion emitter; the permutation
        # shape rides as composite.attributes — all four ints are required by
        # the zkx recognizer. W=16, E=4, I=20, alpha=3 for koalabear-16.
        p = koalabear16_perm()
        txt = jax.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{POSEIDON2_MARKER}"', composite_line)
        self.assertIn(KOALABEAR16_POSEIDON2_ATTRS, composite_line)
        self.assertIn(f"version = {POSEIDON2_MARKER_VERSION}", composite_line)
        # Exactly the 6 ABI operands [state, ext_init_rc, int_rc, ext_term_rc,
        # diag, off_diag]. A closed-over external matrix is lifted to a leading
        # 7th operand (jax.lax.composite prepends consts) and breaks the
        # Poseidon2Fusion operand ABI — the e2e GPU failure this guards against.
        operands = composite_line.split(f'"{POSEIDON2_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 6, composite_line)

    @absltest.skipUnless(
        _composite._HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp"
    )
    def test_free_form_external_matrix_uses_generic_marker(self) -> None:
        # The Poseidon2Fusion emitter assumes an (I + J_blocks) ⊗ M4 external
        # layer (M4 rides as a marker attribute), so a free-form matrix that is
        # NOT M4-block-structured must NOT take the zorch.poseidon2 route — it
        # falls back to the generic zorch.fused_region marker (LoopFusion lowers
        # the real body) to stay correct. (An M4-block-structured matrix — e.g.
        # the HorizenLabs reference — does take the dedicated route.)
        custom = jnp.arange(16 * 16, dtype=F).reshape(16, 16)
        p = Poseidon2(dataclasses.replace(koalabear16_params(), external_matrix=custom))
        txt = jax.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
        self.assertNotIn(POSEIDON2_MARKER, txt)
        self.assertIn("zorch.fused_region", txt)

    @absltest.skipUnless(
        _composite._HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp"
    )
    def test_non_plonky3_m4_takes_dedicated_route(self) -> None:
        # A non-default but M4-block-structured matrix (here the HorizenLabs
        # reference M4 that pil2/ZisK use) must take the dedicated zorch.poseidon2
        # route, carrying its own M4 as the external_m4 attribute — not fall back
        # to the generic marker. This is what makes the dedicated emitter usable
        # by the HorizenLabs variant without a per-matrix special case.
        hl_m4 = [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]]
        w = 16
        mds = jnp.array(
            [
                [hl_m4[i % 4][j % 4] * (2 if i // 4 == j // 4 else 1) for j in range(w)]
                for i in range(w)
            ],
            dtype=F,
        )
        p = Poseidon2(dataclasses.replace(koalabear16_params(), external_matrix=mds))
        txt = jax.jit(p.permute).lower(jnp.arange(w, dtype=F)).as_text()
        self.assertIn(POSEIDON2_MARKER, txt)
        self.assertIn(
            "external_m4 = dense<[5, 7, 1, 3, 4, 6, 1, 1, 1, 3, 5, 7, 1, 1, 4, 6]> :"
            " tensor<16xi64>",
            txt,
        )


if __name__ == "__main__":
    absltest.main()
