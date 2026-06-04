"""Poseidon2 koalabear-16 byte-matches the honest Plonky3 reference."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

import zorch._composite as _composite
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import (
    KOALABEAR16_EXPECTED,
    koalabear16_params,
    koalabear16_perm,
)


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
        # The standard-MDS permute marks its region "poseidon2:W:E:I:S" so zkx
        # routes it to the dedicated Poseidon2Fusion emitter. W=16, E=4, I=20,
        # alpha=3 for koalabear-16.
        p = koalabear16_perm()
        txt = jax.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        self.assertIn("poseidon2:16:4:20:3", txt)
        # Exactly the 6 ABI operands [state, ext_init_rc, int_rc, ext_term_rc,
        # diag, off_diag]. A closed-over external matrix is lifted to a leading
        # 7th operand (jax.lax.composite prepends consts) and breaks the
        # Poseidon2Fusion operand ABI — the e2e GPU failure this guards against.
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        operands = composite_line.split('"poseidon2:16:4:20:3"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 6, composite_line)

    @absltest.skipUnless(
        _composite._HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp"
    )
    def test_custom_external_matrix_uses_generic_marker(self) -> None:
        # The GPU Poseidon2Fusion emitter hardcodes the standard M4-circulant
        # MDS, so a non-standard external matrix must NOT take the poseidon2:
        # route — it falls back to the generic zorch.fused_region marker
        # (LoopFusion lowers the real body) to stay correct.
        custom = jnp.arange(16 * 16, dtype=F).reshape(16, 16)
        p = Poseidon2(dataclasses.replace(koalabear16_params(), external_matrix=custom))
        txt = jax.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
        self.assertNotIn("poseidon2:", txt)
        self.assertIn("zorch.fused_region", txt)


if __name__ == "__main__":
    absltest.main()
