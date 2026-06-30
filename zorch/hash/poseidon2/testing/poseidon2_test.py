"""Poseidon2 koalabear-16 byte-matches the honest Plonky3 reference."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.poseidon2 import (
    POSEIDON2_MARKER,
    POSEIDON2_MARKER_VERSION,
    SPONGE_HASH_MARKER,
    Poseidon2,
    _permute_body,
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

    def test_permute_reuses_one_trace_across_instances(self) -> None:
        # Freshly built same-params permutations must share one module-level
        # permute trace — the static key compares by value (#214). Without the
        # zone, every composite emission re-traced the permutation body, which
        # dominated the PCS first-trace-per-config cost (#216).
        x = jnp.arange(16, dtype=F)
        calls = [functools.partial(koalabear16_perm().permute, x) for _ in (0, 1)]
        assert_single_trace(self, _permute_body, calls)

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

    def test_vmap_permute_keeps_dedicated_marker(self) -> None:
        # If jax's composite batching rule regresses, vmap silently falls back to
        # generic loop fusion — the dedicated kernel lost with no error.
        p = koalabear16_perm()
        states = jnp.arange(5 * 16, dtype=F).reshape(5, 16)
        txt = jax.jit(lambda x: jax.vmap(p.permute)(x)).lower(states).as_text()
        comp = [ln for ln in txt.splitlines() if "stablehlo.composite" in ln]
        self.assertEqual(len(comp), 1, txt)  # one composite over the whole batch
        self.assertIn(f'"{POSEIDON2_MARKER}"', comp[0])  # dedicated, not generic
        self.assertIn("5x16x", comp[0])  # batched operand (b=5), not per-element

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

    def test_sponge_hash_marker_carries_external_m4(self) -> None:
        # The fused sponge kernel applies the matrix's M4 like the permute kernel,
        # so the sponge marker must carry external_m4 too — including a non-default
        # M4 (here the HorizenLabs reference), not a hardcoded one.
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
        txt = (
            jax.jit(lambda x: p.sponge_hash(x, 8, 8))
            .lower(jnp.arange(w, dtype=F))
            .as_text()
        )
        self.assertIn(f'"{SPONGE_HASH_MARKER}"', txt)
        self.assertIn(
            "external_m4 = dense<[5, 7, 1, 3, 4, 6, 1, 1, 1, 3, 5, 7, 1, 1, 4, 6]> :"
            " tensor<16xi64>",
            txt,
        )


if __name__ == "__main__":
    absltest.main()
