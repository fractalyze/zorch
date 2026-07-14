"""Poseidon2 koalabear-16 byte-matches the honest Plonky3 reference."""

from __future__ import annotations

import dataclasses
import functools

import frx
import frx.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.poseidon2 import (
    POSEIDON2_MARKER,
    POSEIDON2_MARKER_VERSION,
    Poseidon2,
    _permute_body,
)
from zorch.hash.poseidon2.testing.koalabear16 import (
    KOALABEAR16_EXPECTED,
    KOALABEAR16_POSEIDON2_ATTRS,
    koalabear16_params,
    koalabear16_perm,
    koalabear16_scaled_perm,
)
from zorch.hash.sponge import SPONGE_HASH_MARKER, Sponge, SpongeParams
from zorch.testkit.jit_cache import assert_single_trace


class Poseidon2Koalabear16Test(absltest.TestCase):
    def test_permute_byte_matches_plonky3(self) -> None:
        p = koalabear16_perm()
        out = p.permute(jnp.arange(16, dtype=F))
        self.assertTrue(bool(jnp.array_equal(out, KOALABEAR16_EXPECTED)))

    def test_vmap_batch_matches(self) -> None:
        p = koalabear16_perm()
        x = jnp.arange(16, dtype=F)
        batch = frx.vmap(p.permute)(jnp.stack([x, x]))
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
        # The standard-MDS permute marks its region "zorch.poseidon2" so Fractal XLA
        # routes it to the dedicated Poseidon2Fusion emitter; the permutation
        # shape rides as composite.attributes — all four ints are required by
        # the Fractal XLA recognizer. W=16, E=4, I=20, alpha=3 for koalabear-16.
        p = koalabear16_perm()
        txt = frx.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
        self.assertEqual(txt.count("stablehlo.composite"), 1, txt)
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn(f'"{POSEIDON2_MARKER}"', composite_line)
        self.assertIn(KOALABEAR16_POSEIDON2_ATTRS, composite_line)
        self.assertIn(f"version = {POSEIDON2_MARKER_VERSION}", composite_line)
        # Exactly the 5 ABI operands [state, ext_init_rc, int_rc, ext_term_rc,
        # diag]. The internal J scale rides as the `internal_j_scale` attribute,
        # not a 6th operand (issue #440). A closed-over external matrix would be
        # lifted to a leading 6th operand (frx.lax.composite prepends consts) and
        # break the Poseidon2Fusion operand ABI — the e2e GPU failure this guards.
        operands = composite_line.split(f'"{POSEIDON2_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 5, composite_line)

    def test_non_identity_j_scale_stays_five_operands_canonical(self) -> None:
        # A non-identity J scale must ride as the CANONICAL attribute value and
        # still emit exactly 5 operands (materialized in-trace, never lifted to a
        # 6th operand). koalabear16_scaled_perm's scale is R⁻¹: canonical
        # 1057030144, Montgomery STORAGE 1 — so the attribute must read 1057030144,
        # not the storage 1 a raw-bits/canonical mixup would emit (fractalyze/xla#206).
        p = koalabear16_scaled_perm()
        txt = frx.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
        composite_line = next(
            ln for ln in txt.splitlines() if "stablehlo.composite" in ln
        )
        self.assertIn("internal_j_scale = 1057030144 : i64", composite_line)
        operands = composite_line.split(f'"{POSEIDON2_MARKER}"')[1].split("{")[0]
        self.assertEqual(operands.count("%"), 5, composite_line)

    def test_vmap_permute_keeps_dedicated_marker(self) -> None:
        # If frx's composite batching rule regresses, vmap silently falls back to
        # generic loop fusion — the dedicated kernel lost with no error.
        p = koalabear16_perm()
        states = jnp.arange(5 * 16, dtype=F).reshape(5, 16)
        txt = frx.jit(lambda x: frx.vmap(p.permute)(x)).lower(states).as_text()
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
        txt = frx.jit(p.permute).lower(jnp.arange(16, dtype=F)).as_text()
        self.assertNotIn(POSEIDON2_MARKER, txt)
        self.assertIn("zorch.fused_region", txt)

    def test_non_plonky3_m4_takes_dedicated_route(self) -> None:
        # A non-default but M4-block-structured matrix (here the HorizenLabs
        # reference M4) must take the dedicated zorch.poseidon2 route, carrying its
        # own M4 as the external_m4 attribute — not fall back to the generic
        # marker — so the dedicated emitter serves any M4 without a special case.
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
        txt = frx.jit(p.permute).lower(jnp.arange(w, dtype=F)).as_text()
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
        s = Sponge(p, SpongeParams(rate=8, out=8))
        txt = frx.jit(lambda x: s.hash(x)).lower(jnp.arange(w, dtype=F)).as_text()
        self.assertIn(f'"{SPONGE_HASH_MARKER}"', txt)
        self.assertIn(
            "external_m4 = dense<[5, 7, 1, 3, 4, 6, 1, 1, 1, 3, 5, 7, 1, 1, 4, 6]> :"
            " tensor<16xi64>",
            txt,
        )


if __name__ == "__main__":
    absltest.main()
