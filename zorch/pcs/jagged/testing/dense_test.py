# zorch/pcs/jagged/testing/dense_test.py
"""from_blocks — column-major pack to a static M_max=2^tier dense buffer —
plus the fail-loud `validate_layout` contract on untrusted layouts."""
from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.pcs.jagged.dense import JaggedLayout, from_blocks, validate_layout
from zorch.pcs.jagged.poly import build_jagged_layout
from zorch.pcs.jagged.prover import _indicator_inputs
from zorch.utils.bits import log2_ceil_usize


def _oracle_pack(blocks: list[object], log_s: int) -> tuple[np.ndarray, int]:
    """Independent reference for `from_blocks` — column-major flatten + concat +
    zero-pad to 2^tier. Mirrors `from_blocks`'s tier formula; keep in sync."""
    parts = [np.ravel(np.asarray(b), order="F") for b in blocks]
    flat = np.concatenate(parts) if parts else np.zeros(0)
    total = flat.shape[0]
    log_m = max(log2_ceil_usize(total) + 1, log_s)
    m_max = 1 << log_m
    padded = np.concatenate([flat, np.zeros(m_max - total, dtype=flat.dtype)])
    return padded, log_m


class FromBlocksTest(unittest.TestCase):
    def test_pack_matches_oracle(self) -> None:
        blocks = [
            jnp.arange(6, dtype=F).reshape(3, 2),
            jnp.arange(4, dtype=F).reshape(2, 2),
        ]
        packed, layout = from_blocks(blocks, log_stacking_height=1)
        exp, log_m = _oracle_pack(blocks, 1)
        self.assertEqual(np.asarray(packed).tolist(), exp.tolist())
        self.assertEqual(layout.log_m, log_m)
        self.assertEqual(layout.heights, (3, 2))
        self.assertEqual(layout.widths, (2, 2))
        self.assertEqual(layout.S * layout.K, layout.m_max)  # [S,K] tiles M_max

    def test_padding_invariance_prefix(self) -> None:
        # Single column: column-major order == row-major order, so prefix is
        # just [0,1,2,3,4,5]; padding zeros fill the rest.
        blocks = [jnp.arange(6, dtype=F).reshape(6, 1)]
        packed, layout = from_blocks(blocks, log_stacking_height=0)
        self.assertEqual(np.asarray(packed)[:6].tolist(), list(range(6)))
        self.assertTrue(all(x == 0 for x in np.asarray(packed)[6:].tolist()))

    def test_empty_blocks_raises(self) -> None:
        with self.assertRaises(ValueError):
            from_blocks([], log_stacking_height=0)

    def test_negative_stacking_height_raises(self) -> None:
        with self.assertRaises(ValueError):
            from_blocks([jnp.arange(2, dtype=F).reshape(2, 1)], log_stacking_height=-1)

    def test_tier_matches_indicator_n_d(self) -> None:
        # The opening seam requires cfg.n_d == layout.log_m; both must derive
        # the same tier from the same total area so the constraint holds by
        # construction whenever log_s <= n_d.
        blocks = [
            jnp.arange(6, dtype=F).reshape(3, 2),
            jnp.arange(3, dtype=F).reshape(3, 1),
        ]
        _, layout = from_blocks(blocks, log_stacking_height=2)
        _, cfg = build_jagged_layout([3, 3, 3], l_max=3, n_r=2, dtype=F)
        self.assertEqual(layout.log_m, cfg.n_d)

    def test_k_gt_1_reachable(self) -> None:
        # area 9 -> tier log2_ceil(9)+1 = 5; log_s=3 leaves K = 2^(5-3) = 4.
        blocks = [jnp.arange(9, dtype=F).reshape(9, 1)]
        _, layout = from_blocks(blocks, log_stacking_height=3)
        self.assertEqual(layout.log_m, 5)
        self.assertEqual(layout.K, 4)

    def test_stacking_height_dominates_tier(self) -> None:
        # total_area=1 but log_stacking_height=3 floors the tier: log_m=3.
        packed, layout = from_blocks(
            [jnp.arange(1, dtype=F).reshape(1, 1)], log_stacking_height=3
        )
        self.assertEqual(layout.log_m, 3)
        self.assertEqual(layout.m_max, 8)
        self.assertEqual(layout.S, 8)
        self.assertEqual(layout.K, 1)
        self.assertEqual(int(packed.shape[0]), 8)


def _layout(
    heights: list[int], widths: list[int], log_m: int = 6, log_s: int = 2
) -> JaggedLayout:
    return JaggedLayout(
        log_m=log_m,
        log_s=log_s,
        heights=tuple(heights),
        widths=tuple(widths),
        dtype=F,
    )


class ValidateLayoutTest(absltest.TestCase):
    """The verifier consumes the layout from an untrusted statement; malformed
    counts must raise before any arithmetic — see docs/jagged.md § Verifier
    input contract."""

    def test_valid_layout_passes(self) -> None:
        validate_layout(_layout([3, 0, 5], [2, 4, 1]))

    def test_negative_height_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "height"):
            validate_layout(_layout([3, -1], [2, 2]))

    def test_negative_width_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "width"):
            validate_layout(_layout([3, 1], [2, -2]))

    def test_mismatched_height_width_lengths_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "heights.*widths"):
            validate_layout(_layout([3, 1], [2]))

    def test_empty_blocks_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "block"):
            validate_layout(_layout([], []))

    def test_zero_total_area_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "area"):
            validate_layout(_layout([0, 0], [1, 1]))

    def test_count_above_capacity_rejected(self) -> None:
        # Width-0 block hides an oversized height from the area bound.
        with self.assertRaisesRegex(ValueError, "height"):
            validate_layout(_layout([2, 1 << 40], [2, 0]))

    def test_block_count_above_capacity_rejected(self) -> None:
        # Zero-area blocks grow the block count without growing the area.
        with self.assertRaisesRegex(ValueError, "block count"):
            validate_layout(_layout([2] + [0] * 5, [1] * 6))

    def test_area_tier_above_safe_bound_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tier"):
            validate_layout(_layout([1 << 30], [1]))

    def test_log_s_above_log_m_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "log_s"):
            validate_layout(_layout([4], [4], log_m=4, log_s=5))

    def test_negative_log_s_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "log_s"):
            validate_layout(_layout([4], [4], log_s=-1))

    def test_indicator_inputs_rejects_malformed_structure(self) -> None:
        # The shared consumption path (commit/open/verify) must fire the
        # validation before any bit tensor is built.
        with self.assertRaisesRegex(ValueError, "height"):
            _indicator_inputs(_layout([2, -1], [2, 2]))


if __name__ == "__main__":
    absltest.main()
