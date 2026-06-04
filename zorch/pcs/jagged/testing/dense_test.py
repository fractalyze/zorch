# zorch/pcs/jagged/testing/dense_test.py
"""from_blocks — column-major pack to a static M_max=2^tier dense buffer."""
from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np
from zk_dtypes import koalabear_mont as F

from zorch.pcs.jagged.dense import from_blocks
from zorch.utils.bits import log2_ceil_usize


def _oracle_pack(blocks: list[object], log_s: int) -> tuple[np.ndarray, int]:
    """Independent reference for `from_blocks` — column-major flatten + concat +
    zero-pad to 2^tier. Mirrors `from_blocks`'s tier formula; keep in sync."""
    parts = [np.ravel(np.asarray(b), order="F") for b in blocks]
    flat = np.concatenate(parts) if parts else np.zeros(0)
    total = flat.shape[0]
    log_m = max(log2_ceil_usize(total), log_s)
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


if __name__ == "__main__":
    unittest.main()
