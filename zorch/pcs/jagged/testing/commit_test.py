# zorch/pcs/jagged/testing/commit_test.py
"""JaggedPcsProver.commit — e2e commit, the AOT compile-once property, and that
the structure binding actually moves the root."""

from __future__ import annotations

import functools
import unittest
from collections.abc import Callable

import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.basefold.prover import BasefoldProver
from zorch.pcs.jagged.prover import JaggedPcsProver, _commit_device
from zorch.testkit.jit_cache import assert_single_trace


def _jagged_pcs(
    log_s: int = 2, blowup: int = 2, num_queries: int = 4
) -> JaggedPcsProver:
    S = 1 << log_s
    rs = ReedSolomon(message_len=S, blowup=blowup, dtype=F)
    sponge, comp, tree = koalabear16_merkle()
    return JaggedPcsProver(BasefoldProver(rs, tree, num_queries), sponge, comp)


def _blocks(heights: list[int]) -> list[jnp.ndarray]:
    return [jnp.arange(h, dtype=F).reshape(h, 1) for h in heights]


class JaggedPcsTest(unittest.TestCase):
    def test_commit_returns_root_and_prover_data(self) -> None:
        c = _jagged_pcs()
        commitment, pdata = c.commit(_blocks([2, 2]), log_stacking_height=2)
        self.assertEqual(commitment.shape, (8,))  # koalabear16 digest width
        self.assertTrue(hasattr(pdata.basefold_prover_data, "digest_layers"))

    def test_compile_once_per_tier_across_instances(self) -> None:
        # Three height vectors, all total_area == 14 -> same tier log_m=5,
        # and all have the same block count (2), so only heights vary within
        # the tier — confirming that height variation alone does not retrace.
        # The freshly built same-config prover and the prover differing only
        # in num_queries (which commit never reads) must also hit the same
        # module-level zone: its static keys compare by value (#214).
        calls: list[Callable[[], object]] = []
        for c in (_jagged_pcs(), _jagged_pcs(), _jagged_pcs(num_queries=8)):
            for hv in ([5, 9], [7, 7], [10, 4]):
                calls.append(
                    functools.partial(c.commit, _blocks(hv), log_stacking_height=2)
                )
        assert_single_trace(self, _commit_device, calls)

    def test_binding_moves_root(self) -> None:
        c = _jagged_pcs()
        # Identical packed dense content ([0,1,2,3]), different block structure:
        # one height-4 block vs two height-2 blocks. Only the structure binding
        # differs, so the root must move — proving the bind is not a no-op.
        one_block = [jnp.arange(4, dtype=F).reshape(4, 1)]
        two_blocks = [
            jnp.arange(2, dtype=F).reshape(2, 1),
            jnp.arange(2, 4, dtype=F).reshape(2, 1),
        ]
        a, _ = c.commit(one_block, log_stacking_height=2)
        b, _ = c.commit(two_blocks, log_stacking_height=2)
        self.assertNotEqual(a.tolist(), b.tolist())


if __name__ == "__main__":
    unittest.main()
