# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The trace commit's fold+bind zone is K-independent: one compile across K.

The fold+bind zone must compile once per (S, blowup, chip-count) and be reused
by every K — it breaks if any K-shaped array reaches the zone (the separator
width rides in as a traced value). The encode + leaf-hash prologue recompiles
per K by design. Byte-parity with the eager commit is ``commit_test``'s job
(``test_jit_matches_eager``); this locks the compile key.
"""

from __future__ import annotations

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as BF

from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.commit.testing.sp1_koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.pcs.jagged.commit import _fold_bind_jit, _prologue_jit, commit_region
from zorch.pcs.jagged.region import JaggedRegion


def _region(heights: tuple[int, ...]) -> JaggedRegion:
    chips = [
        fnp.arange(100 * i, 100 * i + h * 3, dtype=fnp.uint32).reshape(h, 3).view(BF)
        for i, h in enumerate(heights)
    ]
    return JaggedRegion.from_chips(chips, log_stacking_height=3, max_log_row_count=4)


class FoldBindZoneKIndependenceTest(absltest.TestCase):
    def test_fold_bind_zone_compiles_once_across_k(self) -> None:
        perm = Poseidon2(koalabear16_params())
        smcs = SingleMatrixCommitmentScheme(
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
        )
        prologue_before = _prologue_jit._cache_size()
        fold_bind_before = _fold_bind_jit._cache_size()
        # Same chip count, different stacked column counts under S = 8:
        # areas 18 -> K=3 and 27 -> K=4.
        widths = []
        for heights in ((4, 2), (6, 3)):
            region = _region(heights)
            S = 1 << region.log_stacking_height
            widths.append(region.dense.shape[0] // S)
            commit_region(region, smcs, log_blowup=2, jit=True)
        self.assertNotEqual(widths[0], widths[1])  # the test needs two real Ks
        # The prologue pays one compile per K; the fold+bind — the commit's
        # dominant O(depth) compile — is shared.
        self.assertEqual(_prologue_jit._cache_size() - prologue_before, 2)
        self.assertEqual(_fold_bind_jit._cache_size() - fold_bind_before, 1)


if __name__ == "__main__":
    absltest.main()
