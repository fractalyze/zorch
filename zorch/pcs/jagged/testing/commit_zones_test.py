# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The trace commit's fold zone keys on the leaf count alone: one compile
across K *and* across chip counts.

The fold zone must compile once per (S, blowup) and be reused by every shard —
it breaks if any K-shaped array reaches it (the separator width rides in as a
traced value) or any counts-shaped array does (the root/structure binds live in
their own tail zone; folding them in re-paid the whole O(depth) graph per chip
count). The encode + leaf-hash prologue recompiles per K by design; the bind
tail recompiles per chip count by design. Byte-parity with the eager commit is
``commit_test``'s job (``test_jit_matches_eager``); this locks the compile keys.
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
from zorch.pcs.jagged.commit import _bind_jit, _fold_jit, _prologue_jit, commit_region
from zorch.pcs.jagged.region import JaggedRegion


def _region(heights: tuple[int, ...]) -> JaggedRegion:
    chips = [
        fnp.arange(100 * i, 100 * i + h * 3, dtype=fnp.uint32).reshape(h, 3).view(BF)
        for i, h in enumerate(heights)
    ]
    return JaggedRegion.from_chips(chips, log_stacking_height=3, max_log_row_count=4)


class FoldZoneLeafCountKeyTest(absltest.TestCase):
    def test_fold_zone_compiles_once_across_k_and_chip_counts(self) -> None:
        perm = Poseidon2(koalabear16_params())
        smcs = SingleMatrixCommitmentScheme(
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
        )
        prologue_before = _prologue_jit._cache_size()
        fold_before = _fold_jit._cache_size()
        bind_before = _bind_jit._cache_size()
        # Under S = 8: two chip counts and two stacked column counts —
        # (4, 2) area 18 -> K=3, (6, 3) area 27 -> K=4, (4, 2, 3) area 27 ->
        # K=4 again but a third chip.
        widths = []
        for heights in ((4, 2), (6, 3), (4, 2, 3)):
            region = _region(heights)
            S = 1 << region.log_stacking_height
            widths.append(region.dense.shape[0] // S)
            commit_region(region, smcs, log_blowup=2, jit=True)
        self.assertNotEqual(widths[0], widths[1])  # the test needs two real Ks
        self.assertEqual(widths[1], widths[2])  # ... and a K reuse across chips
        # The prologue pays one compile per K; the bind tail one per chip
        # count; the fold — the commit's dominant O(depth) compile — is shared
        # by all three regions (same S*blowup leaf count).
        self.assertEqual(_prologue_jit._cache_size() - prologue_before, 2)
        self.assertEqual(_bind_jit._cache_size() - bind_before, 2)
        self.assertEqual(_fold_jit._cache_size() - fold_before, 1)


if __name__ == "__main__":
    absltest.main()
