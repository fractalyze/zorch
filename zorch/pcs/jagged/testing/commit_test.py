# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Trace-commit structure: determinism, structure binding, and parity with the
inline commit the stacked open consumes.

``commit_region`` is the commit half of the jagged PCS — it must produce exactly
the ``(mle, codeword, digest_layers)`` a ``StackedRound`` carries, so the open
(byte-matched against SP1 in ``open_test``) accepts what it commits. The
structure-hash byte-match against the SP1 reference dump is the rsp-scale test;
here we pin the local invariants plus parity on the vendored gpu_fibonacci prep
region.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from zk_dtypes import koalabear_mont as BF

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.commit.testing.sp1_koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.pcs.jagged.commit import commit_region
from zorch.pcs.jagged.region import JaggedRegion

_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"
_ZC_INPUTS = Path(__file__).parent / "testdata" / "zerocheck_dense"


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _u32(a: Array) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, fnp.uint32)).reshape(-1)


def _from_u32(u32: Any, dtype: Any) -> Array:
    return frx.lax.bitcast_convert_type(fnp.asarray(u32, dtype=fnp.uint32), dtype)


def _raw_area(round_meta: dict[str, Any]) -> int:
    return sum(
        int(r) * int(c)
        for r, c in zip(
            round_meta["row_counts"], round_meta["column_counts"], strict=True
        )
    )


def _region(heights: tuple[int, ...] = (4, 2)) -> JaggedRegion:
    chips = [
        fnp.arange(100 * i, 100 * i + h * 3, dtype=fnp.uint32).reshape(h, 3).view(BF)
        for i, h in enumerate(heights)
    ]
    return JaggedRegion.from_chips(chips, log_stacking_height=3, max_log_row_count=4)


class CommitRegionTest(absltest.TestCase):
    def test_commitment_shape_and_determinism(self) -> None:
        smcs = _smcs()
        c1, data1 = commit_region(_region(), smcs, log_blowup=2)
        c2, _ = commit_region(_region(), smcs, log_blowup=2)
        self.assertEqual(c1.shape, (8,))
        self.assertEqual(c1.dtype, BF)
        self.assertTrue(bool(fnp.all(c1 == c2)))
        self.assertEqual(data1.dense.shape, _region().dense.shape)
        self.assertNotEmpty(data1.digest_layers)

    def test_jit_matches_eager(self) -> None:
        """The @jit zone exists for memory, not semantics — the commitment and
        every retained prover-data leaf must be byte-identical to eager."""
        smcs = _smcs()
        eager = commit_region(_region(), smcs, log_blowup=2)
        jitted = commit_region(_region(), smcs, log_blowup=2, jit=True)
        for le, lj in zip(frx.tree.leaves(eager), frx.tree.leaves(jitted), strict=True):
            np.testing.assert_array_equal(np.asarray(le), np.asarray(lj))

    def test_structure_binding_separates_same_dense(self) -> None:
        """Identical dense bytes split into different chips must commit
        differently — that is what the structure hash is for."""
        smcs = _smcs()
        flat = fnp.arange(24, dtype=fnp.uint32).view(BF)
        a = JaggedRegion.from_chips(
            [flat.reshape(8, 3).view(BF)], log_stacking_height=3, max_log_row_count=4
        )
        b = JaggedRegion.from_chips(
            [flat[:12].reshape(4, 3).view(BF), flat[12:].reshape(4, 3).view(BF)],
            log_stacking_height=3,
            max_log_row_count=4,
        )
        ca, _ = commit_region(a, smcs, log_blowup=2)
        cb, _ = commit_region(b, smcs, log_blowup=2)
        self.assertFalse(bool(fnp.all(ca == cb)))

    def test_unaligned_dense_raises(self) -> None:
        # Bypasses from_chips (which pads by construction) to hit the guard.
        bad = JaggedRegion(
            dense=fnp.zeros(10, dtype=BF),
            chip_starts=(0, 10),
            row_counts=(2, 8, 2),
            column_counts=(5, 0, 1),
            log_stacking_height=3,
        )
        with self.assertRaises(ValueError):
            commit_region(bad, _smcs(), log_blowup=2)

    def test_blowup_changes_commitment(self) -> None:
        smcs = _smcs()
        c2, _ = commit_region(_region(), smcs, log_blowup=2)
        c1, _ = commit_region(_region(), smcs, log_blowup=1)
        self.assertFalse(bool(fnp.all(c1 == c2)))

    def test_matches_inline_round_commit(self) -> None:
        """The retained ``(mle, digest_layers)`` and the shape-bound root are
        byte-identical to the inline encode + row-major SMCS commit a
        ``StackedRound`` is built from (open_test's ``build_round``). The commit
        reads the codeword column-major; the open's view is its leaf-major
        transpose — same leaf content, identical Merkle tree."""
        smcs = _smcs()
        region = _region(heights=(6, 3))
        S = 1 << region.log_stacking_height
        _bound, data = commit_region(region, smcs, log_blowup=2)

        dense = region.dense
        k = dense.shape[0] // S
        code = BitReversedReedSolomon(message_len=S, blowup=1 << 2, dtype=BF)
        mle_ref = dense.reshape(k, S).T
        codeword_ref = code.encode(dense.reshape(k, S)).T
        root_ref, layers_ref = smcs.commit(codeword_ref)  # row-major leaf view

        np.testing.assert_array_equal(_u32(data.mle), _u32(mle_ref))
        np.testing.assert_array_equal(_u32(data.smcs_commitment), _u32(root_ref))
        self.assertEqual(len(data.digest_layers), len(layers_ref))
        for got, ref in zip(data.digest_layers, layers_ref, strict=True):
            np.testing.assert_array_equal(_u32(got), _u32(ref))

    def test_gpu_fibonacci_prep_round_parity(self) -> None:
        """Commit the vendored prep region and match the inline round commit on
        real shard data — the same dense buffer ``open_test`` opens against."""
        meta = json.loads((_FIXTURE / "meta.json").read_text())
        round_meta = meta["rounds"][0]
        log_s = int(round_meta["log_stacking_height"])
        S = 1 << log_s
        log_blowup = int(meta["basefold"]["log_blowup"])
        smcs = _smcs()

        dense = _from_u32(
            np.load(_ZC_INPUTS / "prep_dense.npy")[: _raw_area(round_meta)], BF
        )
        region = JaggedRegion(
            dense=dense,
            chip_starts=(0, int(dense.shape[0])),
            row_counts=tuple(round_meta["row_counts"]),
            column_counts=tuple(round_meta["column_counts"]),
            log_stacking_height=log_s,
        )
        _bound, data = commit_region(region, smcs, log_blowup=log_blowup)

        k = dense.shape[0] // S
        code = BitReversedReedSolomon(message_len=S, blowup=1 << log_blowup, dtype=BF)
        codeword_ref = code.encode(dense.reshape(k, S)).T
        _root, layers_ref = smcs.commit(codeword_ref)

        np.testing.assert_array_equal(_u32(data.mle), _u32(dense.reshape(k, S).T))
        for got, ref in zip(data.digest_layers, layers_ref, strict=True):
            np.testing.assert_array_equal(_u32(got), _u32(ref))


if __name__ == "__main__":
    absltest.main()
