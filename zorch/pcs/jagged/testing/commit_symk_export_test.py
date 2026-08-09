# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Symbolic-K export of the jagged commit byte-matches the concrete commit.

The commit side of the jagged PCS is RS-``encode`` (one ``lax.ntt`` over the
column batch) followed by ``smcs.commit`` (poseidon2 Merkle). Both ``ntt`` and
the poseidon2 ops carry VHLO twins, so the whole commit serializes to portable
bytecode and ONE ``frx.export`` binary serves every column count ``K`` — the
commit is recompile-free under symbolic ``K`` with no code change. This locks
that: a single symbolic binary must produce a byte-identical root + codeword to
the concrete commit for two distinct ``K``. Pairs with
``open_symk_export_test`` (the open half). Mont-u32, no tolerances.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array, export
from hash_frx.compression import Compression, CompressionParams
from hash_frx.poseidon2.poseidon2 import Poseidon2
from hash_frx.sponge import Sponge, SpongeParams
from zk_dtypes import koalabear_mont as BF

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.commit.testing.sp1_koalabear16 import koalabear16_params

_LOG_S = 6
_S = 1 << _LOG_S
_BLOWUP = 4
_PRIME = 2013265921  # koalabear


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _u32(x: Array) -> list[int]:
    return np.asarray(frx.lax.bitcast_convert_type(x, fnp.uint32)).reshape(-1).tolist()


class CommitSymkExportTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.smcs = _smcs()
        cls.code = BitReversedReedSolomon(message_len=_S, blowup=_BLOWUP, dtype=BF)

    def _commit(self, message: Array) -> tuple[Array, Array]:
        # The StackedRound build: encode the [K, S] block to a [S*blowup, K]
        # codeword and Merkle-commit it.
        codeword = self.code.encode(message).T
        root, _layers = self.smcs.commit(codeword)
        return root, codeword

    def test_one_binary_byte_matches_concrete_for_every_k(self) -> None:
        (k,) = export.symbolic_shape("k", constraints=["k <= 16", "k >= 9"])
        msg_abs = frx.ShapeDtypeStruct((k, _S), BF)
        exported = export.export(frx.jit(self._commit))(msg_abs)

        for k_val in (12, 16):
            rng = np.random.default_rng(k_val)
            message = fnp.asarray(rng.integers(0, _PRIME, (k_val, _S), np.uint32)).view(
                BF
            )
            ref_root, ref_cw = self._commit(message)
            got_root, got_cw = exported.call(message)
            self.assertEqual(_u32(ref_root), _u32(got_root), f"root K={k_val}")
            self.assertEqual(_u32(ref_cw), _u32(got_cw), f"codeword K={k_val}")


if __name__ == "__main__":
    absltest.main()
