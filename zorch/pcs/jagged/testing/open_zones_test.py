# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The stacked open's fold zone is K-independent: one compile across K.

The fold zone must compile once per (S, blowup, num_queries, pow_bits) and be
reused by every K — it breaks if any K-shaped array reaches the zone. The two
K-shaped zones (prologue, component queries) recompile per K by design.
Byte-correctness vs SP1 is ``open_test``'s job; this locks the compile key.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from hash_frx.compression import Compression, CompressionParams
from hash_frx.poseidon2.poseidon2 import Poseidon2
from hash_frx.sponge import Sponge, SpongeParams
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.commit.testing.sp1_koalabear16 import koalabear16_params
from zorch.pcs.jagged.open import (
    StackedRound,
    _open_fold,
    _open_prologue,
    _open_queries,
    stacked_basefold_open,
)
from zorch.transcript import DuplexTranscript

_LOG_S = 4  # stacking height S = 16
_S = 1 << _LOG_S
_BLOWUP = 2
_NUM_QUERIES = 2
_PRIME = 2013265921  # koalabear


class FoldZoneKIndependenceTest(absltest.TestCase):
    def test_fold_zone_compiles_once_across_k(self) -> None:
        perm = Poseidon2(koalabear16_params())
        smcs = SingleMatrixCommitmentScheme(
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
        )
        code = BitReversedReedSolomon(message_len=_S, blowup=_BLOWUP, dtype=BF)
        rng = np.random.default_rng(0)

        fold_before = _open_fold._cache_size()
        prologue_before = _open_prologue._cache_size()
        queries_before = _open_queries._cache_size()
        for k in (3, 5):
            block = fnp.asarray(rng.integers(0, _PRIME, (k, _S), np.uint32)).view(BF)
            codeword = code.encode(block).T
            _root, digest_layers = smcs.commit(codeword)
            rd = StackedRound(block=block, digest_layers=digest_layers)
            z_final = fnp.asarray(
                rng.integers(0, _PRIME, (_LOG_S * 4,), np.uint32)
            ).view(EF)
            dense_eval = fnp.asarray(rng.integers(0, _PRIME, (4,), np.uint32)).view(EF)[
                0
            ]
            proof, _ = stacked_basefold_open(
                smcs,
                code,
                [rd],
                z_final,
                dense_eval,
                _LOG_S,
                num_queries=_NUM_QUERIES,
                pow_bits=0,
                transcript=DuplexTranscript.new(perm, rate=8),
            )
            self.assertEqual(proof.batch_evals[0].shape, (k,))
        # The K-shaped zones pay one compile per K; the fold zone — the open's
        # dominant codegen — is shared.
        self.assertEqual(_open_fold._cache_size() - fold_before, 1)
        self.assertEqual(_open_prologue._cache_size() - prologue_before, 2)
        self.assertEqual(_open_queries._cache_size() - queries_before, 2)


if __name__ == "__main__":
    absltest.main()
