# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Symbolic-K export of the stacked BaseFold open byte-matches the concrete open.

``stacked_basefold_open(rlc_bits=...)`` makes the open shape-polymorphic in each
round's column count ``K`` (the per-column evals run under ``vmap`` and the RLC
samples a static ``rlc_bits`` challenges), so ONE ``jax.export`` binary serves
every ``K`` in a power-of-2 column bracket — killing the per-shard open
recompile. This locks that path: a single symbolic binary (bracket ``B=4``,
``K in [9, 16]``) must produce byte-identical proofs to the concrete open for
two distinct ``K`` in the bracket. The concrete open is the SP1-byte-matched
reference (``open_test``), so equivalence to it transitively pins the symbolic
path to SP1. Mont-u32, no tolerances.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import Array, export
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.commit.testing.sp1_koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.pcs.jagged.open import (
    StackedOpenProof,
    StackedRound,
    stacked_basefold_open,
)
from zorch.transcript import DuplexTranscript, GrindingTranscript

_LOG_S = 6  # stacking height S = 64
_S = 1 << _LOG_S
_BLOWUP = 4
_NUM_QUERIES = 4
_RLC_BITS = 4  # bracket: 2^4 = 16 columns => K in [9, 16]
_PRIME = 2013265921  # koalabear


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _u32(x: Array) -> list[int]:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint32)).reshape(-1).tolist()


class SymbolicKOpenExportTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.smcs = _smcs()
        cls.perm = Poseidon2(koalabear16_params())
        cls.code = BitReversedReedSolomon(message_len=_S, blowup=_BLOWUP, dtype=BF)
        cls.rng = np.random.default_rng(0)

    def _make_round(self, k: int) -> StackedRound:
        block = jnp.asarray(self.rng.integers(0, _PRIME, (k, _S), np.uint32)).view(BF)
        mle = block.T
        codeword = self.code.encode(block).T
        _root, digest_layers = self.smcs.commit(codeword)
        return StackedRound(mle=mle, codeword=codeword, digest_layers=digest_layers)

    def _export_symbolic(self) -> export.Exported:
        # Abstract round over a symbolic column count constrained to the bracket;
        # the digest tree shape is K-independent (its leaf count is S*blowup), so
        # a concrete template supplies it.
        template = self._make_round(12)
        (k,) = export.symbolic_shape(
            "k", constraints=[f"k <= {1 << _RLC_BITS}", "k >= 9"]
        )
        rd_abs = StackedRound(
            mle=jax.ShapeDtypeStruct((_S, k), BF),
            codeword=jax.ShapeDtypeStruct((_S * _BLOWUP, k), BF),
            digest_layers=[
                jax.ShapeDtypeStruct(layer.shape, layer.dtype)
                for layer in template.digest_layers
            ],
        )
        z_abs = jax.ShapeDtypeStruct((_LOG_S,), EF)
        de_abs = jax.ShapeDtypeStruct((), EF)
        tr_abs = jax.tree_util.tree_map(
            lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype),
            DuplexTranscript.new(self.perm, rate=8),
        )

        def fn(
            rounds: Sequence[StackedRound],
            z: Array,
            dense_eval: Array,
            transcript: DuplexTranscript,
        ) -> tuple[StackedOpenProof, GrindingTranscript]:
            return stacked_basefold_open(
                self.smcs,
                self.code,
                rounds,
                z,
                dense_eval,
                _LOG_S,
                num_queries=_NUM_QUERIES,
                pow_bits=0,
                rlc_bits=_RLC_BITS,
                transcript=transcript,
            )

        return export.export(jax.jit(fn))([rd_abs], z_abs, de_abs, tr_abs)

    def test_one_binary_byte_matches_concrete_for_every_k(self) -> None:
        exported = self._export_symbolic()
        for k in (12, 16):
            rd = self._make_round(k)
            z = jnp.asarray(
                self.rng.integers(0, _PRIME, (_LOG_S * 4,), np.uint32)
            ).view(EF)
            dense_eval = jnp.asarray(
                self.rng.integers(0, _PRIME, (4,), np.uint32)
            ).view(EF)[0]

            ref, _ = stacked_basefold_open(
                self.smcs,
                self.code,
                [rd],
                z,
                dense_eval,
                _LOG_S,
                num_queries=_NUM_QUERIES,
                pow_bits=0,
                transcript=DuplexTranscript.new(self.perm, rate=8),
            )
            got, _ = exported.call(
                [rd], z, dense_eval, DuplexTranscript.new(self.perm, rate=8)
            )

            ref_leaves = jax.tree_util.tree_leaves(ref)
            got_leaves = jax.tree_util.tree_leaves(got)
            self.assertEqual(len(ref_leaves), len(got_leaves), f"leaf count K={k}")
            for i, (a, b) in enumerate(zip(ref_leaves, got_leaves)):
                self.assertEqual(_u32(a), _u32(b), f"leaf {i} diverged at K={k}")


if __name__ == "__main__":
    absltest.main()
