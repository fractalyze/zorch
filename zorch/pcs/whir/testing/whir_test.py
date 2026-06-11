# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""WhirProver.open <-> WhirVerifier.verify round-trip on ReedSolomon + koalabear +
DuplexTranscript.

Correctness is the self-test: a freshly opened proof must verify against the same
commitment (`ok == True`), and a tampered proof must not. Exercises the full round
driver — sumcheck folds, per-round re-encode + out-of-domain sampling, strided
query openings, the binary k-fold consistency, and the final constraint — across
several `(num_variables, k_whir)` shapes. No golden vector, no byte-match.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.pcs.whir.config import WhirParams
from zorch.pcs.whir.prover import WhirProver
from zorch.pcs.whir.verifier import WhirVerifier
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.transcript import DuplexTranscript


def _whir(
    num_vars: int, k_whir: int, num_queries: int = 3, blowup: int = 2
) -> tuple[WhirProver, WhirVerifier]:
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    code = ReedSolomon(message_len=1 << num_vars, blowup=blowup, dtype=F)
    tree = StridedMerkleTree(sponge, comp, rows_per_query=1 << k_whir)
    params = WhirParams(
        k_whir=k_whir, num_queries=(num_queries,) * (num_vars // k_whir)
    )
    return WhirProver(code, tree, params), WhirVerifier(code, tree, params)


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


class WhirTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("m1round_k2", 2, 2),  # degenerate single round (no re-encode/OOD)
        ("m2rounds_k2", 4, 2),  # re-encode + OOD + weight update + EF-limb cosets
        ("k1_three_rounds", 3, 1),  # one variable folded per round
    )
    def test_open_verify_roundtrip(self, num_vars: int, k_whir: int) -> None:
        prover, verifier = _whir(num_vars, k_whir)
        poly = rand_field(0, (1 << num_vars,), F)
        z = rand_ext_field(1, (num_vars,), F, EF)
        root, prover_data = prover.commit([poly])
        value, proof, _ = prover.open(prover_data, [z], _transcript())
        ok, _ = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_verify_rejects_tampered_final_poly(self) -> None:
        prover, verifier = _whir(num_vars=4, k_whir=2)
        poly = rand_field(2, (16,), F)
        z = rand_ext_field(3, (4,), F, EF)
        root, prover_data = prover.commit([poly])
        value, proof, _ = prover.open(prover_data, [z], _transcript())
        tampered = dataclasses.replace(
            proof, final_poly=proof.final_poly.at[0].add(jnp.ones((), EF))
        )
        ok, _ = verifier.verify(root, [z], value, tampered, _transcript())
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
