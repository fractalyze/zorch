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
    num_vars: int,
    k_whir: int,
    num_queries: int = 3,
    blowup: int = 2,
    rate_increase: bool = False,
) -> tuple[WhirProver, WhirVerifier]:
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    code = ReedSolomon(message_len=1 << num_vars, blowup=blowup, dtype=F)
    tree = StridedMerkleTree(sponge, comp, rows_per_query=1 << k_whir)
    params = WhirParams(
        k_whir=k_whir,
        num_queries=(num_queries,) * (num_vars // k_whir),
        rate_increase=rate_increase,
    )
    return WhirProver(code, tree, params), WhirVerifier(code, tree, params)


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


class WhirTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # (num_vars, k_whir, num_polys, rate_increase)
        ("single_m1round_k2", 2, 2, 1, False),  # degenerate: 1 round, 1 poly
        ("single_m2rounds_k2", 4, 2, 1, False),  # re-encode + OOD + EF-limb cosets
        ("single_k1_three_rounds", 3, 1, 1, False),  # one variable folded per round
        ("batch3_m2rounds_k2", 4, 2, 3, False),  # μ-batch across the round machinery
        ("batch5_k1", 3, 1, 5, False),  # μ-batch, fold one variable per round
        # Rate-increasing schedule (SWIRL / openvm-stark-backend): only diverges
        # from constant-rate at k_whir > 1, where the re-encode domain shrinks by
        # 2^1 (not 2^k) per round, so the rate climbs each round.
        ("rate_inc_m2rounds_k2", 4, 2, 1, True),  # one rate-climbing re-encode
        ("rate_inc_m3rounds_k2", 6, 2, 1, True),  # two re-encodes, rising rate
        ("rate_inc_batch3_k2", 4, 2, 3, True),  # μ-batch under rate-increase
        ("rate_inc_k1_matches_const", 3, 1, 1, True),  # k=1 ⇒ == constant-rate
    )
    def test_open_verify_roundtrip(
        self, num_vars: int, k_whir: int, num_polys: int, rate_increase: bool
    ) -> None:
        prover, verifier = _whir(num_vars, k_whir, rate_increase=rate_increase)
        polys = [rand_field(i, (1 << num_vars,), F) for i in range(num_polys)]
        z = rand_ext_field(99, (num_vars,), F, EF)
        root, prover_data = prover.commit(polys)
        values, proof, _ = prover.open(prover_data, [z], _transcript())
        self.assertEqual(values.shape, (num_polys,))
        ok, _ = verifier.verify(root, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_verify_rejects_tampered_final_poly(self) -> None:
        prover, verifier = _whir(num_vars=4, k_whir=2)
        polys = [rand_field(i, (16,), F) for i in range(3)]
        z = rand_ext_field(3, (4,), F, EF)
        root, prover_data = prover.commit(polys)
        values, proof, _ = prover.open(prover_data, [z], _transcript())
        tampered = dataclasses.replace(
            proof, final_poly=proof.final_poly.at[0].add(jnp.ones((), EF))
        )
        ok, _ = verifier.verify(root, [z], values, tampered, _transcript())
        self.assertFalse(bool(ok))

    def test_verify_rejects_tampered_value(self) -> None:
        """A wrong claimed per-column evaluation must not verify."""
        prover, verifier = _whir(num_vars=4, k_whir=2)
        polys = [rand_field(i, (16,), F) for i in range(3)]
        z = rand_ext_field(3, (4,), F, EF)
        root, prover_data = prover.commit(polys)
        values, proof, _ = prover.open(prover_data, [z], _transcript())
        bad = values.at[1].add(jnp.ones((), EF))
        ok, _ = verifier.verify(root, [z], bad, proof, _transcript())
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
