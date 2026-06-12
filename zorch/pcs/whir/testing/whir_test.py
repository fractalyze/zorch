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
from jax import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.strided_merkle import StridedMerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.pcs.whir.config import WhirParams
from zorch.pcs.whir.prover import WhirProver
from zorch.pcs.whir.scheme import EqWhirScheme, WhirScheme
from zorch.pcs.whir.verifier import WhirVerifier
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.transcript import DuplexTranscript, Transcript


@dataclasses.dataclass(frozen=True)
class _MobiusScheme(EqWhirScheme):
    """A non-default `WhirScheme` for the seam test: opens the SWIRL-style
    möbius-eq weight (per-variable kernel K(0)=1−2u, K(1)=u) instead of plain eq.
    Inherits the μ-combine (`combined_f_evals`) unchanged and overrides only the
    three weight-dependent maps. The table mirrors `expand_eq_to_hypercube`'s
    LSB-add convention (so the driver's plain-eq out-of-domain / query weight
    updates stay on the same hypercube), and `final_prefix` is that weight's
    multilinear at the folds — a structurally different functional that round-trips
    only if all hooks are threaded consistently through prover and verifier. Lives
    in the test, not in zorch: möbius is SWIRL glue (the real one ships in the
    openvm consumer)."""

    def _table(self, u: Array) -> Array:
        state = jnp.ones((1,), u.dtype)
        for j in range(u.shape[0]):  # u[j] added as the new LSB, kernel (1−2u, u)
            high = state * u[j]
            low = state * (jnp.ones((), u.dtype) - u[j] - u[j])
            state = jnp.column_stack([low, high]).flatten()
        return state

    def claimed_values(self, mle: Array, z: Array) -> Array:
        return (mle.astype(z.dtype) * self._table(z)[:, None]).sum(0)

    def initial_weight(self, z: Array) -> Array:
        return self._table(z)

    def final_prefix(self, z: Array, alphas: Array) -> Array:
        x = alphas[::-1]  # same z↔fold pairing as eval_eq(z, alphas[::-1])
        one = jnp.ones((), z.dtype)
        return jnp.prod((one - x) * (one - z - z) + x * z)


@dataclasses.dataclass(frozen=True)
class _NoBindEqScheme(EqWhirScheme):
    """Default opening, but `bind` is a no-op — the pattern a consumer uses when
    its outer protocol already bound the commitment in an earlier stage (so WHIR
    must NOT re-absorb it). Round-trips because prover and verifier skip the bind
    symmetrically; proves the `bind` seam is threaded on both sides."""

    def bind(
        self, transcript: Transcript, commitment: Array, values: Array
    ) -> Transcript:
        return transcript


def _whir(
    num_vars: int,
    k_whir: int,
    num_queries: int = 3,
    blowup: int = 2,
    rate_increase: bool = False,
    scheme: WhirScheme | None = None,
    pow_bits: int = 0,
) -> tuple[WhirProver, WhirVerifier]:
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    code = ReedSolomon(message_len=1 << num_vars, blowup=blowup, dtype=F)
    tree = StridedMerkleTree(sponge, comp, rows_per_query=1 << k_whir)
    params = WhirParams(
        k_whir=k_whir,
        num_queries=(num_queries,) * (num_vars // k_whir),
        mu_pow_bits=pow_bits,
        folding_pow_bits=pow_bits,
        query_pow_bits=pow_bits,
        rate_increase=rate_increase,
    )
    scheme = scheme if scheme is not None else EqWhirScheme()
    return WhirProver(code, tree, params, scheme), WhirVerifier(
        code, tree, params, scheme
    )


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

    @parameterized.named_parameters(
        # (num_vars, k_whir, num_polys, rate_increase)
        ("const", 4, 2, 1, False),  # möbius weight, constant-rate schedule
        ("rate_inc", 4, 2, 1, True),  # möbius weight composes with rate-increase
        ("batch", 4, 2, 3, False),  # möbius weight under the μ-batch combine
    )
    def test_open_verify_roundtrip_mobius_scheme(
        self, num_vars: int, k_whir: int, num_polys: int, rate_increase: bool
    ) -> None:
        """A non-default scheme (möbius weight) threads through prover and verifier
        and round-trips — the proof that the injection seam is wired both sides."""
        prover, verifier = _whir(
            num_vars, k_whir, rate_increase=rate_increase, scheme=_MobiusScheme()
        )
        polys = [rand_field(i, (1 << num_vars,), F) for i in range(num_polys)]
        z = rand_ext_field(42, (num_vars,), F, EF)
        root, prover_data = prover.commit(polys)
        values, proof, _ = prover.open(prover_data, [z], _transcript())
        ok, _ = verifier.verify(root, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))

    def test_open_verify_roundtrip_with_grinds(self) -> None:
        """All three proof-of-work grinds active (μ, folding, query) round-trip,
        and a tampered μ witness is rejected. The eager-driver / jitted-island
        structure exists precisely so `grind(pow_bits>0)` runs (it validates on the
        host and cannot be traced) — under the old single-`@jit` `open` this raised
        `TracerBoolConversionError`."""
        prover, verifier = _whir(num_vars=4, k_whir=2, pow_bits=4)
        polys = [rand_field(i, (16,), F) for i in range(3)]
        z = rand_ext_field(7, (4,), F, EF)
        root, prover_data = prover.commit(polys)
        values, proof, _ = prover.open(prover_data, [z], _transcript())
        ok, _ = verifier.verify(root, [z], values, proof, _transcript())
        self.assertTrue(bool(ok))
        w = proof.mu_pow_witness
        tampered = dataclasses.replace(proof, mu_pow_witness=w + jnp.ones((), w.dtype))
        ok, _ = verifier.verify(root, [z], values, tampered, _transcript())
        self.assertFalse(bool(ok))

    def test_open_verify_roundtrip_no_bind_scheme(self) -> None:
        """A scheme whose `bind` is a no-op (the consumer pattern: commitment
        already bound upstream) round-trips, proving the bind seam threads."""
        prover, verifier = _whir(num_vars=4, k_whir=2, scheme=_NoBindEqScheme())
        polys = [rand_field(i, (16,), F) for i in range(2)]
        z = rand_ext_field(11, (4,), F, EF)
        root, prover_data = prover.commit(polys)
        values, proof, _ = prover.open(prover_data, [z], _transcript())
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
