# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Virtual-to-committed reduction — a product claim reduced to factor claims
the opening then serves, checked against the virtual table built outright.

The virtual value is computed from the product's full table, the factors'
tables broadcast over their slices, and `eval_mle` at the point: a route that
shares nothing with the sumcheck. The chain tests run the reduction and the
opening on one transcript, as a consumer would.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest
from lattice_frx.ring import Eval, RnsRing

from zorch.byte_transcript import ByteHashTranscript, ByteTranscript
from zorch.pcs.akita.challenge import FixedWeightTernary
from zorch.pcs.akita.commit import AkitaCommitter
from zorch.pcs.akita.config import AkitaConfig, Decomposition, SisProfile
from zorch.pcs.akita.prover import AkitaProver
from zorch.pcs.akita.verifier import AkitaVerifier
from zorch.pcs.akita.virtual import VirtualClaim, VirtualProof, shared_vars
from zorch.pcs.stage import OpeningWitness
from zorch.poly.multilinear import eval_mle
from zorch.testkit.byte_hash import HostSha256
from zorch.testkit.random_field import rand_field

_Q = (34359753217, 34359754753)
_D = 64
_LOG_BASE = 8
# Factors 0 and 1 own 2 and 1 leading variables over 6 shared ones — the
# chunked-selector shape; factors 2 and 1 own all of theirs, so they share
# none and no round runs.
_LENS = (256, 128, 64)
_FIELD = zk_dtypes.goldilocks


def _config() -> AkitaConfig:
    modulus = int(zk_dtypes.pfinfo(_FIELD).modulus)
    profile = SisProfile(_D, _Q)
    return AkitaConfig(
        profile=profile,
        inner_decomposition=Decomposition.covering(_LOG_BASE, modulus // 2),
        outer_decomposition=Decomposition.covering(_LOG_BASE, profile.modulus // 2),
        inner_rows=2,
        outer_rows=2,
        message_lens=_LENS,
    )


def _uniform_matrix(
    ring: RnsRing, rng: np.random.Generator, rows: int, cols: int
) -> Eval:
    def element() -> Eval:
        host = np.array(
            [rng.integers(0, q, size=ring.d, dtype=np.uint64) for q in ring.q_moduli],
            dtype=np.uint64,
        )
        return ring.eval_from_host(host)

    return ring.stack(
        [ring.stack([element() for _ in range(cols)]) for _ in range(rows)]
    )


def _transcript() -> ByteTranscript:
    return ByteHashTranscript.new(b"akita-virtual-test", HostSha256())


def _ints(values: frx.Array) -> list[int]:
    return [int(v) for v in np.asarray(values).astype(object).reshape(-1)]


def _virtual_value(
    polys: list[frx.Array],
    messages: tuple[int, ...],
    widths: tuple[int, ...],
    point: frx.Array,
) -> frx.Array:
    """`V(point)` from the product's whole table, `[x_1, …, x_m, y]` MSB-first."""
    shared = point.shape[0] - sum(widths)
    table = fnp.ones((1 << shared,), _FIELD)
    for message, width in zip(messages, widths, strict=True):
        table = table[..., None, :] * polys[message].reshape(1 << width, 1 << shared)
    return eval_mle(table.reshape(-1), point)


class AkitaVirtualTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.config = _config()
        ring = cls.config.profile.ring
        rng = np.random.default_rng(7)
        inner = _uniform_matrix(ring, rng, cls.config.inner_rows, cls.config.inner_cols)
        outer = _uniform_matrix(ring, rng, cls.config.outer_rows, cls.config.outer_cols)
        cls.committer = AkitaCommitter(cls.config, inner, outer)
        cls.polys = [
            rand_field(1 + i, (length,), _FIELD) for i, length in enumerate(_LENS)
        ]
        cls.commitment, cls.data = cls.committer.commit(cls.polys)
        policy = FixedWeightTernary(_D, 8)
        cls.prover = AkitaProver(cls.committer, policy)
        cls.verifier = AkitaVerifier(cls.committer, policy)

    def _claim(
        self, messages: tuple[int, ...], widths: tuple[int, ...], seed: int
    ) -> VirtualClaim:
        variables = _LENS[messages[0]].bit_length() - 1 - widths[0] + sum(widths)
        point = rand_field(seed, (variables,), _FIELD)
        value = _virtual_value(self.polys, messages, widths, point)
        return VirtualClaim(self.commitment, messages, widths, point, value)

    def _reduce(self, claim: VirtualClaim) -> tuple[VirtualProof, bool]:
        _, proof, _ = self.prover.reduce_virtual(
            claim, OpeningWitness(self.data), _transcript()
        )
        _, _, holds, _ = self.verifier.reduce_virtual(claim, proof, _transcript())
        return proof, holds

    def _assert_chain(self, claim: VirtualClaim) -> None:
        witness = OpeningWitness(self.data)
        opening_claim, proof, proved = self.prover.reduce_virtual(
            claim, witness, _transcript()
        )
        checked, expected, holds, verified = self.verifier.reduce_virtual(
            claim, proof, _transcript()
        )
        self.assertTrue(holds)
        self.assertEqual(
            [_ints(p) for p in checked.points], [_ints(p) for p in opening_claim.points]
        )
        opening, _ = self.prover.open(opening_claim, witness, proved)
        ok, _ = self.verifier.verify(checked, opening, verified)
        self.assertTrue(ok)
        # The link the caller closes: the opening proves exactly the factor
        # values the reduction's last check consumed.
        self.assertEqual(_ints(opening.values), _ints(expected))
        self.assertEqual(
            _ints(expected),
            [
                int(np.asarray(eval_mle(self.polys[m], p)).astype(object))
                for m, p in zip(checked.messages, checked.points, strict=True)
            ],
        )

    def test_chain_over_shared_variables(self) -> None:
        claim = self._claim((0, 1), (2, 1), seed=11)
        self.assertEqual(shared_vars(self.config, claim), 6)
        self._assert_chain(claim)

    def test_chain_without_shared_variables(self) -> None:
        claim = self._claim((2, 1), (6, 7), seed=12)
        proof, holds = self._reduce(claim)
        self.assertTrue(holds)
        self.assertEqual(tuple(proof.rounds.shape), (0, 4))
        self._assert_chain(claim)

    def test_rejects_a_wrong_virtual_value(self) -> None:
        claim = self._claim((0, 1), (2, 1), seed=13)
        _, proof, _ = self.prover.reduce_virtual(
            claim, OpeningWitness(self.data), _transcript()
        )
        wrong = VirtualClaim(
            claim.commitment,
            claim.messages,
            claim.widths,
            claim.point,
            claim.value + fnp.ones((), _FIELD),
        )
        _, _, holds, _ = self.verifier.reduce_virtual(wrong, proof, _transcript())
        self.assertFalse(holds)

    def test_rejects_a_wrong_factor_value(self) -> None:
        claim = self._claim((0, 1), (2, 1), seed=14)
        _, proof, _ = self.prover.reduce_virtual(
            claim, OpeningWitness(self.data), _transcript()
        )
        factors = proof.factors.at[0].add(fnp.ones((), _FIELD))
        _, _, holds, _ = self.verifier.reduce_virtual(
            claim, VirtualProof(proof.rounds, factors), _transcript()
        )
        self.assertFalse(holds)

    def test_rejects_a_round_that_keeps_its_sum(self) -> None:
        # `s(0) + s(1)` unchanged, so the round's own check passes and only the
        # claim it reduces to — hence the final check — can catch it.
        claim = self._claim((0, 1), (2, 1), seed=15)
        _, proof, _ = self.prover.reduce_virtual(
            claim, OpeningWitness(self.data), _transcript()
        )
        one = fnp.ones((), _FIELD)
        rounds = proof.rounds.at[0, 0].add(one).at[0, 1].add(-one)
        _, _, holds, _ = self.verifier.reduce_virtual(
            claim, VirtualProof(rounds, proof.factors), _transcript()
        )
        self.assertFalse(holds)

    def test_refuses_a_factor_of_the_wrong_length(self) -> None:
        point = rand_field(16, (9,), _FIELD)
        claim = VirtualClaim(
            self.commitment, (0, 1), (3, 1), point, fnp.zeros((), _FIELD)
        )
        with self.assertRaisesRegex(ValueError, "needs 64"):
            shared_vars(self.config, claim)


if __name__ == "__main__":
    absltest.main()
