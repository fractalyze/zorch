# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Ligerito recursive open/verify: prover<->verifier round-trip on the
multiplicative Reed-Solomon code, against an independent multilinear evaluation
(`eval_mle`) of the committed polynomial. Reed-Solomon is the de-risk vehicle for
the code-generic recursion; the binary-field (GHASH) instantiation is deferred to
the additive-NTT code (fractalyze/flock-zorch#11, #27).
"""
from __future__ import annotations

import dataclasses
import hashlib

import frx
import frx.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.fold import sample_distinct_positions
from zorch.pcs.ligerito.choreography import LigeritoChoreography
from zorch.pcs.ligerito.config import LigeritoCommitment, LigeritoConfig, LigeritoProof
from zorch.pcs.ligerito.prover import LigeritoProver, LigeritoProverData
from zorch.pcs.ligerito.verifier import LigeritoVerifier
from zorch.poly.multilinear import eval_mle
from zorch.testkit.random_field import rand_ext_field
from zorch.transcript import DuplexTranscript, TranscriptT


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


def _rand_ef(seed: int, shape: tuple[int, ...]) -> jnp.ndarray:
    return rand_ext_field(seed, shape, F, EF)


def _make_code(message_len: int, log_inv_rate: int) -> ReedSolomon:
    return ReedSolomon(message_len=message_len, blowup=1 << log_inv_rate, dtype=EF)


def _setup(
    cfg: LigeritoConfig,
) -> tuple[
    LigeritoProver,
    LigeritoVerifier,
    LigeritoCommitment,
    jnp.ndarray,
    LigeritoProverData,
]:
    _, _, tree = koalabear16_merkle()
    prover = LigeritoProver(_make_code, tree, cfg)
    verifier = LigeritoVerifier(_make_code, tree, cfg)
    f = _rand_ef(1, (1 << cfg.num_vars,))
    root, pdata = prover.commit([f])
    return prover, verifier, root, f, pdata


class LigeritoTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # One recursive level (L=2): the smallest case exercising induce/glue.
        dict(
            testcase_name="l2_4v",
            num_vars=4,
            fold_ks=(1, 1),
            rates=(1, 1),
            queries=(4, 4),
        ),
        # Two recursive levels (L=3), residual 3.
        dict(
            testcase_name="l3_6v",
            num_vars=6,
            fold_ks=(1, 1, 1),
            rates=(1, 1, 1),
            queries=(4, 4, 4),
        ),
        # Multi-variable folds per level.
        dict(
            testcase_name="l2_bigfold",
            num_vars=5,
            fold_ks=(2, 1),
            rates=(1, 1),
            queries=(4, 4),
        ),
        # Shrinking rate across levels (the real Ligerito rate schedule).
        dict(
            testcase_name="l2_shrinkrate",
            num_vars=4,
            fold_ks=(1, 1),
            rates=(2, 1),
            queries=(4, 4),
        ),
        # L=1 (no recursion): value + direct proximity + terminal only, no glue.
        dict(
            testcase_name="l1_2res", num_vars=3, fold_ks=(1,), rates=(1,), queries=(4,)
        ),
        # Minimal residual (1 var) across a recursive level.
        dict(
            testcase_name="l2_res1",
            num_vars=3,
            fold_ks=(1, 1),
            rates=(1, 1),
            queries=(2, 2),
        ),
        # Three recursive levels, mixed folds + shrinking rate.
        dict(
            testcase_name="l4_8v",
            num_vars=8,
            fold_ks=(2, 1, 1, 1),
            rates=(3, 2, 1, 1),
            queries=(6, 4, 4, 2),
        ),
        # OOD binding blocks per recursive commit (mixed counts, including 0).
        dict(
            testcase_name="l3_6v_ood",
            num_vars=6,
            fold_ks=(1, 1, 1),
            rates=(1, 1, 1),
            queries=(4, 4, 4),
            ood_samples=(2, 0),
        ),
        # LSB-first alpha orientation; queries=6 is non-power-of-two, so the
        # glue weights are a genuinely different set than the MSB-first default.
        dict(
            testcase_name="l4_8v_alpha_lsb",
            num_vars=8,
            fold_ks=(2, 1, 1, 1),
            rates=(3, 2, 1, 1),
            queries=(6, 4, 4, 2),
            alpha_lsb_first=True,
        ),
        # Compressed [c0, c2] round messages (c1 reconstructed from the claim),
        # combined with the LSB-first alpha — the two wire-convention knobs a
        # byte-fixed consumer flips together.
        dict(
            testcase_name="l4_8v_compressed_msgs",
            num_vars=8,
            fold_ks=(2, 1, 1, 1),
            rates=(3, 2, 1, 1),
            queries=(6, 4, 4, 2),
            alpha_lsb_first=True,
            compressed_sumcheck_messages=True,
        ),
    )
    def test_open_verify_round_trip(
        self,
        num_vars: int,
        fold_ks: tuple[int, ...],
        rates: tuple[int, ...],
        queries: tuple[int, ...],
        alpha_lsb_first: bool = False,
        compressed_sumcheck_messages: bool = False,
        ood_samples: tuple[int, ...] = (),
    ) -> None:
        cfg = LigeritoConfig(
            num_vars=num_vars,
            fold_ks=fold_ks,
            log_inv_rates=rates,
            queries=queries,
            ood_samples=ood_samples,
            alpha_lsb_first=alpha_lsb_first,
            compressed_sumcheck_messages=compressed_sumcheck_messages,
        )
        prover, verifier, root, f, pdata = _setup(cfg)
        z = _rand_ef(2, (cfg.num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        self.assertEqual(value.tolist(), eval_mle(f, z).tolist())
        ok, _ = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))


# A recursive config (L=2) with a residual — exercises every proof component the
# tamper cases below poke: recursive root, opened rows, residual, sumcheck msg,
# OOD value.
_TAMPER_CFG = LigeritoConfig(
    num_vars=4,
    fold_ks=(1, 1),
    log_inv_rates=(1, 1),
    queries=(4, 4),
    ood_samples=(1,),
)


class LigeritoTamperTest(parameterized.TestCase):
    def _open(
        self,
    ) -> tuple[
        LigeritoVerifier, LigeritoCommitment, jnp.ndarray, jnp.ndarray, LigeritoProof
    ]:
        prover, verifier, root, f, pdata = _setup(_TAMPER_CFG)
        z = _rand_ef(3, (_TAMPER_CFG.num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        return verifier, root, z, value, proof

    def _reject(
        self,
        verifier: LigeritoVerifier,
        root: LigeritoCommitment,
        z: jnp.ndarray,
        value: jnp.ndarray,
        proof: LigeritoProof,
    ) -> None:
        ok, _ = verifier.verify(root, [z], value, proof, _transcript())
        self.assertFalse(bool(ok))

    def test_rejects_tampered_recursive_root(self) -> None:
        verifier, root, z, value, proof = self._open()
        roots = list(proof.recursive_roots)
        roots[0] = roots[0] + jnp.ones_like(roots[0])
        self._reject(
            verifier, root, z, value, dataclasses.replace(proof, recursive_roots=roots)
        )

    def test_rejects_tampered_opened_row(self) -> None:
        verifier, root, z, value, proof = self._open()
        opens = list(proof.component_openings)
        co = opens[0]
        opens[0] = dataclasses.replace(co, row=co.row + jnp.array(1, F))
        self._reject(
            verifier,
            root,
            z,
            value,
            dataclasses.replace(proof, component_openings=opens),
        )

    def test_rejects_tampered_residual(self) -> None:
        verifier, root, z, value, proof = self._open()
        bad = dataclasses.replace(
            proof, final_residual=proof.final_residual + jnp.array(1, EF)
        )
        self._reject(verifier, root, z, value, bad)

    def test_rejects_tampered_sumcheck_message(self) -> None:
        verifier, root, z, value, proof = self._open()
        msgs = list(proof.sumcheck_messages)
        msgs[0] = msgs[0] + jnp.array(1, EF)
        self._reject(
            verifier, root, z, value, dataclasses.replace(proof, sumcheck_messages=msgs)
        )

    def test_rejects_tampered_value(self) -> None:
        verifier, root, z, value, proof = self._open()
        self._reject(verifier, root, z, value + jnp.array(1, EF), proof)

    def test_rejects_tampered_ood_value(self) -> None:
        # A shifted OOD claim desyncs the glued claim from the basis; the next
        # round's s(0)+s(1) == claim link breaks.
        verifier, root, z, value, proof = self._open()
        oods = list(proof.ood_values)
        oods[0] = oods[0] + jnp.array(1, EF)
        self._reject(
            verifier, root, z, value, dataclasses.replace(proof, ood_values=oods)
        )

    def test_rejects_missing_ood_value(self) -> None:
        verifier, root, z, value, proof = self._open()
        with self.assertRaisesRegex(ValueError, "OOD values"):
            verifier.verify(
                root,
                [z],
                value,
                dataclasses.replace(proof, ood_values=[]),
                _transcript(),
            )


@dataclasses.dataclass(frozen=True)
class _FlockShapedChoreography(LigeritoChoreography):
    """flock `pcs::ligerito`'s FS shape over the generic transcript: claim+root
    statement binding (the point rides the outer basis), eager message
    emission, tapered per-fold PoW, unconditional per-level query PoW (0 bits
    still advances the stream), rejection-sampled distinct sorted queries, and
    element-wise residual framing — the zorch-side rehearsal of the byte-fixed
    consumer (fractalyze/flock-zorch#32), exercising every seam hook at once."""

    fold_bits: tuple[int, ...] = ()
    query_bits: tuple[int, ...] = ()

    @property
    def eager_messages(self) -> bool:
        return True

    def bind_statement(
        self, transcript: TranscriptT, root: Array, point: Array, value: Array
    ) -> TranscriptT:
        del point
        transcript = transcript.observe(value)
        return transcript.observe(root)

    def fold_challenge(
        self, transcript: TranscriptT, msg: Array | None, level: int, fold_idx: int
    ) -> tuple[TranscriptT, Array]:
        del msg, level, fold_idx  # eager: the message is already absorbed
        transcript, r = transcript.sample(1)
        return transcript, r[0]

    def fold_grind_bits(self, level: int, fold_idx: int) -> int | None:
        bits = self.fold_bits[level] - fold_idx
        return bits if bits > 0 else None

    def query_grind_bits(self, level: int) -> int | None:
        return self.query_bits[level]

    def sample_queries(
        self, transcript: TranscriptT, block_len: int, count: int
    ) -> tuple[TranscriptT, Array]:
        return sample_distinct_positions(transcript, block_len, count)

    def observe_residual(self, transcript: TranscriptT, residual: Array) -> TranscriptT:
        for k in range(residual.shape[0]):
            transcript = transcript.observe(residual[k])
        return transcript


class LigeritoChoreographyTest(parameterized.TestCase):
    """Round-trips under the flock-shaped choreography: eager emission,
    grinding, distinct queries, and custom binding must keep prover and
    verifier in Fiat-Shamir lockstep (pinned by comparing a post-open and a
    post-verify squeeze) for both message wire forms, with and without OOD."""

    @parameterized.named_parameters(
        dict(testcase_name="value_form", compressed=False, ood=()),
        dict(testcase_name="compressed_ood", compressed=True, ood=(2, 0)),
    )
    def test_open_verify_round_trip(
        self, compressed: bool, ood: tuple[int, ...]
    ) -> None:
        cfg = LigeritoConfig(
            num_vars=6,
            fold_ks=(1, 1, 1),
            log_inv_rates=(1, 1, 1),
            queries=(4, 4, 4),
            ood_samples=ood,
            alpha_lsb_first=True,
            compressed_sumcheck_messages=compressed,
        )
        chor = _FlockShapedChoreography(fold_bits=(2, 1, 0), query_bits=(1, 0, 2))
        _, _, tree = koalabear16_merkle()
        prover = LigeritoProver(_make_code, tree, cfg, chor)
        verifier = LigeritoVerifier(_make_code, tree, cfg, chor)
        f = _rand_ef(1, (1 << cfg.num_vars,))
        root, pdata = prover.commit([f])
        z = _rand_ef(2, (cfg.num_vars,))
        value, proof, t_open = prover.open(pdata, [z], _transcript())
        self.assertEqual(value.tolist(), eval_mle(f, z).tolist())
        # The eager wire carries the extras: initial + post-fold + introduces.
        self.assertEqual(len(proof.sumcheck_messages), chor.num_messages(cfg))
        self.assertEqual(len(proof.pow_witnesses), chor.num_pow_witnesses(cfg))
        ok, t_verify = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))
        _, s_open = t_open.sample()
        _, s_verify = t_verify.sample()
        self.assertEqual(s_open.tolist(), s_verify.tolist())


class LigeritoBasisConventionTest(parameterized.TestCase):
    """Round-trips of the coefficient-basis conventions: the `monomial_commit`
    knob (bit-reversed raw commit + monomial proximity expansion + reversed
    lane weights) and the raw-basis entries (`open_with_basis` /
    `verify_with_basis`, driven through a choreography that binds no point),
    separately and combined."""

    @parameterized.named_parameters(
        dict(testcase_name="monomial_point_entry", monomial=True, basis_entry=False),
        dict(testcase_name="eval_basis_entry", monomial=False, basis_entry=True),
        dict(testcase_name="monomial_basis_entry", monomial=True, basis_entry=True),
    )
    def test_round_trip(self, monomial: bool, basis_entry: bool) -> None:
        cfg = LigeritoConfig(
            num_vars=6,
            fold_ks=(2, 1),
            log_inv_rates=(1, 1),
            queries=(4, 3),
            ood_samples=(1,),
            alpha_lsb_first=True,
            compressed_sumcheck_messages=True,
            monomial_commit=monomial,
        )
        chor = (
            _FlockShapedChoreography(fold_bits=(1, 0), query_bits=(0, 1))
            if basis_entry
            else LigeritoChoreography()
        )
        _, _, tree = koalabear16_merkle()
        prover = LigeritoProver(_make_code, tree, cfg, chor)
        verifier = LigeritoVerifier(_make_code, tree, cfg, chor)
        f = _rand_ef(3, (1 << cfg.num_vars,))
        root, pdata = prover.commit([f])
        if basis_entry:
            # A RAW basis (not an eq expansion) — the batched-claim shape the
            # entry exists for.
            basis = _rand_ef(4, (1 << cfg.num_vars,))
            value = (f * basis).sum()
            proof, t_open = prover.open_with_basis(pdata, basis, value, _transcript())
            ok, t_verify = verifier.verify_with_basis(
                root, basis, value, proof, _transcript()
            )
        else:
            z = _rand_ef(4, (cfg.num_vars,))
            value, proof, t_open = prover.open(pdata, [z], _transcript())
            self.assertEqual(value.tolist(), eval_mle(f, z).tolist())
            ok, t_verify = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))
        _, s_open = t_open.sample()
        _, s_verify = t_verify.sample()
        self.assertEqual(s_open.tolist(), s_verify.tolist())

    def test_native_binding_refuses_basis_entry(self) -> None:
        cfg = LigeritoConfig(
            num_vars=4, fold_ks=(1, 1), log_inv_rates=(1, 1), queries=(4, 4)
        )
        _, _, tree = koalabear16_merkle()
        prover = LigeritoProver(_make_code, tree, cfg)
        f = _rand_ef(5, (1 << cfg.num_vars,))
        _, pdata = prover.commit([f])
        basis = _rand_ef(6, (1 << cfg.num_vars,))
        with self.assertRaisesRegex(ValueError, "basis binds the statement"):
            prover.open_with_basis(pdata, basis, (f * basis).sum(), _transcript())


# L=2 with one OOD block and grinds on both schedules — the smallest config
# whose eager wire carries every extra: [initial, post-fold, OOD intro,
# induce, terminal] messages plus a fold witness and two query witnesses.
_FLOCK_TAMPER_CFG = LigeritoConfig(
    num_vars=4,
    fold_ks=(1, 1),
    log_inv_rates=(1, 1),
    queries=(4, 4),
    ood_samples=(1,),
)
_FLOCK_TAMPER_CHOR = _FlockShapedChoreography(fold_bits=(1, 0), query_bits=(0, 1))


class LigeritoChoreographyTamperTest(absltest.TestCase):
    def _open(
        self,
    ) -> tuple[
        LigeritoVerifier, LigeritoCommitment, jnp.ndarray, jnp.ndarray, LigeritoProof
    ]:
        _, _, tree = koalabear16_merkle()
        prover = LigeritoProver(_make_code, tree, _FLOCK_TAMPER_CFG, _FLOCK_TAMPER_CHOR)
        verifier = LigeritoVerifier(
            _make_code, tree, _FLOCK_TAMPER_CFG, _FLOCK_TAMPER_CHOR
        )
        f = _rand_ef(1, (1 << _FLOCK_TAMPER_CFG.num_vars,))
        root, pdata = prover.commit([f])
        z = _rand_ef(3, (_FLOCK_TAMPER_CFG.num_vars,))
        value, proof, _ = prover.open(pdata, [z], _transcript())
        return verifier, root, z, value, proof

    def test_rejects_tampered_pow_witness(self) -> None:
        # A shifted witness desyncs the replayed stream (and likely fails the
        # zero-bits check outright); every later challenge diverges.
        verifier, root, z, value, proof = self._open()
        wits = list(proof.pow_witnesses)
        wits[0] = wits[0] + jnp.array(1, F)
        ok, _ = verifier.verify(
            root,
            [z],
            value,
            dataclasses.replace(proof, pow_witnesses=wits),
            _transcript(),
        )
        self.assertFalse(bool(ok))

    def test_rejects_tampered_introduce_message(self) -> None:
        # Index 2 is the OOD introduce message; the recombined round poly then
        # disagrees with the tracked claim at the next round.
        verifier, root, z, value, proof = self._open()
        msgs = list(proof.sumcheck_messages)
        msgs[2] = msgs[2] + jnp.array(1, EF)
        ok, _ = verifier.verify(
            root,
            [z],
            value,
            dataclasses.replace(proof, sumcheck_messages=msgs),
            _transcript(),
        )
        self.assertFalse(bool(ok))

    def test_rejects_tampered_terminal_message(self) -> None:
        # The last emission is pinned against the in-clear residual's poly.
        verifier, root, z, value, proof = self._open()
        msgs = list(proof.sumcheck_messages)
        msgs[-1] = msgs[-1] + jnp.array(1, EF)
        ok, _ = verifier.verify(
            root,
            [z],
            value,
            dataclasses.replace(proof, sumcheck_messages=msgs),
            _transcript(),
        )
        self.assertFalse(bool(ok))

    def test_rejects_missing_pow_witness(self) -> None:
        verifier, root, z, value, proof = self._open()
        with self.assertRaisesRegex(ValueError, "proof-of-work"):
            verifier.verify(
                root,
                [z],
                value,
                dataclasses.replace(proof, pow_witnesses=[]),
                _transcript(),
            )


class LigeritoWireGoldenTest(parameterized.TestCase):
    """Byte-pin of the default (zorch-native) wire: the digest covers the
    opened value, every proof leaf, and a post-open / post-verify squeeze from
    each side's transcript, so ANY reordered or reframed Fiat-Shamir
    interaction moves it. Captured before the `LigeritoChoreography` seam landed —
    the default choreography must keep these bytes forever; regenerate only
    for an intentional wire change."""

    @parameterized.named_parameters(
        dict(
            testcase_name="l4_8v",
            alpha_lsb_first=False,
            compressed_sumcheck_messages=False,
            want="6b19850c48d1e63f3c6243418969d6287937b9a0f10b476738b06e93bb1959c8",
        ),
        dict(
            testcase_name="l4_8v_flockknobs",
            alpha_lsb_first=True,
            compressed_sumcheck_messages=True,
            want="4e326ca50a0762cbd576e728e37958941ed0ea3013cf54295d60471ae9be6dc6",
        ),
    )
    def test_default_wire_digest(
        self, alpha_lsb_first: bool, compressed_sumcheck_messages: bool, want: str
    ) -> None:
        cfg = LigeritoConfig(
            num_vars=8,
            fold_ks=(2, 1, 1, 1),
            log_inv_rates=(3, 2, 1, 1),
            queries=(6, 4, 4, 2),
            alpha_lsb_first=alpha_lsb_first,
            compressed_sumcheck_messages=compressed_sumcheck_messages,
        )
        prover, verifier, root, _, pdata = _setup(cfg)
        z = _rand_ef(2, (cfg.num_vars,))
        value, proof, t_open = prover.open(pdata, [z], _transcript())
        ok, t_verify = verifier.verify(root, [z], value, proof, _transcript())
        self.assertTrue(bool(ok))
        _, s_open = t_open.sample()
        _, s_verify = t_verify.sample()
        h = hashlib.sha256()
        for leaf in [value, *frx.tree_util.tree_leaves(proof), s_open, s_verify]:
            h.update(
                np.asarray(jnp.asarray(leaf).reshape(-1).view(jnp.uint32)).tobytes()
            )
        self.assertEqual(h.hexdigest(), want)


class LigeritoConfigTest(absltest.TestCase):
    def test_rejects_empty_fold_ks(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            LigeritoConfig(num_vars=4, fold_ks=(), log_inv_rates=(), queries=())

    def test_rejects_non_positive_fold_ks(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            LigeritoConfig(
                num_vars=4, fold_ks=(0, 1), log_inv_rates=(1, 1), queries=(4, 4)
            )

    def test_rejects_wrong_ood_samples_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "ood_samples"):
            LigeritoConfig(
                num_vars=4,
                fold_ks=(1, 1),
                log_inv_rates=(1, 1),
                queries=(4, 4),
                ood_samples=(1, 1),
            )

    def test_rejects_negative_ood_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LigeritoConfig(
                num_vars=4,
                fold_ks=(1, 1),
                log_inv_rates=(1, 1),
                queries=(4, 4),
                ood_samples=(-1,),
            )


class LigeritoCommitTest(absltest.TestCase):
    def test_commit_independent_of_query_count(self) -> None:
        # Two provers differing ONLY in the query count must produce an identical
        # commitment — commit never reads `queries`, so the jitted `_commit`
        # (keyed on code+tree+interleave, #214) does not re-trace across them.
        def _cfg(queries: tuple[int, ...]) -> LigeritoConfig:
            return LigeritoConfig(
                num_vars=4, fold_ks=(1, 1), log_inv_rates=(1, 1), queries=queries
            )

        cfg_a = _cfg((4, 4))
        cfg_b = _cfg((8, 2))
        _, _, tree = koalabear16_merkle()
        f = _rand_ef(1, (1 << 4,))
        root_a, _ = LigeritoProver(_make_code, tree, cfg_a).commit([f])
        root_b, _ = LigeritoProver(_make_code, tree, cfg_b).commit([f])
        self.assertEqual(root_a.tolist(), root_b.tolist())

    def test_commit_rejects_multiple_polys(self) -> None:
        _, _, tree = koalabear16_merkle()
        f = _rand_ef(1, (1 << 4,))
        prover = LigeritoProver(_make_code, tree, _TAMPER_CFG)
        with self.assertRaisesRegex(ValueError, "exactly one polynomial"):
            prover.commit([f, f])


if __name__ == "__main__":
    absltest.main()
