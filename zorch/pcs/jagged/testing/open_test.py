# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match the stacked BaseFold open against the SP1 pipeline dump.

Drives ``stacked_basefold_open`` over the vendored ``gpu_fibonacci`` regions
(preprocessed + main, committed separately) with a scripted transcript replaying
the dumped basefold challenges (the RLC weights, the per-round FRI fold betas,
the query positions, and the proof-of-work witness), then byte-matches the
structural proof the open produces: per-round batch evaluations, the FRI
fold-layer roots (SP1 separator-bound), the final poly, the proof-of-work
witness, and the component + query Merkle openings. The duplex encoding that
derives those challenges is the pipeline's concern (verified elsewhere via the
real transcript), so replaying them here isolates the open's commit/fold/open
math — same precedent as ``prover_test`` for the sumcheck half. Mont-u32, no
tolerances.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import frx
import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from frx import Array
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
from zorch.pcs.jagged.verifier import stacked_basefold_verify
from zorch.transcript import DuplexTranscript, GrindingTranscript

_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"
# The committed dense buffer is the same shard the zerocheck stage commits; its
# prep/main slices ride alongside the jagged fixture (testdata/zerocheck_dense)
# rather than under a zerocheck package (absent from zorch).
_ZC_INPUTS = Path(__file__).parent / "testdata" / "zerocheck_dense"


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _from_u32(u32: Any, dtype: Any) -> Array:
    return frx.lax.bitcast_convert_type(jnp.asarray(u32, dtype=jnp.uint32), dtype)


def _u32(a: Array) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _raw_area(round_meta: dict[str, Any]) -> int:
    return sum(
        int(r) * int(c)
        for r, c in zip(
            round_meta["row_counts"], round_meta["column_counts"], strict=True
        )
    )


def _out(name: str) -> Any:
    return np.load(_FIXTURE / "outputs" / name)


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["_samples", "_cursor", "_witness"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _ScriptedTranscript:
    """Replays the dumped basefold challenges so the byte-match exercises the
    open's commit/fold/open math against the reference run's Fiat-Shamir outcomes
    rather than re-deriving them (the duplex encoding is the pipeline's concern).
    Mirrors ``prover_test._ScriptedTranscript`` for the sumcheck half.

    A registered pytree so it threads through the open's ``@frx.jit`` zone: the
    flat base-field squeeze stream ``_samples`` is consumed in order via a traced
    ``_cursor`` (a Python ``list.pop`` cursor would not trace). Each extension
    challenge is four base squeezes; ``_witness`` is the dumped proof-of-work
    witness the grind returns. ``observe`` is a no-op (replay, not absorb)."""

    _samples: jnp.ndarray  # (TOTAL,) flat base-field squeeze stream
    _cursor: jnp.ndarray  # () int32 — squeezes consumed so far
    _witness: jnp.ndarray

    @classmethod
    def of(cls, samples: Sequence[Array], witness: Array) -> _ScriptedTranscript:
        return cls(jnp.stack(list(samples)), jnp.zeros((), jnp.int32), witness)

    def observe(self, values: Array) -> _ScriptedTranscript:
        return self

    def sample(self, n: int = 1) -> tuple[_ScriptedTranscript, Array]:
        out = frx.lax.dynamic_slice_in_dim(self._samples, self._cursor, n, axis=0)
        return dataclasses.replace(self, _cursor=self._cursor + n), out

    def grind(self, pow_bits: int) -> tuple[_ScriptedTranscript, Array]:
        return self, self._witness


# smcs/code/round_widths/log_stacking_height are static; the proof, eval point
# and transcript trace. One compile, cached across runs.
_verify_jit = frx.jit(
    stacked_basefold_verify,
    static_argnums=(0, 1, 2, 5),
    static_argnames=("num_queries", "pow_bits"),
)


class StackedOpenByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        meta = json.loads((_FIXTURE / "meta.json").read_text())
        log_s = int(meta["rounds"][0]["log_stacking_height"])
        cfg = meta["basefold"]
        S = 1 << log_s

        smcs = _smcs()
        code = BitReversedReedSolomon(
            message_len=S, blowup=1 << int(cfg["log_blowup"]), dtype=BF
        )

        prep = _from_u32(
            np.load(_ZC_INPUTS / "prep_dense.npy")[: _raw_area(meta["rounds"][0])], BF
        )
        main = _from_u32(
            np.load(_ZC_INPUTS / "main_dense.npy")[: _raw_area(meta["rounds"][1])], BF
        )

        def build_round(dense: Array) -> StackedRound:
            k = dense.shape[0] // S
            # Column-major stacking: dense block k is column k. Encode all columns
            # in one batched FFT (the code rides the leading axis), bit-reversed
            # exactly as trace_commit writes the committed codeword.
            block = dense.reshape(k, S)  # [K, S]
            codeword = code.encode(block).T  # [S*blowup, K]
            _root, digest_layers = smcs.commit(codeword)
            return StackedRound(block=block, digest_layers=digest_layers)

        rounds = [build_round(prep), build_round(main)]

        # Scripted squeeze stream: the RLC weights then each fold round's beta,
        # all as four base squeezes per extension challenge, then one base squeeze
        # per query (the canonical value the open masks to a position).
        ch = _out("challenges.npz")
        ef_stream = _from_u32(
            np.concatenate(
                [ch["batch_challenges"].reshape(-1), ch["fri_betas"].reshape(-1)]
            ),
            BF,
        )
        query_stream = jnp.asarray(ch["query_indices"], dtype=jnp.uint32).astype(BF)
        samples = list(ef_stream) + list(query_stream)
        witness = _from_u32(_out("pow_witness.npy"), BF)

        z_final = _from_u32(_out("outer_sumcheck_point.npy"), EF)
        dense_eval = _from_u32(_out("dense_eval.npy"), EF)

        cls.proof, _ = stacked_basefold_open(
            smcs,
            code,
            rounds,
            z_final,
            dense_eval,
            log_s,
            num_queries=int(cfg["num_queries"]),
            pow_bits=int(cfg["pow_bits"]),
            # The scripted replay implements only the observe/sample/grind
            # subset the open exercises; cast to the protocol it stands in for.
            transcript=cast(
                GrindingTranscript, _ScriptedTranscript.of(samples, witness)
            ),
        )
        # A second open under a real Fiat-Shamir transcript (not the SP1-scripted
        # one) for the open->verify completeness roundtrip: open and verify derive
        # challenges by observation, so a valid proof must verify. pow_bits is 0
        # here, so grind takes its trivial branch and the open stays jittable.
        # Statement (z_final, dense_eval) is unchanged; only the FS challenges
        # differ from the byte-match run.
        cls.smcs, cls.code, cls.rounds = smcs, code, rounds
        cls.z_final, cls.dense_eval, cls.log_s, cls.cfg = (
            z_final,
            dense_eval,
            log_s,
            cfg,
        )
        cls.perm = Poseidon2(koalabear16_params())
        cls.real_proof, _ = stacked_basefold_open(
            smcs,
            code,
            rounds,
            z_final,
            dense_eval,
            log_s,
            num_queries=int(cfg["num_queries"]),
            pow_bits=int(cfg["pow_bits"]),
            transcript=DuplexTranscript.new(cls.perm, rate=8),
        )

    def _assert_match(
        self, got: Array, exp_u32: Any, name: str, allow_zero: bool = False
    ) -> None:
        exp = np.asarray(exp_u32, dtype=np.uint32).reshape(-1)
        if not allow_zero:
            self.assertGreater(int(exp.sum()), 0, f"degenerate fixture {name}")
        got = _u32(got)
        self.assertEqual(got.shape, exp.shape, f"{name} shape")
        mism = np.nonzero(got != exp)[0]
        self.assertEqual(mism.size, 0, f"{name} diverged at u32 {mism[:8]}")

    def test_batch_evals(self) -> None:
        for r in range(2):
            self._assert_match(
                self.proof.batch_evals[r],
                _out(f"batch_evals_r{r}.npz")["mle0"],
                f"batch_evals_r{r}",
            )

    # No fri_raw_roots check: the raw (pre-binding) fold-layer root is a
    # zorch-internal digest with no SP1 reference — SP1's proof carries only the
    # separator-bound root, byte-matched by test_fri_commitments below.

    def test_fri_commitments(self) -> None:
        self._assert_match(
            self.proof.fri_commitments, _out("fri_commitments.npy"), "fri_commitments"
        )

    def test_univariate_messages(self) -> None:
        self._assert_match(
            self.proof.univariate_messages,
            _out("univariate_messages.npy"),
            "univariate_messages",
        )

    def test_final_poly(self) -> None:
        self._assert_match(self.proof.final_poly, _out("final_poly.npy"), "final_poly")

    def test_pow_witness(self) -> None:
        # pow_bits == 0 on this dev fixture, so the witness is the canonical zero.
        self._assert_match(
            self.proof.pow_witness,
            _out("pow_witness.npy"),
            "pow_witness",
            allow_zero=True,
        )

    def test_component_openings(self) -> None:
        for r in range(2):
            rows, paths = self.proof.component_openings[r]
            dump = _out(f"component_openings_r{r}.npz")
            self._assert_match(rows, dump["rows"], f"component_openings_r{r}.rows")
            for lvl, path in enumerate(paths):
                self._assert_match(
                    path, dump[f"proof_l{lvl}"], f"component_r{r}.proof_l{lvl}"
                )

    def test_query_openings(self) -> None:
        for i, (rows, paths) in enumerate(self.proof.query_openings):
            dump = _out(f"query_openings_f{i}.npz")
            self._assert_match(rows, dump["rows"], f"query_openings_f{i}.rows")
            for lvl, path in enumerate(paths):
                self._assert_match(
                    path, dump[f"proof_l{lvl}"], f"query_f{i}.proof_l{lvl}"
                )

    def _verify(self, proof: StackedOpenProof) -> bool:
        round_widths = tuple(r.mle.shape[1] for r in self.rounds)
        _, ok = _verify_jit(
            self.smcs,
            self.code,
            round_widths,
            self.z_final,
            self.dense_eval.reshape(
                ()
            ),  # dump stores it as (1,); verify wants the scalar
            self.log_s,
            proof,
            DuplexTranscript.new(self.perm, rate=8),
            num_queries=int(self.cfg["num_queries"]),
            pow_bits=int(self.cfg["pow_bits"]),
        )
        return bool(ok)

    def test_open_verify_roundtrip(self) -> None:
        # Completeness: the verifier accepts the open's own proof (no SP1 dump,
        # no shard prover) — the real-transcript dual of the open.
        self.assertTrue(self._verify(self.real_proof))

    def test_verify_rejects_tampered_final_poly(self) -> None:
        # The final poly is the fold chain's residual; a flipped lane breaks it.
        u = frx.lax.bitcast_convert_type(self.real_proof.final_poly, jnp.uint32)
        u = u.at[0].set(u[0] ^ jnp.uint32(1))
        bad = dataclasses.replace(
            self.real_proof, final_poly=frx.lax.bitcast_convert_type(u, EF)
        )
        self.assertFalse(self._verify(bad))


if __name__ == "__main__":
    absltest.main()
