# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match the jagged-eval sumcheck Round against the SP1 pipeline dump.

Drives ``prove_jagged_eval`` with a
scripted transcript replaying the dumped outer + inner challenges, and
byte-matches the full sumcheck half: the outer Hadamard sumcheck ``Σ D·J̃``
(round polys, folded point, ``dense_eval``) and the inner branching-program
sumcheck. The committed dense buffer ``D`` is not re-dumped — it is the same
shard packing the zerocheck stage commits, reconstructed here from the shared
``zerocheck`` dense fixture (the eval dump carries only its own outputs).
Mont-u32, no tolerances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.pcs.jagged.prover import (
    JaggedEvalInputs,
    _eval_inputs,
    assemble_columns,
    prove_jagged_eval,
)
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript, sample_challenge

BF = koalabear_mont
EF = koalabearx4_mont
_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"
# The packed dense buffer D is the same shard the zerocheck stage commits; its
# prep/main slices ride alongside the jagged fixture (testdata/zerocheck_dense)
# rather than under a zerocheck package (absent from zorch).
_ZC_INPUTS = Path(__file__).parent / "testdata" / "zerocheck_dense"


def _from_u32(u32: Any, dtype: Any) -> Array:
    return frx.lax.bitcast_convert_type(fnp.asarray(u32, dtype=fnp.uint32), dtype)


def _u32(a: Array) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, fnp.uint32)).reshape(-1)


def _raw_area(round_meta: dict[str, Any]) -> int:
    """Σ row_count·column_count — the round's unpadded packed-dense length."""
    return sum(
        int(r) * int(c)
        for r, c in zip(
            round_meta["row_counts"], round_meta["column_counts"], strict=True
        )
    )


@partial(
    frx.tree_util.register_dataclass, data_fields=["stream", "pos"], meta_fields=[]
)
@dataclass(frozen=True)
class _ScriptedTranscript:
    """Replays the dumped per-round challenges — the byte-match reproduces the
    reference run's Fiat-Shamir outcomes rather than re-deriving them (the duplex
    encoding is the pipeline's concern, not this round's). Mirrors
    ``zerocheck/jagged_byte_match_test``: the sumchecks squeeze base limbs and
    reassemble each EF challenge (the ``sample_challenge`` rule), so the script holds
    one flat base-limb stream.

    A registered pytree with a traced ``pos``: ``inner_sumcheck_core`` threads the
    transcript through a ``lax.scan``, whose carry must be a JAX type (the prior
    mutable double broke under scan)."""

    stream: Array
    pos: Array

    @classmethod
    def create(cls, challenges: Array) -> _ScriptedTranscript:
        stream = frx.lax.bitcast_convert_type(fnp.asarray(challenges), BF).reshape(-1)
        return cls(stream=stream, pos=fnp.array(0, fnp.int32))

    # No dedicated-fusion permutation: observe_and_sample falls back to the plain
    # observe+sample body, so the scripted replay drives it unchanged.
    has_dedicated_fusion = False

    def observe(self, values: Array) -> _ScriptedTranscript:
        return self

    @property
    def field(self) -> Any:
        return self.stream.dtype

    def sample(self, n: int = 1) -> tuple[_ScriptedTranscript, Array]:
        out = frx.lax.dynamic_slice(self.stream, (self.pos,), (n,))
        return replace(self, pos=self.pos + n), out

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[_ScriptedTranscript, Array]:
        return self.sample(n)


class JaggedEvalByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        meta = json.loads((_FIXTURE / "meta.json").read_text())
        row_counts_rounds = [[int(x) for x in r["row_counts"]] for r in meta["rounds"]]
        column_counts_rounds = [
            [int(x) for x in r["column_counts"]] for r in meta["rounds"]
        ]

        z_row = _from_u32(np.load(_FIXTURE / "inputs" / "z_row.npy"), EF)
        claims = [
            _from_u32(np.load(_FIXTURE / "inputs" / f"claims_r{r}.npy"), EF)
            for r in range(len(meta["rounds"]))
        ]
        ch = np.load(_FIXTURE / "outputs" / "challenges.npz")
        z_col = _from_u32(ch["z_col"], EF)
        outer_alphas = _from_u32(ch["outer_alphas"], EF)
        inner_alphas = _from_u32(ch["inner_alphas"], EF)

        # Reconstruct the committed dense buffer D: per round, strip the
        # stacking pad to the raw packed area, concat in round order
        # (prep, main), matching SP1's _build_combined_dense. The two rounds'
        # raw areas sum to a power of two here, so no extra pad is needed.
        prep = _from_u32(
            np.load(_ZC_INPUTS / "prep_dense.npy")[: _raw_area(meta["rounds"][0])], BF
        )
        main = _from_u32(
            np.load(_ZC_INPUTS / "main_dense.npy")[: _raw_area(meta["rounds"][1])], BF
        )
        dense = fnp.concatenate([prep, main])

        col_heights, all_claims = assemble_columns(
            row_counts_rounds, column_counts_rounds, claims, dtype=EF
        )
        carry = JaggedEvalInputs(
            col_heights=tuple(col_heights),
            all_claims=all_claims,
            z_row=z_row,
            z_col=z_col,
            dense=dense,
        )
        # The outer sumcheck samples its 23 alphas first, then the inner its 48.
        script = fnp.concatenate([outer_alphas, inner_alphas])
        # _ScriptedTranscript is a replay-only test double that reproduces the
        # dumped Fiat-Shamir stream rather than re-deriving it.
        cls.msg, _ = prove_jagged_eval(
            carry, _ScriptedTranscript.create(script), dtype=EF
        )

    def _expect(self, name: str) -> np.ndarray:
        return np.load(_FIXTURE / "outputs" / name).reshape(-1)

    def _assert_match(self, got: Array, name: str) -> None:
        exp = self._expect(name)
        self.assertGreater(int(exp.sum()), 0, "degenerate fixture")
        got = _u32(got)
        self.assertEqual(got.shape, exp.shape)
        mism = np.nonzero(got != exp)[0]
        self.assertEqual(mism.size, 0, f"{name} diverged at u32 {mism[:8]}")

    def test_outer_sumcheck_claim(self) -> None:
        self._assert_match(self.msg.outer_sumcheck_claim, "outer_sumcheck_claim.npy")

    def test_outer_sumcheck_polys(self) -> None:
        self._assert_match(self.msg.outer_sumcheck_polys, "outer_sumcheck_polys.npy")

    def test_outer_sumcheck_point(self) -> None:
        self._assert_match(self.msg.outer_sumcheck_point, "outer_sumcheck_point.npy")

    def test_dense_eval(self) -> None:
        self._assert_match(self.msg.dense_eval, "dense_eval.npy")

    def test_inner_claimed_sum(self) -> None:
        self._assert_match(self.msg.inner_claimed_sum, "inner_claimed_sum.npy")

    def test_inner_sumcheck_polys(self) -> None:
        self._assert_match(self.msg.inner_sumcheck_polys, "inner_sumcheck_polys.npy")

    def test_inner_point(self) -> None:
        self._assert_match(self.msg.inner_point, "inner_point.npy")


class ChallengeRuleTest(absltest.TestCase):
    """Pin the squeeze rule on a real transcript: SP1 binds each outer and
    inner round with ``sample_ext_element`` — degree base squeezes
    reinterpreted as one extension element, the shared ``sample_challenge``
    definition. The scripted byte-match above
    bypasses the rule entirely, so it cannot catch a squeeze-count drift."""

    def test_outer_and_inner_rounds_sample_extension_challenges(self) -> None:
        def rand_ef(seed: int, shape: tuple[int, ...]) -> Array:
            ints = np.random.default_rng(seed).integers(
                1, 1 << 30, size=(*shape, 4), dtype=np.int64
            )
            return frx.lax.bitcast_convert_type(fnp.array(ints, dtype=BF), EF)

        col_heights = (2, 2)
        inputs = JaggedEvalInputs(
            col_heights=col_heights,
            all_claims=rand_ef(1, (2,)),
            z_row=rand_ef(2, (2,)),
            z_col=rand_ef(3, (1,)),
            dense=fnp.array([3, 5, 7, 11], dtype=BF),
        )
        msg, _ = prove_jagged_eval(inputs, cheap_transcript(BF), dtype=EF)

        # Independent replay of the stage's whole challenge stream off the
        # message: observe each round poly, take one extension sample, and
        # match the point entry it bound (points are the challenge lists
        # reversed — SP1's insert-at-front).
        def replay_rounds(
            t: DuplexTranscript, polys: Array, point: Array, label: str
        ) -> DuplexTranscript:
            self.assertEqual(point.dtype, EF, label)
            for r in range(polys.shape[0]):
                t = t.observe(polys[r])
                t, want = sample_challenge(t, EF, 4)
                self.assertTrue(
                    bool(fnp.array_equal(want, point[-1 - r])), f"{label} round {r}"
                )
            return t

        t = cheap_transcript(BF)
        t = replay_rounds(
            t, msg.outer_sumcheck_polys, msg.outer_sumcheck_point, "outer"
        )
        # SP1 absorbs the claimed J̃ value before the inner rounds
        # .
        t = t.observe(msg.inner_claimed_sum)
        replay_rounds(t, msg.inner_sumcheck_polys, msg.inner_point, "inner")


class EvalInputsGuardTest(absltest.TestCase):
    """`_eval_inputs` fails loud when z_col carries too few variables for L:
    col_eq = expand_eq(z_col) then has < L entries, silently truncating weights."""

    _HEIGHTS = (3, 5, 2, 7)  # L=4 -> needs ceil(log2 4) = 2 z_col variables

    def test_exact_z_col_count_ok(self) -> None:
        _eval_inputs(self._HEIGHTS, fnp.zeros((2,), EF), EF)  # no raise

    def test_too_few_z_col_raises(self) -> None:
        with self.assertRaises(ValueError):
            _eval_inputs(self._HEIGHTS, fnp.zeros((1,), EF), EF)


if __name__ == "__main__":
    absltest.main()
