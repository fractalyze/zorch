# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jaxlib.mlir.dialects import stablehlo

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.prove import SUMCHECK_MARKER, fold_rounds, prove, prove_fs
from zorch.round import Round
from zorch.sumcheck import prover
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript, StubTranscript

KB = zk_dtypes.koalabear
KBM = zk_dtypes.koalabear_mont  # the poseidon2 koalabear16 sponge field
_HAS_COMPOSITE_OP = hasattr(stablehlo, "CompositeOp")


class _CollectRound(Round):
    """Halves a 1-element-per-factor carry; emits a heterogeneous dict message."""

    def __call__(self, state: Any, transcript: Any) -> Any:
        (xs,) = state
        half = xs.shape[-1] // 2
        msg = {"first": xs[0], "len": xs.shape[-1]}  # non-stackable on purpose
        return [xs[:half]], transcript, msg


class FoldRoundsTest(absltest.TestCase):
    def test_collects_structured_messages_as_list(self) -> None:
        xs = jnp.arange(8, dtype=KB)
        _, _, msgs = fold_rounds(
            _CollectRound(), [xs], StubTranscript(jnp.zeros(0, KB)), 3
        )
        self.assertEqual([m["len"] for m in msgs], [8, 4, 2])
        self.assertEqual(len(msgs), 3)

    def test_prove_rejects_zero_round_state(self) -> None:
        # A width-1 carry derives 0 rounds: the scan would yield no round polys.
        # Fail fast with a clear message instead.
        with self.assertRaisesRegex(ValueError, "at least one round"):
            prove(
                prover.SumcheckRound(1),
                [jnp.arange(1, dtype=KB)],
                StubTranscript(jnp.zeros(0, KB)),
            )

    def test_prove_still_stacks_sumcheck_messages(self) -> None:
        f = jnp.arange(1, 17, dtype=KB)
        ch = jnp.arange(2, 6, dtype=KB)
        _, _, msgs = prove(prover.SumcheckRound(degree=1), [f], StubTranscript(ch))
        self.assertEqual(msgs.round_poly.shape, (4, 2))  # n rounds × (degree+1)
        self.assertEqual(msgs.challenge.shape, (4,))  # one challenge per round


class ProveFsTest(absltest.TestCase):
    """`prove_fs` wraps `prove`'s scan in a `zorch.sumcheck` marker with
    Fiat-Shamir sampled INSIDE -- the duplex sponge threads through the marker as
    operands. Unrecognized it inlines to the exact `(proof, transcript)` `prove`
    produces from the same fresh transcript; the recognized->fused GPU path is
    exercised in zkx."""

    def _transcript(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def _assert_inline_equals_prove(
        self, round: prover.SumcheckRound, factors: list[Any]
    ) -> None:
        # FS-internal samples its own challenges from the threaded transcript, so
        # both the proof AND the advanced transcript must match `prove` driven from
        # an identical fresh transcript.
        got_proof, got_t = prove_fs(round, list(factors), self._transcript())
        _, want_t, msgs = prove(round, list(factors), self._transcript())
        want_proof = msgs.round_poly.reshape(-1)  # flat round-major, marker layout
        self.assertEqual(got_proof.shape, want_proof.shape)
        self.assertTrue(bool(jnp.all(got_proof == want_proof)))
        if not isinstance(want_t, DuplexTranscript):
            raise AssertionError("prove should thread the DuplexTranscript back")
        g, w = got_t.state, want_t.state
        self.assertTrue(bool(jnp.all(g.sponge_state == w.sponge_state)))
        self.assertTrue(bool(jnp.all(g.output_buffer == w.output_buffer)))
        self.assertEqual(int(g.in_pos), int(w.in_pos))
        self.assertEqual(int(g.out_pos), int(w.out_pos))

    def test_inline_equals_prove_degree2(self) -> None:
        a = rand_field(40, (1 << 4,), KBM)
        b = rand_field(41, (1 << 4,), KBM)
        self._assert_inline_equals_prove(prover.SumcheckRound(degree=2), [a, b])

    def test_inline_equals_prove_degree1_single_mle(self) -> None:
        f = rand_field(43, (1 << 5,), KBM)
        self._assert_inline_equals_prove(prover.SumcheckRound(degree=1), [f])

    def test_marker_envelope_names_degree_and_num_vars(self) -> None:
        # Recognition contract: name carries degree:num_vars; results are
        # [proof][5 transcript leaves]; the poseidon2 round constants auto-lift, so
        # the marker carries more operands than the [factors][5 leaves] explicit.
        a = rand_field(40, (1 << 4,), KBM)
        b = rand_field(41, (1 << 4,), KBM)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._transcript()  # build the sponge eagerly, not under the trace
        jaxpr = jax.make_jaxpr(lambda x, y: prove_fs(rnd, [x, y], t0))(a, b).jaxpr
        eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
        self.assertEqual(eqn.params["name"], f"{SUMCHECK_MARKER}:2:4")
        self.assertEqual(len(eqn.outvars), 1 + 5)
        self.assertGreater(
            len(eqn.invars), 2 + 5
        )  # RC auto-lifted past the explicit operands

    @absltest.skipUnless(_HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp")
    def test_marker_and_nested_permute_survive_lowering(self) -> None:
        # The whole sumcheck lowers under the hash-agnostic zorch.sumcheck marker;
        # the FS permute survives as a nested poseidon2: marker the vendor reads to
        # run the sponge in-kernel.
        a = rand_field(40, (1 << 4,), KBM)
        b = rand_field(41, (1 << 4,), KBM)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._transcript()  # build the sponge eagerly, not under the trace
        text = jax.jit(lambda x, y: prove_fs(rnd, [x, y], t0)).lower(a, b).as_text()
        self.assertIn(SUMCHECK_MARKER, text)
        self.assertIn("poseidon2:", text)

    def test_rejects_empty_state(self) -> None:
        with self.assertRaises(ValueError):
            prove_fs(prover.SumcheckRound(degree=1), [], self._transcript())

    def test_rejects_zero_round_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one round"):
            prove_fs(
                prover.SumcheckRound(degree=1),
                [jnp.arange(1, dtype=KBM)],
                self._transcript(),
            )


if __name__ == "__main__":
    absltest.main()
