# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for the LogUp-GKR prover module.

`LogupSumcheckRound` -- the per-variable LogUp sumcheck round -- and
`GkrLayerRound` -- one GKR layer -- both live in `prover.py`, so both are
exercised here. Full correctness (per-layer oracle + reduction to the input leaf
MLE) is the prove->verify roundtrip in verifier_test; this pins the prover's own
invariants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array, tree_util

from zorch.logup_gkr.circuit import build_pyramid, extract_outputs
from zorch.logup_gkr.prover import (
    GkrLayerRound,
    LogupSumcheckRound,
    LogupSummand,
    bind_output,
    logup_combine,
)
from zorch.logup_gkr.testing import prove_gkr, prove_gkr_jitted, random_first_layer
from zorch.poly.univariate import eval_univariate
from zorch.prove import fold_rounds
from zorch.round import ProveChain
from zorch.sumcheck.prover import prove
from zorch.testkit.fusion import assert_fusion_ready
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont


def _state(seed: int, width: int) -> list[Array]:
    """Five MLE-eval factors in combine order [eq, n0, d1, n1, d0]."""
    return [rand_field(seed + i, (width,), KB) for i in range(5)]


class LogupSummandTest(absltest.TestCase):
    """`LogupSummand` is the seam shared by the dense round (here) and the
    jagged round's `paired_evals` (jagged_prover_test); pinning it directly
    keeps both consumers honest against the same combine."""

    def test_combine_matches_module_level_logup_combine(self) -> None:
        lam = jnp.array(6, KB)
        eq, n0, d1, n1, d0 = (jnp.array(v, KB) for v in (2, 3, 4, 5, 7))
        got = LogupSummand(lam).combine((lam,), eq, n0, d1, n1, d0)
        want = logup_combine(lam, eq, n0, d1, n1, d0)
        self.assertTrue(bool(got == want))

    def test_degree_is_three(self) -> None:
        self.assertEqual(LogupSummand(jnp.array(1, KB)).degree, 3)

    def test_combine_guards_factor_count(self) -> None:
        lam = jnp.array(1, KB)
        with self.assertRaises(ValueError):
            LogupSummand(lam).combine((lam,), *_state(70, 8)[:4])


class LogupSumcheckRoundTest(absltest.TestCase):
    def test_round_poly_matches_naive_cubic(self) -> None:
        st = _state(20, 8)
        rnd = LogupSumcheckRound(jnp.array(7, KB))
        msg = rnd._round_poly(st)
        self.assertEqual(msg.shape, (4,))  # degree 3 -> 4 evals
        half = 4
        for u in range(4):
            uf = jnp.array(u, KB)
            folded = [x[:half] + uf * (x[half:] - x[:half]) for x in st]
            want = jnp.sum(rnd._combine(*folded))
            self.assertTrue(bool(msg[u] == want))

    def test_round_poly_with_batch_dimension(self) -> None:
        # Leading batch dims must broadcast: msg is (degree+1, *batch).
        batch = 3
        st = [x.reshape(batch, -1) for x in _state(20, 24)]
        msg = LogupSumcheckRound(jnp.array(7, KB))._round_poly(st)
        self.assertEqual(msg.shape, (4, batch))

    def test_sumcheck_invariant_s0_plus_s1(self) -> None:
        st = _state(40, 16)
        rnd = LogupSumcheckRound(jnp.array(9, KB))
        msg = rnd._round_poly(st)
        # s(0)+s(1) == sum over the full hypercube of the combine (the claim)
        self.assertTrue(bool(msg[0] + msg[1] == jnp.sum(rnd._combine(*st))))

    def test_call_returns_round_msg_with_challenge(self) -> None:
        st = _state(50, 8)
        state, _, msg = LogupSumcheckRound(jnp.array(2, KB))(st, cheap_transcript(KB))
        self.assertEqual(msg.round_poly.shape, (4,))
        self.assertEqual(len(state), 5)
        self.assertEqual(state[0].shape, (4,))  # width halved — one round consumed

    def test_end_to_end_sumcheck_reduction(self) -> None:
        # Drive the round to completion; the running claim must reduce
        # consistently each round and match the final width-1 evaluation.
        width = 16
        st = _state(60, width)
        lam = jnp.array(11, KB)
        rnd = LogupSumcheckRound(lam)
        n = 4  # log2(16)

        # Replay round-by-round to check the per-round sumcheck identity, reducing
        # at the challenge actually sampled from the sponge each round.
        claim = jnp.sum(rnd._combine(*st))
        state = st
        t: Transcript = cheap_transcript(KB)
        for _ in range(n):
            state, t, msg = rnd(state, t)
            self.assertTrue(bool(msg.round_poly[0] + msg.round_poly[1] == claim))
            claim = eval_univariate(msg.round_poly, msg.challenge)
        self.assertTrue(bool(state[0].shape == (1,)))  # collapsed to a point
        self.assertTrue(bool(rnd._combine(*state)[0] == claim))

        # `fold_rounds` collects the round polys and the bound point, and the
        # homogeneous scan driver `prove` (what GkrLayerRound uses) is byte-identical
        # to that loop -- same Fiat-Shamir order over an identical fresh sponge, so
        # same round polys + point.
        _, _, msgs = fold_rounds(rnd, st, cheap_transcript(KB), n)
        round_polys = jnp.stack([m.round_poly for m in msgs])
        point = jnp.stack([m.challenge for m in msgs])
        self.assertEqual(round_polys.shape, (n, 4))
        _, _, scan_msgs = prove(rnd, st, cheap_transcript(KB))
        self.assertTrue(bool(jnp.all(scan_msgs.round_poly == round_polys)))
        self.assertTrue(bool(jnp.all(scan_msgs.challenge == point)))

    def test_scan_prove_is_flat_in_variable_count(self) -> None:
        # The per-variable LogUp loop scans rather than unrolls: the prove jaxpr
        # equation count is invariant under the variable count (#58), the compile
        # win the scan driver buys over the old unrolled fold_rounds. Trace only,
        # no execution, so it runs on any backend.
        def eqn_count(num_vars: int) -> int:
            st = _state(7, 1 << num_vars)
            jaxpr = jax.make_jaxpr(
                lambda s, t: prove(LogupSumcheckRound(jnp.array(3, KB)), s, t)
            )(st, cheap_transcript(KB))
            return len(jaxpr.jaxpr.eqns)

        self.assertEqual(eqn_count(3), eqn_count(5))

    def test_round_poly_is_fusion_ready(self) -> None:
        # Straight-line element-wise field ops + the one inherent Sigma; see
        # zorch.testkit.fusion (proxy for issue #21's ZorchRoundRewriter).
        st = _state(80, 8)
        assert_fusion_ready(
            LogupSumcheckRound(jnp.array(5, KB))._round_poly, st, reduces=1
        )

    def test_state_must_have_five_factors(self) -> None:
        rnd = LogupSumcheckRound(jnp.array(1, KB))
        # The summand guards arity; the scan driver reaches _combine directly, so
        # the guard lives there and _round_poly inherits it by delegation.
        with self.assertRaises(ValueError):
            rnd._combine(*_state(70, 8)[:4])
        with self.assertRaises(ValueError):
            rnd._round_poly(_state(70, 8)[:4])

    def test_combine_delegates_to_shared_logup_combine(self) -> None:
        # The round's _combine and the module-level logup_combine (which the GKR
        # verifier oracle also calls) must stay identical.
        lam = jnp.array(6, KB)
        eq, n0, d1, n1, d0 = (jnp.array(v, KB) for v in (2, 3, 4, 5, 7))
        self.assertTrue(
            bool(
                LogupSumcheckRound(lam)._combine(eq, n0, d1, n1, d0)
                == logup_combine(lam, eq, n0, d1, n1, d0)
            )
        )


class LogupSumcheckRoundPytreeTest(absltest.TestCase):
    """The round is a JAX pytree, so it threads through jit/vmap as an ARGUMENT
    rather than being closed over -- the `lam` leaf is what lets a layer's
    batching challenge be vmapped, which baking it into a constant cannot do."""

    def test_flatten_roundtrip(self) -> None:
        rnd = LogupSumcheckRound(jnp.array(7, KB))
        leaves, treedef = tree_util.tree_flatten(rnd)
        self.assertEqual(len(leaves), 1)  # lam is the only leaf
        self.assertTrue(bool(leaves[0] == jnp.array(7, KB)))
        self.assertTrue(bool(tree_util.tree_unflatten(treedef, leaves).lam == rnd.lam))

    def test_threads_through_jit_as_argument(self) -> None:
        rnd = LogupSumcheckRound(jnp.array(7, KB))
        st = _state(20, 8)
        got = jax.jit(lambda r, s: r._round_poly(s))(rnd, st)
        self.assertTrue(bool(jnp.all(got == rnd._round_poly(st))))

    def test_vmap_over_lam_batches_layers(self) -> None:
        # One vmap over distinct batching challenges, sharing the MLE state.
        st = _state(30, 8)
        lams = rand_field(99, (4,), KB)
        got = jax.vmap(lambda r, s: r._round_poly(s), in_axes=(0, None))(
            LogupSumcheckRound(lams), st
        )
        self.assertEqual(got.shape, (4, 4))  # (batch, degree+1)
        for i in range(4):
            want = LogupSumcheckRound(lams[i])._round_poly(st)
            self.assertTrue(bool(jnp.all(got[i] == want)))


class BindOutputTest(absltest.TestCase):
    def test_initial_carry_shapes(self) -> None:
        first = random_first_layer(7, 2, 3)
        output = extract_outputs(build_pyramid(first)[-1])
        (num_eval, den_eval, point), _ = bind_output(output, cheap_transcript(KB))
        self.assertEqual(num_eval.shape, ())
        self.assertEqual(den_eval.shape, ())
        # The output layer has num_batch_variables + 1 variables.
        self.assertEqual(point.shape, (first.num_batch_variables + 1,))


class GkrProverTest(absltest.TestCase):
    def test_chain_emits_one_proof_per_non_floor_layer(self) -> None:
        first = random_first_layer(11, 1, 2)
        layers, _, proofs, _ = prove_gkr(first)
        self.assertEqual(len(proofs), len(layers) - 1)
        for lp, layer in zip(proofs, reversed(layers[:-1]), strict=True):
            self.assertEqual(
                lp.round_polys.shape, (layer.num_variables, LogupSummand.DEGREE + 1)
            )
            openings = (
                lp.numerator_0,
                lp.numerator_1,
                lp.denominator_0,
                lp.denominator_1,
            )
            for opening in openings:
                self.assertEqual(opening.shape, ())  # final width-1 openings

    def test_first_layer_sumcheck_identity(self) -> None:
        # s(0) + s(1) of the first round equals the layer's claim lam*num + den.
        first = random_first_layer(13, 2, 2)
        layers = build_pyramid(first)
        output = extract_outputs(layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB))
        # The first layer round samples lam off this same transcript state; peeking
        # is non-destructive (sample returns a fresh transcript, leaving this one).
        _, lam = transcript.sample(1)
        claim = lam[0] * carry[0] + carry[1]
        chain = ProveChain([GkrLayerRound(layer) for layer in reversed(layers[:-1])])
        _, _, proofs = chain(carry, transcript)
        first_round = proofs[0].round_polys[0]
        self.assertTrue(bool(first_round[0] + first_round[1] == claim))

    def test_proof_records_bound_point(self) -> None:
        # The carry appends the child selector as the last bit, so the
        # retained point is the next carry's eval_point minus that bit — the
        # invariant a wire consumer reads the point through.
        first = random_first_layer(19, 2, 2)
        layers = build_pyramid(first)
        output = extract_outputs(layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB))
        (_, _, new_point), _, proof = GkrLayerRound(layers[-2])(carry, transcript)
        self.assertEqual(proof.point.shape, (new_point.shape[0] - 1,))
        self.assertTrue(bool(jnp.all(proof.point == new_point[:-1])))

    def test_deterministic_under_fixed_transcript(self) -> None:
        first = random_first_layer(17, 1, 3)
        _, _, a, _ = prove_gkr(first)
        _, _, b, _ = prove_gkr(first)
        for pa, pb in zip(a, b):
            self.assertTrue(bool(jnp.all(pa.round_polys == pb.round_polys)))

    def test_prove_under_jit_matches_eager(self) -> None:
        # prove_gkr_jitted fuses the whole prove into one program; guard that
        # fusing changes nothing — GkrLayer / cheap_transcript / fold_rounds are
        # all jit-traceable and the round polynomials are bit-identical to the
        # eager prove_gkr.
        first = random_first_layer(23, 2, 3)
        eager = prove_gkr(first)[2][-1].round_polys
        jitted = prove_gkr_jitted(
            first.numerator_0,
            first.numerator_1,
            first.denominator_0,
            first.denominator_1,
            first.num_batch_variables,
        )
        self.assertTrue(bool(jnp.all(eager == jitted)))


if __name__ == "__main__":
    absltest.main()
