# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for the jagged LogUp-GKR prover.

Every check is anchored against the brute-force virtual-dense expansion
(`testing.virtual_planes`): the prover's closed-form virtual-mass corrections
must reproduce what a full `2^(niv+nrv)` materialization would sum. The
prove->verify roundtrip lives in jagged_verifier_test; this pins the prover's
own invariants.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from dataclasses import fields

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array
from jaxlib.mlir.dialects import stablehlo

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _interleave,
    extract_jagged_outputs,
    jagged_layer_transition,
)
from zorch.logup_gkr.jagged_prover import (
    _DEGREE,
    JaggedGkrLayerRound,
    JaggedLayerProof,
    _expand_eq_prefix,
    _jagged_round_zone,
    _layer_plane_width,
    _padded_round_schedule,
    _padded_round_schedule_jax,
    _round_metadata,
    _run_jagged_rounds,
    _run_jagged_rounds_padded,
    prove_jagged_layer,
    prove_jagged_pyramid,
)
from zorch.logup_gkr.prover import Carry, bind_output
from zorch.logup_gkr.testing import (
    build_jagged_pyramid,
    random_jagged_layer,
    virtual_planes,
)
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.poly.univariate import compute_inv_vandermonde, eval_coeffs
from zorch.round import ProveChain
from zorch.sumcheck.prover import SUMCHECK_MARKER, SUMCHECK_MARKER_VERSION
from zorch.testkit.jit_cache import assert_single_trace
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript, Transcript, sample_challenge

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont
_HAS_COMPOSITE_OP = hasattr(stablehlo, "CompositeOp")


def _logup_combine(
    lam: Array, eq: Array, n0: Array, d1: Array, n1: Array, d0: Array
) -> Array:
    """The LogUp summand, restated locally: the oracle must not import the
    production combine, or a drift there would leave these tests silently
    self-consistent."""
    return eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1)


def _virtual_claim(layer: JaggedGkrLayer, nrv: int, lam: Array, z: Array) -> Array:
    """Brute-force `sum_x eq(z, x) * combine(x)` over the virtual hypercube."""
    n0, n1, d0, d1 = virtual_planes(layer, nrv)
    eq = expand_eq_to_hypercube(z, jnp.ones((), z.dtype))
    return jnp.sum(_logup_combine(lam, eq, n0, d1, n1, d0))


class ProveJaggedLayerTest(absltest.TestCase):
    # Jagged shape exercising all padding paths: odd segments (3, 5), an
    # already-saturated segment (1), and an even one (2); nrv = 3 embeds the
    # max count 5 into 8 virtual rows.
    ROW_COUNTS = (3, 1, 5, 2)
    NRV = 3

    def _prove(
        self, seed: int = 7
    ) -> tuple[JaggedGkrLayer, Array, Array, Array, Array, JaggedLayerProof]:
        layer = random_jagged_layer(seed, self.ROW_COUNTS)
        lam = rand_field(seed + 10, (), KB)
        z = rand_field(seed + 11, (self.NRV + 2,), KB)
        claim = _virtual_claim(layer, self.NRV, lam, z)
        point, _, proof = prove_jagged_layer(layer, lam, claim, z, cheap_transcript(KB))
        return layer, lam, z, claim, point, proof

    def test_first_round_matches_virtual_hypercube_claim(self) -> None:
        _, _, _, claim, _, proof = self._prove()
        p0 = proof.round_polys[0]
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        self.assertTrue(bool(eval_coeffs(p0, zero) + eval_coeffs(p0, one) == claim))

    def test_round_polys_thread_claims(self) -> None:
        # Each round's s(0) + s(1) must equal the previous round's claim
        # reduction -- the per-round sumcheck identity over all five rounds.
        _, _, _, claim, point, proof = self._prove()
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        for poly, r in zip(proof.round_polys, point[::-1], strict=True):
            self.assertTrue(
                bool(eval_coeffs(poly, zero) + eval_coeffs(poly, one) == claim)
            )
            claim = eval_coeffs(poly, r)

    def test_final_openings_are_virtual_plane_evals(self) -> None:
        layer, _, _, _, point, proof = self._prove()
        n0, n1, d0, d1 = virtual_planes(layer, self.NRV)
        for got, plane in (
            (proof.numerator_0, n0),
            (proof.numerator_1, n1),
            (proof.denominator_0, d0),
            (proof.denominator_1, d1),
        ):
            self.assertTrue(bool(got == eval_mle(plane, point)))

    def test_final_claim_matches_oracle(self) -> None:
        # The fully-reduced claim equals the combine at the bound point with
        # eq evaluated in closed form -- the verifier's final check.
        _, lam, z, claim, point, proof = self._prove()
        for poly, r in zip(proof.round_polys, point[::-1], strict=True):
            claim = eval_coeffs(poly, r)
        want = _logup_combine(
            lam,
            eval_eq(z, point),
            proof.numerator_0,
            proof.denominator_1,
            proof.numerator_1,
            proof.denominator_0,
        )
        self.assertTrue(bool(claim == want))

    def test_saturated_layer_folds_pure_virtual_rows(self) -> None:
        # One materialized row per segment under nrv=3: seven of every eight
        # virtual rows are non-materialized, so the closed-form corrections
        # carry almost the whole sum.
        layer = random_jagged_layer(21, (1, 1, 1, 1))
        lam = rand_field(31, (), KB)
        z = rand_field(32, (5,), KB)
        claim = _virtual_claim(layer, 3, lam, z)
        point, _, proof = prove_jagged_layer(layer, lam, claim, z, cheap_transcript(KB))
        p0 = proof.round_polys[0]
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        self.assertTrue(bool(eval_coeffs(p0, zero) + eval_coeffs(p0, one) == claim))
        n0, _, _, d1 = virtual_planes(layer, 3)
        self.assertTrue(bool(proof.numerator_0 == eval_mle(n0, point)))
        self.assertTrue(bool(proof.denominator_1 == eval_mle(d1, point)))

    def test_multi_limb_challenges_extend_the_transcript_field(self) -> None:
        # A base-field transcript with extension-field claims: every
        # challenge takes four squeezes reinterpreted as the extension
        # element. The oracle identity must close in the extension field.
        layer = random_jagged_layer(41, self.ROW_COUNTS)
        lam = rand_ext_field(51, (), KB, EF)
        z = rand_ext_field(52, (self.NRV + 2,), KB, EF)
        claim = _virtual_claim(layer, self.NRV, lam, z)
        point, _, proof = prove_jagged_layer(
            layer, lam, claim, z, cheap_transcript(KB), challenge_limbs=4
        )
        self.assertEqual(point.dtype, EF)
        for poly, r in zip(proof.round_polys, point[::-1], strict=True):
            claim = eval_coeffs(poly, r)
        want = _logup_combine(
            lam,
            eval_eq(z, point),
            proof.numerator_0,
            proof.denominator_1,
            proof.numerator_1,
            proof.denominator_0,
        )
        self.assertTrue(bool(claim == want))

    def test_proof_records_lam_and_entry_claim(self) -> None:
        _, lam, _, claim, _, proof = self._prove()
        self.assertTrue(bool(proof.lam == lam))
        self.assertTrue(bool(proof.claim == claim))

    def test_rejects_missing_row_variable(self) -> None:
        layer = random_jagged_layer(61, (1, 1))
        with self.assertRaises(ValueError):
            prove_jagged_layer(
                layer,
                jnp.array(3, KB),
                jnp.array(5, KB),
                rand_field(62, (1,), KB),  # only the interaction variable
                cheap_transcript(KB),
            )

    def test_rejects_row_count_beyond_virtual_space(self) -> None:
        layer = random_jagged_layer(71, (5, 1))
        with self.assertRaises(ValueError):
            prove_jagged_layer(
                layer,
                jnp.array(3, KB),
                jnp.array(5, KB),
                rand_field(72, (3,), KB),  # nrv = 2 < log2(5)
                cheap_transcript(KB),
            )


class BaseFieldNumeratorFirstLayerTest(absltest.TestCase):
    """The first GKR layer may carry base-field numerators under extension-field
    denominators. `prove_jagged_layer`'s round 0 then reads the numerators in the
    base field; the fold lifts them to the extension field from round 1, so the
    whole sumcheck is byte-identical to proving an all-extension copy (zkx#681).
    That round-0 base-field read is the first-layer-numerator bandwidth an
    extension-everywhere layer leaves on the table."""

    ROW_COUNTS = (3, 1, 5, 2)
    NRV = 3

    def _layers(self) -> tuple[JaggedGkrLayer, JaggedGkrLayer]:
        # Same numbers, two encodings: base-field numerators vs the all-extension
        # copy (the base elements embedded via astype).
        height = sum(self.ROW_COUNTS)
        n0 = rand_field(7, (height,), KB)
        n1 = rand_field(8, (height,), KB)
        d0 = rand_ext_field(9, (height,), KB, EF)
        d1 = rand_ext_field(10, (height,), KB, EF)
        mixed = JaggedGkrLayer(n0, n1, d0, d1, self.ROW_COUNTS)
        all_ef = JaggedGkrLayer(n0.astype(EF), n1.astype(EF), d0, d1, self.ROW_COUNTS)
        return mixed, all_ef

    def test_layer_enters_with_base_field_numerators(self) -> None:
        # The optimization's premise: numerators base, denominators already EF.
        mixed, _ = self._layers()
        self.assertEqual(mixed.numerator_0.dtype, KB)
        self.assertEqual(mixed.numerator_1.dtype, KB)
        self.assertEqual(mixed.denominator_0.dtype, EF)

    def test_prove_matches_all_ef_byte_for_byte(self) -> None:
        mixed, all_ef = self._layers()
        lam = rand_ext_field(51, (), KB, EF)
        z = rand_ext_field(52, (self.NRV + 2,), KB, EF)
        claim = _virtual_claim(all_ef, self.NRV, lam, z)

        gp, gt, gproof = prove_jagged_layer(
            mixed, lam, claim, z, cheap_transcript(KB), challenge_limbs=4
        )
        wp, wt, wproof = prove_jagged_layer(
            all_ef, lam, claim, z, cheap_transcript(KB), challenge_limbs=4
        )

        self.assertTrue(bool(jnp.all(gp == wp)))  # bound point
        for f in fields(JaggedLayerProof):
            self.assertTrue(
                bool(jnp.all(getattr(gproof, f.name) == getattr(wproof, f.name))),
                f"proof.{f.name} diverged",
            )
        if not isinstance(gt, DuplexTranscript) or not isinstance(wt, DuplexTranscript):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        gs, ws = gt.state, wt.state
        self.assertTrue(bool(jnp.all(gs.input_buffer == ws.input_buffer)))
        self.assertTrue(bool(jnp.all(gs.output_buffer == ws.output_buffer)))
        self.assertTrue(bool(jnp.all(gs.sponge_state == ws.sponge_state)))
        self.assertEqual(int(gs.in_pos), int(ws.in_pos))
        self.assertEqual(int(gs.out_pos), int(ws.out_pos))


class JaggedGkrLayerRoundTest(absltest.TestCase):
    def test_carry_reduction_closes_the_gkr_relation(self) -> None:
        # claim = lam * N(z) + D(z) with N/D read off the *folded* layer's
        # interleaved virtual planes must reduce, through the sumcheck, to
        # the input layer's interleaved planes at the new carry point -- the
        # cross-layer GKR identity on jagged layers.
        row_counts = (3, 1, 5, 2)
        nrv = 3
        layer = random_jagged_layer(81, row_counts)
        schedule = (2, 2, 4, 2)  # over-pads some segments past the fold
        folded = jagged_layer_transition(layer, schedule)

        z = rand_field(91, (nrv + 2,), KB)
        fn0, fn1, fd0, fd1 = virtual_planes(folded, nrv - 1)
        carry = (
            eval_mle(_interleave(fn0, fn1), z),
            eval_mle(_interleave(fd0, fd1), z),
            z,
        )
        transcript = cheap_transcript(KB)
        # Peek lam off the same transcript state (sample is non-destructive
        # on this one; the round re-derives it).
        _, lam = sample_challenge(transcript, KB, 1)

        (num_eval, den_eval, new_point), _, proof = JaggedGkrLayerRound(layer)(
            carry, transcript
        )

        p0 = proof.round_polys[0]
        zero = jnp.zeros((), KB)
        one = jnp.ones((), KB)
        self.assertTrue(
            bool(
                eval_coeffs(p0, zero) + eval_coeffs(p0, one)
                == lam * carry[0] + carry[1]
            )
        )
        self.assertEqual(new_point.shape, (nrv + 3,))
        n0, n1, d0, d1 = virtual_planes(layer, nrv)
        self.assertTrue(bool(num_eval == eval_mle(_interleave(n0, n1), new_point)))
        self.assertTrue(bool(den_eval == eval_mle(_interleave(d0, d1), new_point)))

    def test_proof_records_bound_point(self) -> None:
        # The carry appends the child selector as the last bit, so the
        # retained point is the next carry's eval_point minus that bit — the
        # invariant a wire consumer reads the point through.
        layer = random_jagged_layer(121, (3, 1, 5, 2))
        carry = (
            rand_field(131, (), KB),
            rand_field(132, (), KB),
            rand_field(133, (5,), KB),
        )
        (_, _, new_point), _, proof = JaggedGkrLayerRound(layer)(
            carry, cheap_transcript(KB)
        )
        self.assertEqual(proof.point.shape, (new_point.shape[0] - 1,))
        self.assertTrue(bool(jnp.all(proof.point == new_point[:-1])))

    def test_proof_records_lam_and_opening_claim(self) -> None:
        # The per-layer anchors a consumer diffs when a transcript diverges
        # mid-pyramid: the round's sampled lam and the opening claim it
        # batched from the carry.
        layer = random_jagged_layer(101, (3, 1, 5, 2))
        carry = (
            rand_field(111, (), KB),
            rand_field(112, (), KB),
            rand_field(113, (5,), KB),
        )
        transcript = cheap_transcript(KB)
        # Peek lam off the same transcript state (sample is non-destructive
        # on this one; the round re-derives it).
        _, lam = sample_challenge(transcript, KB, 1)

        _, _, proof = JaggedGkrLayerRound(layer)(carry, transcript)

        self.assertTrue(bool(proof.lam == lam))
        self.assertTrue(bool(proof.claim == lam * carry[0] + carry[1]))


class ChainedJaggedProveTest(absltest.TestCase):
    """The consumer shape from #154: a generator-built `ProveChain` replaying
    the hand loop it replaces, releasing each layer once its round is proved."""

    ROW_COUNTS = (3, 1, 5, 2)

    def _hand_loop(
        self, layers: list[JaggedGkrLayer]
    ) -> tuple[Carry, Transcript, list[JaggedLayerProof]]:
        """The hand-threaded layer loop the chain replaces: the reference
        stream, whose proofs carry the loop's own (lam, claim)."""
        output = extract_jagged_outputs(layers[-1])
        (num_eval, den_eval, eval_point), transcript = bind_output(
            output, cheap_transcript(KB)
        )
        proofs = []
        for layer in reversed(layers[:-1]):
            transcript, lam = sample_challenge(transcript, num_eval.dtype, 1)
            claim = lam * num_eval + den_eval
            point, transcript, proof = prove_jagged_layer(
                layer, lam, claim, eval_point, transcript
            )
            n0, n1 = proof.numerator_0, proof.numerator_1
            d0, d1 = proof.denominator_0, proof.denominator_1
            transcript = transcript.observe(jnp.stack([n0, n1, d0, d1]))
            transcript, r = sample_challenge(transcript, num_eval.dtype, 1)
            num_eval = n0 + (n1 - n0) * r
            den_eval = d0 + (d1 - d0) * r
            eval_point = jnp.concatenate([point, jnp.atleast_1d(r)])
            proofs.append(proof)
        return (num_eval, den_eval, eval_point), transcript, proofs

    def test_generator_chain_replays_the_hand_loop(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(7, self.ROW_COUNTS))
        want_carry, want_t, want_proofs = self._hand_loop(layers)

        output = extract_jagged_outputs(layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB))
        chain = ProveChain(
            JaggedGkrLayerRound(layer) for layer in reversed(layers[:-1])
        )
        got_carry, got_t, got_proofs = chain(carry, transcript)

        # Identical proofs (every field, the (lam, claim) anchors included),
        # carry, and next challenge: the streams agree.
        for got, want in zip(got_proofs, want_proofs, strict=True):
            for field in fields(JaggedLayerProof):
                self.assertTrue(
                    bool(jnp.all(getattr(got, field.name) == getattr(want, field.name)))
                )
        for got, want in zip(got_carry, want_carry, strict=True):
            self.assertTrue(bool(jnp.all(got == want)))
        _, want_r = want_t.sample(1)
        _, got_r = got_t.sample(1)
        self.assertTrue(bool(got_r[0] == want_r[0]))

    def test_chained_prove_keeps_at_most_one_layer_alive(self) -> None:
        # The #154 acceptance bound: of the layers handed to the chain, at
        # most one is alive at any point during the prove, and none survive
        # it -- the release semantics the hand loop's `layers[i] = None` had.
        layers = build_jagged_pyramid(random_jagged_layer(17, self.ROW_COUNTS))
        output = extract_jagged_outputs(layers.pop())
        carry, transcript = bind_output(output, cheap_transcript(KB))

        yielded: list[weakref.ref[JaggedGkrLayer]] = []
        live_log: list[int] = []

        def rounds() -> Iterator[JaggedGkrLayerRound]:
            while layers:
                live_log.append(sum(ref() is not None for ref in yielded))
                layer = layers.pop()
                yielded.append(weakref.ref(layer))
                yield JaggedGkrLayerRound(layer)

        ProveChain(rounds())(carry, transcript)

        # At each build, only the round just proved could still be alive.
        self.assertEqual(live_log, [0] + [1] * (len(yielded) - 1))
        self.assertEqual([ref() for ref in yielded], [None] * len(yielded))


class ProveJaggedMarkedTest(absltest.TestCase):
    """Over a transcript whose permutation has a dedicated fusion marker (real
    poseidon2), `prove_jagged_layer` wraps its round loop in a `zorch.sumcheck`
    composite with Fiat-Shamir INSIDE. Unrecognized -- every CPU path here -- the
    marker decomposes to the same `_run_jagged_rounds`, so the proof, the bound
    point, and the advanced transcript are byte-identical to the plain loop; the
    recognized -> register-resident GPU path is exercised in zkx (zkx#544)."""

    ROW_COUNTS = (3, 1, 5, 2)
    NRV = 3

    def _poseidon_transcript(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def _plain(
        self,
        layer: JaggedGkrLayer,
        lam: Array,
        claim: Array,
        z: Array,
        challenge_limbs: int = 1,
    ) -> tuple[Array, Transcript, JaggedLayerProof]:
        # The unmarked decomposition over a fresh identical transcript: the same
        # setup `prove_jagged_layer` does, then `_run_jagged_rounds` directly,
        # bypassing the dedicated-fusion gate the poseidon transcript would trip.
        niv = layer.num_interaction_variables
        nrv = z.shape[0] - niv
        one = jnp.ones((), z.dtype)
        eq_row = expand_eq_to_hypercube(z[niv:], one)
        eq_int = expand_eq_to_hypercube(z[:niv], one)
        meta = _round_metadata(layer.row_counts, nrv)
        naturals = jnp.stack([jnp.array(j, z.dtype) for j in range(_DEGREE + 1)])
        inv_vand = compute_inv_vandermonde(_DEGREE, z.dtype)
        challenges, t, polys, fn0, fn1, fd0, fd1 = _run_jagged_rounds(
            layer.numerator_0,
            layer.numerator_1,
            layer.denominator_0,
            layer.denominator_1,
            eq_row,
            eq_int,
            z,
            lam,
            claim,
            self._poseidon_transcript(),
            meta,
            naturals,
            inv_vand,
            nrv,
            niv,
            challenge_limbs,
        )
        return (
            challenges,
            t,
            JaggedLayerProof(lam, claim, polys, challenges, fn0, fn1, fd0, fd1),
        )

    def _virtual_claim(self, layer: JaggedGkrLayer, lam: Array, z: Array) -> Array:
        n0, n1, d0, d1 = virtual_planes(layer, self.NRV)
        eq = expand_eq_to_hypercube(z, jnp.ones((), z.dtype))
        return jnp.sum(_logup_combine(lam, eq, n0, d1, n1, d0))

    def _assert_marked_equals_plain(
        self, layer: JaggedGkrLayer, lam: Array, z: Array, challenge_limbs: int = 1
    ) -> None:
        claim = self._virtual_claim(layer, lam, z)
        gp, gt, gproof = prove_jagged_layer(
            layer,
            lam,
            claim,
            z,
            self._poseidon_transcript(),
            challenge_limbs=challenge_limbs,
        )
        wp, wt, wproof = self._plain(layer, lam, claim, z, challenge_limbs)
        self.assertTrue(bool(jnp.all(gp == wp)))  # bound point
        for f in fields(JaggedLayerProof):
            self.assertTrue(
                bool(jnp.all(getattr(gproof, f.name) == getattr(wproof, f.name))),
                f"proof.{f.name} diverged",
            )
        if not isinstance(gt, DuplexTranscript) or not isinstance(wt, DuplexTranscript):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        gs, ws = gt.state, wt.state
        self.assertTrue(bool(jnp.all(gs.input_buffer == ws.input_buffer)))
        self.assertTrue(bool(jnp.all(gs.output_buffer == ws.output_buffer)))
        self.assertTrue(bool(jnp.all(gs.sponge_state == ws.sponge_state)))
        self.assertEqual(int(gs.in_pos), int(ws.in_pos))
        self.assertEqual(int(gs.out_pos), int(ws.out_pos))

    def test_marked_equals_plain_base(self) -> None:
        layer = random_jagged_layer(7, self.ROW_COUNTS)
        self._assert_marked_equals_plain(
            layer, rand_field(17, (), KB), rand_field(18, (self.NRV + 2,), KB)
        )

    def test_marked_equals_plain_multi_limb_ef(self) -> None:
        # koalabearx4 challenges (four squeezes reinterpreted) through the marker.
        layer = random_jagged_layer(41, self.ROW_COUNTS)
        self._assert_marked_equals_plain(
            layer,
            rand_ext_field(51, (), KB, EF),
            rand_ext_field(52, (self.NRV + 2,), KB, EF),
            challenge_limbs=4,
        )

    def test_cheap_transcript_stays_unmarked(self) -> None:
        layer = random_jagged_layer(7, self.ROW_COUNTS)
        lam, z = rand_field(17, (), KB), rand_field(18, (self.NRV + 2,), KB)
        claim = self._virtual_claim(layer, lam, z)
        # Return only the bound point (a JAX array): the marker check only needs
        # the jaxpr's primitives, and the point alone keeps the traced output small.
        jaxpr = jax.make_jaxpr(
            lambda l_, c_, z_: prove_jagged_layer(
                layer, l_, c_, z_, cheap_transcript(KB)
            )[0]
        )(lam, claim, z).jaxpr
        self.assertFalse(any(e.primitive.name == "composite" for e in jaxpr.eqns))

    @absltest.skipUnless(_HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp")
    def test_marker_envelope_carries_jagged_attributes(self) -> None:
        # The recognition contract off the jaxpr: bare name + version, the shape in
        # composite.attributes, and the jagged row_counts vector / fold-order /
        # poly-form declarations the dense scalar path lacks.
        layer = random_jagged_layer(7, self.ROW_COUNTS)
        lam, z = rand_field(17, (), KB), rand_field(18, (self.NRV + 2,), KB)
        claim = self._virtual_claim(layer, lam, z)
        t0 = self._poseidon_transcript()
        jaxpr = jax.make_jaxpr(
            lambda l_, c_, z_: prove_jagged_layer(layer, l_, c_, z_, t0)[0]
        )(lam, claim, z).jaxpr
        eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
        self.assertEqual(eqn.params["name"], SUMCHECK_MARKER)
        self.assertEqual(eqn.params["version"], SUMCHECK_MARKER_VERSION)
        attrs = {k: leaves[0] for k, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(int(attrs["degree"]), _DEGREE)
        self.assertEqual(int(attrs["num_vars"]), self.NRV + 2)
        self.assertEqual(attrs["fold_order"], "lsb")
        self.assertEqual(attrs["poly_form"], "coefficient")
        # row_counts rides as an array<i64> attribute (lax.composite wraps it in a
        # HashableArray); the dense scalar `num_real` path carries no such vector.
        row_counts = attrs["row_counts"]
        row_counts = getattr(row_counts, "val", row_counts)
        self.assertEqual([int(c) for c in row_counts], list(self.ROW_COUNTS))


class JaggedGkrLayerRoundJitTest(absltest.TestCase):
    """`JaggedGkrLayerRound(layer, jit=True)` compiles the per-layer prove once
    and dispatches the cached executable on later calls -- the lever that turns
    the ~20-layer pyramid's per-call composite re-trace into a single trace per
    layer. The jit boundary must be a pure dispatch change: carry, proof, and
    advanced transcript byte-identical to the eager round, on both the plain
    (cheap transcript) and the marked-composite (poseidon) paths the consumer
    drives.

    A small base-field layer over the cheap permutation keeps this fast: jit
    forces XLA to *compile* the round, and compiling the marked path's
    `zorch.sumcheck` composite -- whose body runs the poseidon2 Fiat-Shamir
    permutation -- is a multi-minute XLA CPU-backend compile regardless of
    layer size, so the marked path is not unit-testable here. It does not need
    to be: jit wraps `_run` identically on both paths (a pure dispatch-time
    change), so `jit(marked) == eager(marked)` follows from this test
    (`jit == eager` on the loop) composed with `ProveJaggedMarkedTest`
    (`marked == eager plain`); the full-scale jitted marked prove is validated
    on GPU by the sp1-zorch bench's byte-match anchors."""

    # niv = 1 (two interactions), nrv = 2: an odd segment (3) and a saturated
    # one (1), both row and interaction rounds, in a few-element layer.
    ROW_COUNTS = (3, 1)
    NRV = 2

    def _run(
        self, *, jit: bool, transcript: Transcript
    ) -> tuple[tuple[Array, Array, Array], Transcript, JaggedLayerProof]:
        layer = random_jagged_layer(7, self.ROW_COUNTS)
        carry = (
            rand_field(111, (), KB),
            rand_field(112, (), KB),
            rand_field(113, (self.NRV + 1,), KB),  # niv = 1 for two interactions
        )
        round_ = JaggedGkrLayerRound(layer, jit=jit)
        return round_(carry, transcript)

    def test_jit_matches_eager_plain_path(self) -> None:
        # Cheap transcript -> unmarked loop, jitted; byte-identical to eager.
        (gnum, gden, gpt), gt, gproof = self._run(
            jit=True, transcript=cheap_transcript(KB)
        )
        (wnum, wden, wpt), wt, wproof = self._run(
            jit=False, transcript=cheap_transcript(KB)
        )
        self.assertTrue(bool(jnp.all(gnum == wnum)))
        self.assertTrue(bool(jnp.all(gden == wden)))
        self.assertTrue(bool(jnp.all(gpt == wpt)))
        for f in fields(JaggedLayerProof):
            self.assertTrue(
                bool(jnp.all(getattr(gproof, f.name) == getattr(wproof, f.name))),
                f"proof.{f.name} diverged under jit",
            )
        if not isinstance(gt, DuplexTranscript) or not isinstance(wt, DuplexTranscript):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        gs, ws = gt.state, wt.state
        self.assertTrue(bool(jnp.all(gs.input_buffer == ws.input_buffer)))
        self.assertTrue(bool(jnp.all(gs.output_buffer == ws.output_buffer)))
        self.assertTrue(bool(jnp.all(gs.sponge_state == ws.sponge_state)))
        self.assertEqual(int(gs.in_pos), int(ws.in_pos))
        self.assertEqual(int(gs.out_pos), int(ws.out_pos))

    def test_fresh_rounds_over_one_shape_share_a_single_trace(self) -> None:
        # The consumer that motivates jit=True rebuilds the chain every warm
        # prove iteration (the generator giving lazy one-live-layer release),
        # so it builds a fresh round per layer per iter. The module-level zone
        # keys on layer shape, not instance, so those same-shape rounds reuse
        # one trace; a per-instance jit would re-trace every pyramid layer on
        # every iteration. Distinct seeds (different values, one shape) pin
        # shape-keying, not value- or identity-keying.
        carry = (
            rand_field(111, (), KB),
            rand_field(112, (), KB),
            rand_field(113, (self.NRV + 1,), KB),
        )

        def make_call(seed: int) -> Callable[[], None]:
            def _call() -> None:
                layer = random_jagged_layer(seed, self.ROW_COUNTS)
                JaggedGkrLayerRound(layer, jit=True)(carry, cheap_transcript(KB))

            return _call

        assert_single_trace(self, _jagged_round_zone, [make_call(s) for s in (7, 8, 9)])


class PaddedJaggedRoundsTest(absltest.TestCase):
    """`_run_jagged_rounds_padded` is the fixed-width round loop the pyramid scan
    runs once per layer; over-padding it (a `max_rounds`/`plane_width` larger than
    one layer needs, the way a short floor layer rides a chain sized for the
    tallest) must reproduce the eager `_run_jagged_rounds` byte-for-byte. This
    isolates the hardest mechanic -- the masked materialized sum and the
    transcript-neutral inactive rounds -- from the outer scan."""

    def _assert_padded_equals_eager(
        self, row_counts: tuple[int, ...], nrv: int, max_rounds: int, plane_width: int
    ) -> None:
        layer = random_jagged_layer(7, row_counts)
        niv = layer.num_interaction_variables
        z = rand_field(18, (nrv + niv,), KB)
        lam = rand_field(17, (), KB)
        claim = rand_field(19, (), KB)
        one = jnp.ones((), KB)
        naturals = jnp.stack([jnp.array(j, KB) for j in range(_DEGREE + 1)])
        inv_vand = compute_inv_vandermonde(_DEGREE, KB)

        # Eager reference over the exactly-sized layout.
        eq_row = expand_eq_to_hypercube(z[niv:], one)
        eq_int = expand_eq_to_hypercube(z[:niv], one)
        e_ch, e_t, e_polys, *e_open = _run_jagged_rounds(
            layer.numerator_0,
            layer.numerator_1,
            layer.denominator_0,
            layer.denominator_1,
            eq_row,
            eq_int,
            z,
            lam,
            claim,
            cheap_transcript(KB),
            _round_metadata(layer.row_counts, nrv),
            naturals,
            inv_vand,
            nrv,
            niv,
            1,
        )

        # Fixed-width inputs: neutral-pad the planes, expand eq_row into the max
        # row width, build the round-order coordinates (eval_point reversed).
        pad = plane_width - layer.height
        planes = [
            jnp.concatenate(
                [arr, (jnp.zeros if neutral == 0 else jnp.ones)((pad,), KB)]
            )
            for arr, neutral in (
                (layer.numerator_0, 0),
                (layer.numerator_1, 0),
                (layer.denominator_0, 1),
                (layer.denominator_1, 1),
            )
        ]
        eq_row_p = _expand_eq_prefix(
            jnp.concatenate([z[niv:], jnp.zeros((max_rounds - niv - nrv,), KB)]),
            jnp.asarray(nrv, jnp.int32),
            1 << (max_rounds - niv),
            one,
        )
        sched = {
            k: jnp.asarray(v)
            for k, v in _padded_round_schedule(
                layer.row_counts, nrv, niv, max_rounds, plane_width
            ).items()
        }
        idx = (nrv + niv) - 1 - jnp.arange(max_rounds)
        coords = z[jnp.clip(idx, 0, z.shape[0] - 1)]

        p_ch, p_t, p_polys, *p_open = _run_jagged_rounds_padded(
            planes[0],
            planes[1],
            planes[2],
            planes[3],
            eq_row_p,
            eq_int,
            coords,
            lam,
            claim,
            cheap_transcript(KB),
            sched,
            naturals,
            inv_vand,
            niv,
            max_rounds,
            1,
        )

        # The real challenges land reversed at the tail of the fixed buffer; the
        # real polys are its leading prefix.
        rounds = nrv + niv
        self.assertTrue(bool(jnp.all(e_ch == p_ch[max_rounds - rounds :])))
        self.assertTrue(bool(jnp.all(e_polys == p_polys[:rounds])))
        for got, want in zip(p_open, e_open, strict=True):
            self.assertTrue(bool(got == want))
        if not isinstance(e_t, DuplexTranscript) or not isinstance(
            p_t, DuplexTranscript
        ):
            raise AssertionError("both paths thread the DuplexTranscript back")
        es, ps = e_t.state, p_t.state
        self.assertTrue(bool(jnp.all(es.input_buffer == ps.input_buffer)))
        self.assertTrue(bool(jnp.all(es.output_buffer == ps.output_buffer)))
        self.assertTrue(bool(jnp.all(es.sponge_state == ps.sponge_state)))
        self.assertEqual(int(es.in_pos), int(ps.in_pos))
        self.assertEqual(int(es.out_pos), int(ps.out_pos))

    def test_exact_fit_matches_eager(self) -> None:
        # No over-padding: max_rounds == nrv + niv, plane_width == round-0 height.
        self._assert_padded_equals_eager(
            (3, 1, 5, 2), nrv=3, max_rounds=5, plane_width=14
        )

    def test_over_padded_short_layer_matches_eager(self) -> None:
        # A short layer (nrv=1) run in a chain sized for nrv=3: extra inactive
        # rounds and a wider plane buffer must stay transcript-neutral and
        # byte-identical -- the pyramid's floor-layer case.
        self._assert_padded_equals_eager(
            (1, 1, 2, 1), nrv=1, max_rounds=5, plane_width=14
        )
        self._assert_padded_equals_eager(
            (2, 1, 4, 1), nrv=2, max_rounds=5, plane_width=14
        )


class RolledJaggedPyramidTest(absltest.TestCase):
    """`prove_jagged_pyramid` rolls the floor-outward layer chain into one
    `lax.scan`; it must be byte-identical to the unrolled `ProveChain` over
    `JaggedGkrLayerRound`s -- the sp1-zorch#55 gate. Same fixture as
    `ChainedJaggedProveTest` so the two share a reference stream."""

    ROW_COUNTS = (3, 1, 5, 2)

    def test_rolled_equals_unrolled_chain(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(7, self.ROW_COUNTS))
        proved = list(reversed(layers[:-1]))
        output = extract_jagged_outputs(layers[-1])

        carry_u, t_u = bind_output(output, cheap_transcript(KB))
        want_carry, want_t, want_proofs = ProveChain(
            JaggedGkrLayerRound(layer) for layer in proved
        )(carry_u, t_u)

        carry_r, t_r = bind_output(output, cheap_transcript(KB))
        got_carry, got_t, got_proofs = prove_jagged_pyramid(proved, carry_r, t_r)

        for got, want in zip(got_proofs, want_proofs, strict=True):
            for field in fields(JaggedLayerProof):
                self.assertTrue(
                    bool(
                        jnp.all(getattr(got, field.name) == getattr(want, field.name))
                    ),
                    f"proof.{field.name} diverged",
                )
        for got, want in zip(got_carry, want_carry, strict=True):
            self.assertTrue(bool(jnp.all(got == want)))
        _, want_r = want_t.sample(1)
        _, got_r = got_t.sample(1)
        self.assertTrue(bool(got_r[0] == want_r[0]))

    def test_no_baked_plane_width_schedule(self) -> None:
        # #109: the rolled scan reconstructs its schedule from row_counts in-body,
        # so tracing it bakes NO plane-width int32 schedule constant (the ~GB
        # constant that blocked real-shard lowering). The planes are EF (not
        # int32), so the only int32 const is the compact per-layer row-count
        # channel -- a regression to the host-baked schedule would dwarf it.
        import numpy as _np

        layers = build_jagged_pyramid(random_jagged_layer(7, self.ROW_COUNTS))
        proved = list(reversed(layers[:-1]))
        output = extract_jagged_outputs(layers[-1])
        carry, t = bind_output(output, cheap_transcript(KB))
        jaxpr = jax.make_jaxpr(lambda c, tr: prove_jagged_pyramid(proved, c, tr))(
            carry, t
        )
        max_i32 = max(
            (
                _np.asarray(c).size
                for c in jaxpr.consts
                if _np.asarray(c).dtype == _np.int32
            ),
            default=0,
        )
        # row-count channel is len(proved) * nseg; the dropped schedule was
        # len(proved) * max_rounds * plane_width -- orders of magnitude larger.
        self.assertLess(max_i32, len(proved) * len(self.ROW_COUNTS) * 8)


class PaddedRoundScheduleJaxTest(absltest.TestCase):
    """`_padded_round_schedule_jax` (runtime row_counts, no baked plane-width
    arrays) byte-matches the numpy `_padded_round_schedule` -- the #109 gate that
    lets the rolled scan drop `xs_sched` (sp1-zorch#55)."""

    CASES = [
        ((3, 1), 2, 1),  # odd seg (3) + saturated (1)
        ((3, 1, 5, 2), 3, 2),  # the multi-segment golden
        ((7, 3, 11, 5), 4, 2),  # taller, two odd segs > 1
        ((2, 1, 4, 1), 2, 1),  # odd segs all length 1
        ((1, 7, 1, 9), 3, 2),  # saturated floors, tall odd segs
        ((4, 2, 6, 3), 3, 2),
    ]

    def _check(
        self, row_counts: tuple[int, ...], nrv: int, niv: int, max_rounds: int
    ) -> None:
        plane_width = _layer_plane_width(row_counts, nrv, niv)
        want = _padded_round_schedule(row_counts, nrv, niv, max_rounds, plane_width)
        got = _padded_round_schedule_jax(
            jnp.asarray(row_counts, jnp.int32),
            jnp.asarray(nrv, jnp.int32),
            niv,
            max_rounds,
            plane_width,
        )
        for key in want:
            self.assertTrue(
                bool(jnp.all(jnp.asarray(got[key]) == jnp.asarray(want[key]))),
                f"{key} diverged: row_counts={row_counts} nrv={nrv} niv={niv} "
                f"max_rounds={max_rounds}",
            )

    def test_exact_rounds_match(self) -> None:
        for row_counts, nrv, niv in self.CASES:
            with self.subTest(row_counts=row_counts):
                self._check(row_counts, nrv, niv, nrv + niv)

    def test_with_inactive_rounds_match(self) -> None:
        # max_rounds > nrv+niv (the envelope a shorter pyramid layer rides): the
        # surplus rounds must reconstruct as numpy's identity gather / zeros / False.
        for row_counts, nrv, niv in self.CASES:
            with self.subTest(row_counts=row_counts):
                self._check(row_counts, nrv, niv, nrv + niv + 3)

    def test_under_jit_matches_eager(self) -> None:
        # The reconstruction must lower (no baked plane-width constant) and match
        # eager -- jitting it is the whole point.
        row_counts, nrv, niv = (3, 1, 5, 2), 3, 2
        max_rounds, plane_width = nrv + niv + 2, _layer_plane_width(
            row_counts, nrv, niv
        )
        fn = jax.jit(
            lambda rc, n: _padded_round_schedule_jax(
                rc, n, niv, max_rounds, plane_width
            )
        )
        got = fn(jnp.asarray(row_counts, jnp.int32), jnp.asarray(nrv, jnp.int32))
        want = _padded_round_schedule(row_counts, nrv, niv, max_rounds, plane_width)
        for key in want:
            self.assertTrue(
                bool(jnp.all(jnp.asarray(got[key]) == jnp.asarray(want[key]))), key
            )


if __name__ == "__main__":
    absltest.main()
