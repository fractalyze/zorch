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

import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest
from frx import Array

from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _interleave,
    extract_jagged_outputs,
    jagged_layer_transition,
)
from zorch.logup_gkr.jagged_prover import (
    JaggedGkrLayerRound,
    JaggedLayerProof,
    RoundWidthCaps,
    _jagged_round_zone,
    prove_jagged_layer,
)
from zorch.logup_gkr.prover import LayerClaim, bind_output
from zorch.logup_gkr.testing import (
    build_jagged_pyramid,
    caps_for,
    mixed_field_jagged_layer,
    random_jagged_layer,
    virtual_planes,
    widen_jagged_layer,
)
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.poly.univariate import eval_coeffs
from zorch.round import prove_rounds
from zorch.sumcheck.jagged.buffers import LayerBuffers
from zorch.testkit.jit_cache import assert_single_trace
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript, Transcript, sample_challenge

KB = zk_dtypes.koalabear_mont

# The transcript's own field: the schedule these tests pinned before the
# policy required an explicit field.
_CH = ChallengePolicy(KB)
EF = zk_dtypes.koalabearx4_mont


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
    eq = expand_eq_to_hypercube(z, fnp.ones((), z.dtype))
    return fnp.sum(_logup_combine(lam, eq, n0, d1, n1, d0))


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
        point, _, proof = prove_jagged_layer(
            layer,
            lam,
            claim,
            z,
            cheap_transcript(KB),
            caps=caps_for(self.ROW_COUNTS, self.NRV),
            challenges=_CH,
        )
        return layer, lam, z, claim, point, proof

    def test_first_round_matches_virtual_hypercube_claim(self) -> None:
        _, _, _, claim, _, proof = self._prove()
        p0 = proof.round_polys[0]
        zero = fnp.zeros((), KB)
        one = fnp.ones((), KB)
        self.assertTrue(bool(eval_coeffs(p0, zero) + eval_coeffs(p0, one) == claim))

    def test_round_polys_thread_claims(self) -> None:
        # Each round's s(0) + s(1) must equal the previous round's claim
        # reduction -- the per-round sumcheck identity over all five rounds.
        _, _, _, claim, point, proof = self._prove()
        zero = fnp.zeros((), KB)
        one = fnp.ones((), KB)
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
        point, _, proof = prove_jagged_layer(
            layer,
            lam,
            claim,
            z,
            cheap_transcript(KB),
            caps=caps_for((1, 1, 1, 1), 3),
            challenges=_CH,
        )
        p0 = proof.round_polys[0]
        zero = fnp.zeros((), KB)
        one = fnp.ones((), KB)
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
            layer,
            lam,
            claim,
            z,
            cheap_transcript(KB),
            challenges=ChallengePolicy(EF),
            caps=caps_for(self.ROW_COUNTS, self.NRV),
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
                fnp.array(3, KB),
                fnp.array(5, KB),
                rand_field(62, (1,), KB),  # only the interaction variable
                cheap_transcript(KB),
                caps=caps_for((1, 1), 1),
                challenges=_CH,
            )

    def test_rejects_missing_caps(self) -> None:
        # Row counts are traced, so there is no uncapped exact layout: a
        # caps-less prove must fail loudly, not fall back.
        layer = random_jagged_layer(71, (1, 1))
        with self.assertRaises(ValueError):
            prove_jagged_layer(
                layer,
                fnp.array(3, KB),
                fnp.array(5, KB),
                rand_field(72, (3,), KB),
                cheap_transcript(KB),
                challenges=_CH,
            )


class BaseFieldNumeratorFirstLayerTest(absltest.TestCase):
    """The first GKR layer may carry base-field numerators under extension-field
    denominators. `prove_jagged_layer`'s round 0 then reads the numerators in the
    base field; the fold lifts them to the extension field from round 1, so the
    whole sumcheck is byte-identical to proving an all-extension copy.
    That round-0 base-field read is the first-layer-numerator bandwidth an
    extension-everywhere layer leaves on the table."""

    ROW_COUNTS = (3, 1, 5, 2)
    NRV = 3

    def _layers(self) -> tuple[JaggedGkrLayer, JaggedGkrLayer]:
        # Same numbers, two encodings: base-field numerators vs the all-extension
        # copy (the base elements embedded via astype).
        mixed = mixed_field_jagged_layer(7, self.ROW_COUNTS)
        all_ef = JaggedGkrLayer(
            mixed.numerator_0.astype(EF),
            mixed.numerator_1.astype(EF),
            mixed.denominator_0,
            mixed.denominator_1,
            mixed.row_counts,
        )
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

        caps = caps_for(self.ROW_COUNTS, self.NRV)
        gp, gt, gproof = prove_jagged_layer(
            mixed,
            lam,
            claim,
            z,
            cheap_transcript(KB),
            challenges=ChallengePolicy(EF),
            caps=caps,
        )
        wp, wt, wproof = prove_jagged_layer(
            all_ef,
            lam,
            claim,
            z,
            cheap_transcript(KB),
            challenges=ChallengePolicy(EF),
            caps=caps,
        )

        self.assertTrue(bool(fnp.all(gp == wp)))  # bound point
        for f in fields(JaggedLayerProof):
            self.assertTrue(
                bool(fnp.all(getattr(gproof, f.name) == getattr(wproof, f.name))),
                f"proof.{f.name} diverged",
            )
        if not isinstance(gt, DuplexTranscript) or not isinstance(wt, DuplexTranscript):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        gs, ws = gt.state, wt.state
        self.assertTrue(bool(fnp.all(gs.input_buffer == ws.input_buffer)))
        self.assertTrue(bool(fnp.all(gs.output_buffer == ws.output_buffer)))
        self.assertTrue(bool(fnp.all(gs.sponge_state == ws.sponge_state)))
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

        (num_eval, den_eval, new_point), _, proof = JaggedGkrLayerRound(
            layer, caps=caps_for(row_counts, nrv), challenges=_CH
        )(carry, transcript)

        p0 = proof.round_polys[0]
        zero = fnp.zeros((), KB)
        one = fnp.ones((), KB)
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
        (_, _, new_point), _, proof = JaggedGkrLayerRound(
            layer, caps=caps_for((3, 1, 5, 2), 3), challenges=_CH
        )(carry, cheap_transcript(KB))
        self.assertEqual(proof.point.shape, (new_point.shape[0] - 1,))
        self.assertTrue(bool(fnp.all(proof.point == new_point[:-1])))

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

        _, _, proof = JaggedGkrLayerRound(
            layer, caps=caps_for((3, 1, 5, 2), 3), challenges=_CH
        )(carry, transcript)

        self.assertTrue(bool(proof.lam == lam))
        self.assertTrue(bool(proof.claim == lam * carry[0] + carry[1]))


class ChainedJaggedProveTest(absltest.TestCase):
    """The consumer shape from #154: a generator-built `prove_rounds` replaying
    the hand loop it replaces, releasing each layer once its round is proved."""

    ROW_COUNTS = (3, 1, 5, 2)

    def _hand_loop(
        self, layers: list[JaggedGkrLayer]
    ) -> tuple[LayerClaim, Transcript, list[JaggedLayerProof]]:
        """The hand-threaded layer loop the chain replaces: the reference
        stream, whose proofs carry the loop's own (lam, claim)."""
        caps = caps_for(self.ROW_COUNTS, len(layers) - 1)
        output = extract_jagged_outputs(layers[-1])
        (num_eval, den_eval, eval_point), transcript = bind_output(
            output, cheap_transcript(KB), challenges=_CH
        )
        proofs = []
        for layer in reversed(layers[:-1]):
            transcript, lam = sample_challenge(transcript, num_eval.dtype, 1)
            claim = lam * num_eval + den_eval
            point, transcript, proof = prove_jagged_layer(
                layer, lam, claim, eval_point, transcript, caps=caps, challenges=_CH
            )
            n0, n1 = proof.numerator_0, proof.numerator_1
            d0, d1 = proof.denominator_0, proof.denominator_1
            transcript = transcript.observe(fnp.stack([n0, n1, d0, d1]))
            transcript, r = sample_challenge(transcript, num_eval.dtype, 1)
            num_eval = n0 + (n1 - n0) * r
            den_eval = d0 + (d1 - d0) * r
            eval_point = fnp.concatenate([point, fnp.atleast_1d(r)])
            proofs.append(proof)
        return (num_eval, den_eval, eval_point), transcript, proofs

    def test_generator_chain_replays_the_hand_loop(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(7, self.ROW_COUNTS))
        want_carry, want_t, want_proofs = self._hand_loop(layers)

        output = extract_jagged_outputs(layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB), challenges=_CH)
        got_carry, got_t, got_proofs = prove_rounds(
            (
                JaggedGkrLayerRound(
                    layer,
                    caps=caps_for(self.ROW_COUNTS, len(layers) - 1),
                    challenges=_CH,
                )
                for layer in reversed(layers[:-1])
            ),
            carry,
            transcript,
        )

        # Identical proofs (every field, the (lam, claim) anchors included),
        # carry, and next challenge: the streams agree.
        for got, want in zip(got_proofs, want_proofs, strict=True):
            for field in fields(JaggedLayerProof):
                self.assertTrue(
                    bool(fnp.all(getattr(got, field.name) == getattr(want, field.name)))
                )
        for got, want in zip(got_carry, want_carry, strict=True):
            self.assertTrue(bool(fnp.all(got == want)))
        _, want_r = want_t.sample(1)
        _, got_r = got_t.sample(1)
        self.assertTrue(bool(got_r[0] == want_r[0]))

    def test_capped_chain_with_shared_layer_bufs_matches_fresh(self) -> None:
        # The chain-owned `LayerBuffers` holder: capped rounds lay every
        # layer's planes into the SAME donated cap-wide buffers, so later
        # layers write their live prefix over the previous layer's stale
        # tail — the stream must still match the holder-less chain (fresh
        # cap pads, no stale tails) byte-for-byte, because the rounds mask
        # every read past the live prefix.
        layers = build_jagged_pyramid(random_jagged_layer(7, self.ROW_COUNTS))
        output = extract_jagged_outputs(layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB), challenges=_CH)
        caps = RoundWidthCaps(elements=64, eq_row=64, interaction=8)
        want_carry, want_t, want_proofs = prove_rounds(
            (
                JaggedGkrLayerRound(layer, caps=caps, challenges=_CH)
                for layer in reversed(layers[:-1])
            ),
            carry,
            transcript,
        )

        bufs = LayerBuffers()
        got_carry, got_t, got_proofs = prove_rounds(
            (
                JaggedGkrLayerRound(layer, caps=caps, layer_bufs=bufs, challenges=_CH)
                for layer in reversed(layers[:-1])
            ),
            carry,
            transcript,
        )

        for got, want in zip(got_proofs, want_proofs, strict=True):
            for field in fields(JaggedLayerProof):
                self.assertTrue(
                    bool(fnp.all(getattr(got, field.name) == getattr(want, field.name)))
                )
        for got, want in zip(got_carry, want_carry, strict=True):
            self.assertTrue(bool(fnp.all(got == want)))
        _, want_r = want_t.sample(1)
        _, got_r = got_t.sample(1)
        self.assertTrue(bool(got_r[0] == want_r[0]))
        # The holder actually engaged: it owns laid cap-wide entries.
        self.assertTrue(bufs.pool)

    def test_chained_prove_keeps_at_most_one_layer_alive(self) -> None:
        # The #154 acceptance bound: of the layers handed to the chain, at
        # most one is alive at any point during the prove, and none survive
        # it -- the release semantics the hand loop's `layers[i] = None` had.
        layers = build_jagged_pyramid(random_jagged_layer(17, self.ROW_COUNTS))
        caps = caps_for(self.ROW_COUNTS, len(layers) - 1)
        output = extract_jagged_outputs(layers.pop())
        carry, transcript = bind_output(output, cheap_transcript(KB), challenges=_CH)

        yielded: list[weakref.ref[JaggedGkrLayer]] = []
        live_log: list[int] = []

        def rounds() -> Iterator[JaggedGkrLayerRound]:
            while layers:
                live_log.append(sum(ref() is not None for ref in yielded))
                layer = layers.pop()
                yielded.append(weakref.ref(layer))
                yield JaggedGkrLayerRound(layer, caps=caps, challenges=_CH)

        prove_rounds(rounds(), carry, transcript)

        # At each build, only the round just proved could still be alive.
        self.assertEqual(live_log, [0] + [1] * (len(yielded) - 1))
        self.assertEqual([ref() for ref in yielded], [None] * len(yielded))


class JaggedGkrLayerRoundZoneTest(absltest.TestCase):
    """`JaggedGkrLayerRound` dispatches every layer through the module-level
    `_jagged_round_zone`, which keys on layer *shape*. A consumer that rebuilds the
    chain each warm iter (the generator giving lazy one-live-layer release) builds a
    fresh round per layer per iter, so those same-shape rounds must reuse one trace --
    a per-instance jit would re-trace every pyramid layer on every iteration.

    (Byte-equality of the jitted round loop vs the unrolled eager oracle is the
    round-runner gate's job -- see `jagged_round_runner_test`.)"""

    # niv = 1 (two interactions), nrv = 2: an odd segment (3) and a saturated
    # one (1), both row and interaction rounds, in a few-element layer.
    ROW_COUNTS = (3, 1)
    NRV = 2

    def test_fresh_rounds_over_one_shape_share_a_single_trace(self) -> None:
        # The consumer's warm loop rebuilds the chain every prove iteration (the
        # generator giving lazy one-live-layer release), so it builds a fresh round
        # per layer per iter. The module-level zone keys on layer shape, not
        # instance, so those same-shape rounds reuse one trace; a per-instance jit
        # would re-trace every pyramid layer on every iteration. Distinct seeds
        # (different values, one shape) pin shape-keying, not value/identity-keying.
        carry = (
            rand_field(111, (), KB),
            rand_field(112, (), KB),
            rand_field(113, (self.NRV + 1,), KB),
        )

        def make_call(seed: int) -> Callable[[], None]:
            def _call() -> None:
                layer = random_jagged_layer(seed, self.ROW_COUNTS)
                JaggedGkrLayerRound(
                    layer, caps=caps_for(self.ROW_COUNTS, self.NRV), challenges=_CH
                )(carry, cheap_transcript(KB))

            return _call

        assert_single_trace(self, _jagged_round_zone, [make_call(s) for s in (7, 8, 9)])


class CapacityRoundTest(absltest.TestCase):
    """A widened layer (capacity slack, dead-zero tail) proved through
    `JaggedGkrLayerRound` must byte-match its zero-slack twin, and different
    layouts pre-laid to one caps width must share the zone executable."""

    ROW_COUNTS = (3, 1, 5, 2)  # niv = 2; max rc 5 fits nrv = 3
    NRV = 3
    CAPS = RoundWidthCaps(elements=16, eq_row=8, interaction=8)

    def _carry(self, seed: int) -> LayerClaim:
        return (
            rand_field(seed, (), KB),
            rand_field(seed + 1, (), KB),
            rand_field(seed + 2, (self.NRV + 2,), KB),
        )

    def _assert_stream_equal(
        self,
        got: tuple[LayerClaim, Transcript, JaggedLayerProof],
        want: tuple[LayerClaim, Transcript, JaggedLayerProof],
    ) -> None:
        got_carry, got_t, got_proof = got
        want_carry, want_t, want_proof = want
        for field in fields(JaggedLayerProof):
            self.assertTrue(
                bool(
                    fnp.all(
                        getattr(got_proof, field.name)
                        == getattr(want_proof, field.name)
                    )
                ),
                f"{field.name} diverged",
            )
        for g, w in zip(got_carry, want_carry, strict=True):
            self.assertTrue(bool(fnp.all(g == w)))
        _, want_r = want_t.sample(1)
        _, got_r = got_t.sample(1)
        self.assertTrue(bool(got_r[0] == want_r[0]))

    def test_matches_static_capped_round(self) -> None:
        layer = random_jagged_layer(410, self.ROW_COUNTS)
        carry = self._carry(420)
        want = JaggedGkrLayerRound(layer, caps=self.CAPS, challenges=_CH)(
            carry, cheap_transcript(KB)
        )
        wide = widen_jagged_layer(layer, layer.width + 3)
        got = JaggedGkrLayerRound(wide, caps=self.CAPS, challenges=_CH)(
            carry, cheap_transcript(KB)
        )
        self._assert_stream_equal(got, want)

    def test_matches_static_on_mixed_field_first_layer(self) -> None:
        layer = mixed_field_jagged_layer(430, self.ROW_COUNTS)
        carry = (
            rand_ext_field(440, (), KB, EF),
            rand_ext_field(441, (), KB, EF),
            rand_ext_field(442, (self.NRV + 2,), KB, EF),
        )
        want = JaggedGkrLayerRound(
            layer, challenges=ChallengePolicy(EF), caps=self.CAPS
        )(carry, cheap_transcript(KB))
        wide = widen_jagged_layer(layer, layer.width + 3)
        got = JaggedGkrLayerRound(wide, challenges=ChallengePolicy(EF), caps=self.CAPS)(
            carry, cheap_transcript(KB)
        )
        self._assert_stream_equal(got, want)

    def test_requires_caps(self) -> None:
        layer = random_jagged_layer(450, self.ROW_COUNTS)
        with self.assertRaises(ValueError):
            JaggedGkrLayerRound(layer, challenges=_CH)(
                self._carry(460), cheap_transcript(KB)
            )

    def test_widened_chain_matches_zero_slack_chain(self) -> None:
        # End to end through the pyramid: the zero-slack chain and the
        # widened chain (capacity slack + shared LayerBuffers) must emit one
        # byte stream. Exercises the capacity seams at once: the in-trace
        # live triples, the buffer pre-lay, and the dead-zero tails riding
        # under the pooled cap buffers.
        static_layers = build_jagged_pyramid(random_jagged_layer(470, self.ROW_COUNTS))
        output = extract_jagged_outputs(static_layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB), challenges=_CH)

        want_bufs = LayerBuffers()
        want = prove_rounds(
            (
                JaggedGkrLayerRound(
                    layer, caps=self.CAPS, layer_bufs=want_bufs, challenges=_CH
                )
                for layer in reversed(static_layers[:-1])
            ),
            carry,
            transcript,
        )

        capped_layers = [
            widen_jagged_layer(layer, layer.width + 2) for layer in static_layers[:-1]
        ]
        got_bufs = LayerBuffers()
        got = prove_rounds(
            (
                JaggedGkrLayerRound(
                    layer, caps=self.CAPS, layer_bufs=got_bufs, challenges=_CH
                )
                for layer in reversed(capped_layers)
            ),
            carry,
            transcript,
        )

        want_carry, want_t, want_proofs = want
        got_carry, got_t, got_proofs = got
        for got_p, want_p in zip(got_proofs, want_proofs, strict=True):
            for field in fields(JaggedLayerProof):
                self.assertTrue(
                    bool(
                        fnp.all(
                            getattr(got_p, field.name) == getattr(want_p, field.name)
                        )
                    ),
                    f"{field.name} diverged",
                )
        for g, w in zip(got_carry, want_carry, strict=True):
            self.assertTrue(bool(fnp.all(g == w)))
        _, want_r = want_t.sample(1)
        _, got_r = got_t.sample(1)
        self.assertTrue(bool(got_r[0] == want_r[0]))

    def test_shares_zone_trace_across_layouts_and_widths(self) -> None:
        # One executable serves: two different layouts at one widened
        # capacity AND the zero-slack dispatch -- pre-laid to caps width,
        # all three zone calls carry identical operand shapes, so any
        # same-caps consumer rides a warm cache with zero new compiles.
        carry = self._carry(480)
        bufs = LayerBuffers()

        def capped_call(seed: int, rc: tuple[int, ...]) -> Callable[[], None]:
            def _call() -> None:
                layer = widen_jagged_layer(random_jagged_layer(seed, rc), 14)
                JaggedGkrLayerRound(
                    layer, caps=self.CAPS, layer_bufs=bufs, challenges=_CH
                )(carry, cheap_transcript(KB))

            return _call

        def zero_slack_call() -> None:
            layer = random_jagged_layer(490, self.ROW_COUNTS)
            JaggedGkrLayerRound(layer, caps=self.CAPS, layer_bufs=bufs, challenges=_CH)(
                carry, cheap_transcript(KB)
            )

        assert_single_trace(
            self,
            _jagged_round_zone,
            [
                capped_call(500, (3, 1, 5, 2)),
                capped_call(501, (2, 2, 7, 1)),
                zero_slack_call,
            ],
        )


if __name__ == "__main__":
    absltest.main()
