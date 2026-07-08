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
from typing import cast

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array

from zorch.logup_gkr._jagged_types import _Planes
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    _interleave,
    extract_jagged_outputs,
    jagged_layer_transition,
)
from zorch.logup_gkr.jagged_prover import (
    JaggedGkrLayerRound,
    JaggedLayerProof,
    _flat_planes,
    _jagged_round_via_zone,
    _jagged_round_zone,
    _layer_plane_width,
    _observe_openings_and_fold,
    _padded_round_schedule,
    _padded_round_schedule_jax,
    _plane_window,
    _run_jagged_rounds_padded,
    _sample_lam_and_claim,
    prove_jagged_layer,
    prove_jagged_pyramid,
)
from zorch.logup_gkr.prover import Carry, bind_output
from zorch.logup_gkr.testing import (
    build_jagged_pyramid,
    mixed_field_jagged_layer,
    random_jagged_layer,
    virtual_planes,
)
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.poly.univariate import eval_coeffs
from zorch.round import ProveChain
from zorch.sumcheck.jagged.buffers import _pad_to_width
from zorch.sumcheck.jagged.rounds import _expand_eq_slice
from zorch.sumcheck.jagged.schedule import _check_row_space, _row_counts_operand
from zorch.sumcheck.jagged.types import RoundWidthCaps
from zorch.testkit.jit_cache import assert_single_trace
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript, Transcript, sample_challenge

KB = zk_dtypes.koalabear_mont
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
    eq = expand_eq_to_hypercube(z, jnp.ones((), z.dtype))
    return jnp.sum(_logup_combine(lam, eq, n0, d1, n1, d0))


def _caps_for(layer: JaggedGkrLayer) -> RoundWidthCaps:
    """A tight `RoundWidthCaps` for `layer` (the fixed-width round layout), the
    recipe `jagged_round_runner_test._caps(slack=False)` uses: round-0 even-padded
    height rounded up to a multiple of 4, the row-eq hypercube `1 << nrv`, and the
    interaction width `max(4, 1 << niv)`. `nrv` is the layer's minimal virtual row
    depth `ceil(log2(max(row_counts)))` -- exactly what a floor-outward chain grows
    the entry point to by the time it proves this layer. First needed here; reused
    by the padded/rolled tasks."""
    nrv = max(1, (max(layer.row_counts) - 1).bit_length())
    niv = layer.num_batch_variables
    round0 = sum(rc + rc % 2 for rc in layer.row_counts)
    return RoundWidthCaps(
        row=round0 + (-round0 % 4),
        eq_row=1 << nrv,
        interaction=max(4, 1 << niv),
    )


def _prove_layer_padded(
    layer: JaggedGkrLayer,
    challenge_limbs: int,
    caps: RoundWidthCaps,
    carry: Carry,
    transcript: Transcript,
    *,
    max_rounds: int,
) -> tuple[Carry, Transcript, JaggedLayerProof]:
    """Task-2 test shim: prove ONE layer through the fixed-`max_rounds` padded
    round body (`_run_jagged_rounds_padded`), no scan. Mirrors the Task-4 scan
    `step` per-layer head -- sample `lam`/claim, build the caps-width eq tables +
    the padded schedule + the round-order coords, run the padded rounds, then
    observe the openings and fold the carry -- but for one concrete layer, so
    `nrv`/`real_rounds` are host-known and the ragged proof slices with Python
    ints. Byte-identical to `_jagged_round_via_zone` on the same (layer, carry,
    transcript)."""
    niv = layer.num_batch_variables
    num_eval, den_eval, eval_point = carry
    dtype = num_eval.dtype
    nrv = _check_row_space(layer.row_counts, eval_point.shape[0], niv)
    real_rounds = nrv + niv
    t = cast(DuplexTranscript, transcript)
    t, lam, claim = _sample_lam_and_claim(t, num_eval, den_eval, challenge_limbs, dtype)
    # eq tables + planes laid into the caps buffers exactly as `_run_jagged_rounds`
    # lays them (zero tail past the live prefix).
    eq_row = _pad_to_width(_expand_eq_slice(eval_point, niv, row=True), caps.eq_row, 0)
    eq_int = _pad_to_width(
        _expand_eq_slice(eval_point, niv, row=False), caps.interaction, 0
    )
    planes = _Planes(
        *(
            _pad_to_width(a, caps.row, 0)
            for a in (
                layer.numerator_0,
                layer.numerator_1,
                layer.denominator_0,
                layer.denominator_1,
            )
        )
    )
    sched = _padded_round_schedule_jax(
        _row_counts_operand(layer.row_counts),
        jnp.asarray(nrv, jnp.int32),
        niv,
        max_rounds,
        caps.row,
    )
    # coords[rnd] = eval_point[real_rounds - 1 - rnd]: the round consumes the point
    # from the end (the bound point is the challenges reversed); surplus rounds
    # clamp to an in-bounds coordinate (their contribution is masked out).
    idx = real_rounds - 1 - jnp.arange(max_rounds)
    coords = eval_point[jnp.clip(idx, 0, eval_point.shape[0] - 1)]
    polys, chal, planes_final, round_t = _run_jagged_rounds_padded(
        planes,
        eq_row,
        eq_int,
        coords,
        lam,
        claim,
        t,
        sched,
        niv=niv,
        max_rounds=max_rounds,
        real_rounds=real_rounds,
        caps=caps,
        challenge_limbs=challenge_limbs,
    )
    t = cast(DuplexTranscript, round_t)
    # The padded run stacks over `max_rounds`; the live layer is the leading
    # `real_rounds` (its inactive tail is neutral). `chal` is already the bound
    # point (real challenges reversed, front-aligned).
    point = chal[:real_rounds]
    n0, n1, d0, d1 = (
        planes_final.n0,
        planes_final.n1,
        planes_final.d0,
        planes_final.d1,
    )
    proof = JaggedLayerProof(lam, claim, polys[:real_rounds], point, n0, n1, d0, d1)
    t, num_eval, den_eval, new_point = _observe_openings_and_fold(
        t, n0, n1, d0, d1, point, challenge_limbs, dtype
    )
    return (num_eval, den_eval, new_point), t, proof


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
        mixed = mixed_field_jagged_layer(7, self.ROW_COUNTS)
        all_ef = JaggedGkrLayer(
            mixed.numerator_0.astype(EF),
            mixed.numerator_1.astype(EF),
            mixed.denominator_0,
            mixed.denominator_1,
            self.ROW_COUNTS,
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


class PyramidScanProveTest(absltest.TestCase):
    """`prove_jagged_pyramid` is the #254 entry point: this task pins it as a
    byte-identical drop-in for `ProveChain(JaggedGkrLayerRound(l) for l in
    ...)` before the scan (landing in a later task) replaces its per-layer
    Python loop."""

    ROW_COUNTS = ChainedJaggedProveTest.ROW_COUNTS

    def _chain(
        self,
        layers: list[JaggedGkrLayer],
        carry: Carry,
        transcript: Transcript,
        challenge_limbs: int = 1,
    ) -> tuple[Carry, Transcript, list[JaggedLayerProof]]:
        chain = ProveChain(
            JaggedGkrLayerRound(l, challenge_limbs) for l in reversed(layers[:-1])
        )
        return chain(carry, transcript)

    def _assert_byte_match(
        self,
        got: tuple[Carry, Transcript, list[JaggedLayerProof]],
        want: tuple[Carry, Transcript, list[JaggedLayerProof]],
    ) -> None:
        got_c, got_t, got_p = got
        want_c, want_t, want_p = want
        for g, w in zip(got_p, want_p, strict=True):
            for f in fields(JaggedLayerProof):
                self.assertTrue(
                    bool(jnp.all(getattr(g, f.name) == getattr(w, f.name))),
                    f"proof field {f.name} mismatch",
                )
        for g, w in zip(got_c, want_c, strict=True):
            self.assertTrue(bool(jnp.all(g == w)), "carry mismatch")
        if not isinstance(got_t, DuplexTranscript) or not isinstance(
            want_t, DuplexTranscript
        ):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        gs, ws = got_t.state, want_t.state
        self.assertTrue(bool(jnp.all(gs.input_buffer == ws.input_buffer)))
        self.assertTrue(bool(jnp.all(gs.output_buffer == ws.output_buffer)))
        self.assertTrue(bool(jnp.all(gs.sponge_state == ws.sponge_state)))
        self.assertEqual(int(gs.in_pos), int(ws.in_pos))
        self.assertEqual(int(gs.out_pos), int(ws.out_pos))

    def test_entry_matches_chain_no_caps(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(7, self.ROW_COUNTS))
        output = extract_jagged_outputs(layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB))
        want = self._chain(layers, carry, transcript)
        got = prove_jagged_pyramid(
            list(reversed(layers[:-1])), carry, transcript, caps=None
        )
        self._assert_byte_match(got, want)

    def test_entry_matches_chain_base_field_first_layer(self) -> None:
        # The BF->EF carve (#254): the pyramid's first layer carries
        # base-field LogUp numerators under EF denominators (the
        # `BaseFieldNumeratorFirstLayerTest` shape). One `jagged_layer_transition`
        # fold combines through the EF denominators (n0*d1 + n1*d0), so every
        # layer above the first is already all-EF -- the carry the entry reads
        # off the floor is EF while the entry's last-proved layer (the
        # original first layer) is still base-field-numerator, so the entry
        # must take the carve's true branch.
        layers = build_jagged_pyramid(mixed_field_jagged_layer(7, self.ROW_COUNTS))
        output = extract_jagged_outputs(layers[-1])
        carry, transcript = bind_output(output, cheap_transcript(KB))
        reversed_layers = list(reversed(layers[:-1]))
        self.assertNotEqual(
            reversed_layers[-1].numerator_0.dtype,
            carry[0].dtype,
            "fixture must exercise the BF->EF carve",
        )
        want = self._chain(layers, carry, transcript, challenge_limbs=4)
        got = prove_jagged_pyramid(
            reversed_layers, carry, transcript, caps=None, challenge_limbs=4
        )
        self._assert_byte_match(got, want)

    def test_entry_rejects_host_fs(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(5, self.ROW_COUNTS))
        output = extract_jagged_outputs(layers[-1])
        _, t = bind_output(output, cheap_transcript(KB))
        if not isinstance(t, DuplexTranscript):
            raise AssertionError("bind_output must thread the DuplexTranscript back")
        t_host = DuplexTranscript.new(t.permutation, rate=t.rate, fs_on_host=True)
        carry, _ = bind_output(output, cheap_transcript(KB))
        with self.assertRaises(NotImplementedError):
            prove_jagged_pyramid(list(reversed(layers[:-1])), carry, t_host, caps=None)

    def test_padded_body_matches_natural_single_layer(self) -> None:
        # The soundness-critical core (Task 2): one layer proved at
        # `max_rounds > nrv + niv` must produce identical polys/challenges/openings
        # and leave the transcript in the identical state -- the surplus rounds are
        # inactive, so they must not advance Fiat-Shamir. A weaker transcript check
        # (fewer than all five DuplexState leaves) would miss an FS over-advance,
        # which is the whole point of the masking.
        layers = build_jagged_pyramid(random_jagged_layer(6, self.ROW_COUNTS))
        layer = layers[0]  # the deepest layer -- the most real rounds
        caps = _caps_for(layer)
        output = extract_jagged_outputs(layers[-1])
        carry, t = bind_output(output, cheap_transcript(KB))
        # Thread the carry outward through the interior layers so the deepest layer
        # enters its prove at its full row depth: `bind_output` seeds only the
        # floor's `niv` coordinates, and each interior layer's prove grows the
        # eval_point by one (the child selector).
        for outer in reversed(layers[1:-1]):
            carry, t, _ = _jagged_round_via_zone(outer, 1, _caps_for(outer), carry, t)
        niv = layer.num_batch_variables
        nrv = _check_row_space(layer.row_counts, carry[2].shape[0], niv)

        nat_carry, nat_t, nat_proof = _jagged_round_via_zone(layer, 1, caps, carry, t)
        pad_carry, pad_t, pad_proof = _prove_layer_padded(
            layer, 1, caps, carry, t, max_rounds=nrv + niv + 2
        )

        for f in fields(JaggedLayerProof):
            self.assertTrue(
                bool(jnp.all(getattr(pad_proof, f.name) == getattr(nat_proof, f.name))),
                f"proof field {f.name} diverged",
            )
        for g, w in zip(pad_carry, nat_carry, strict=True):
            self.assertTrue(bool(jnp.all(g == w)), "carry diverged")
        if not isinstance(pad_t, DuplexTranscript) or not isinstance(
            nat_t, DuplexTranscript
        ):
            raise AssertionError("both paths must thread the DuplexTranscript back")
        ps, ns = pad_t.state, nat_t.state
        self.assertTrue(bool(jnp.all(ps.input_buffer == ns.input_buffer)))
        self.assertTrue(bool(jnp.all(ps.output_buffer == ns.output_buffer)))
        self.assertTrue(bool(jnp.all(ps.sponge_state == ns.sponge_state)))
        self.assertEqual(int(ps.in_pos), int(ns.in_pos))
        self.assertEqual(int(ps.out_pos), int(ns.out_pos))

    def test_plane_window_matches_pad_to_width(self) -> None:
        layers = build_jagged_pyramid(random_jagged_layer(6, self.ROW_COUNTS))
        interior = list(reversed(layers[:-1]))
        dtype = interior[0].numerator_0.dtype
        plane_width = max(l.height for l in interior)
        flat_n0, *_, offsets, widths = _flat_planes(interior, dtype, plane_width)
        for j, layer in enumerate(interior):
            live = jnp.arange(plane_width) < widths[j]
            got = _plane_window(
                flat_n0, offsets[j], plane_width, live, jnp.zeros((), dtype)
            )
            want = _pad_to_width(layer.numerator_0.astype(dtype), plane_width, 0)
            self.assertTrue(bool(jnp.all(got == want)), f"layer {j} window mismatch")

    def test_scan_matches_chain_with_caps(self) -> None:
        # Task 4: the whole interior chain rolled into ONE lax.scan, forced on and
        # sized by the widest layer's caps -- byte-identical to the unrolled
        # ProveChain (every proof field, the full carry, all five sponge leaves).
        layers = build_jagged_pyramid(random_jagged_layer(9, self.ROW_COUNTS))
        caps = _caps_for(layers[0])
        output = extract_jagged_outputs(layers[-1])
        carry, t = bind_output(output, cheap_transcript(KB))
        want = self._chain(layers, carry, t)
        got = prove_jagged_pyramid(
            list(reversed(layers[:-1])), carry, t, caps=caps, force_scan=True
        )
        self._assert_byte_match(got, want)

    def test_scan_matches_chain_with_caps_taller_shape(self) -> None:
        # A second, deeper pyramid (more layers / larger max_rounds, taller odd
        # segments) under one caps: the scan is shape-invariant, so a narrower
        # layer runs at the widest layer's caps width -- the caps-padding a single
        # shape would leave untested (closes the Task-2 single-shape note).
        layers = build_jagged_pyramid(random_jagged_layer(3, (7, 3, 11, 5)))
        caps = _caps_for(layers[0])
        output = extract_jagged_outputs(layers[-1])
        carry, t = bind_output(output, cheap_transcript(KB))
        want = self._chain(layers, carry, t)
        got = prove_jagged_pyramid(
            list(reversed(layers[:-1])), carry, t, caps=caps, force_scan=True
        )
        self._assert_byte_match(got, want)

    def test_scan_matches_chain_base_field_first_layer(self) -> None:
        # The BF->EF carve UNDER the scan: caps + force_scan routes the all-EF
        # interior through `_scan_pyramid` (the carve recursion forwards
        # force_scan), while the base-field-numerator widest layer is carved to the
        # per-layer path and proved last. Pins that the carve's interior is
        # genuinely scanned (not the per-layer loop), on the mixed-field pyramid.
        layers = build_jagged_pyramid(mixed_field_jagged_layer(7, self.ROW_COUNTS))
        caps = _caps_for(layers[0])
        output = extract_jagged_outputs(layers[-1])
        carry, t = bind_output(output, cheap_transcript(KB))
        reversed_layers = list(reversed(layers[:-1]))
        self.assertNotEqual(
            reversed_layers[-1].numerator_0.dtype,
            carry[0].dtype,
            "fixture must exercise the BF->EF carve",
        )
        want = self._chain(layers, carry, t, challenge_limbs=4)
        got = prove_jagged_pyramid(
            reversed_layers, carry, t, caps=caps, challenge_limbs=4, force_scan=True
        )
        self._assert_byte_match(got, want)


class PaddedRoundScheduleJaxTest(absltest.TestCase):
    """`_padded_round_schedule_jax` (the runtime, `row_counts`-derived schedule the
    fixed-`max_rounds` body reads) byte-matches the readable numpy
    `_padded_round_schedule` oracle -- the derivation stays cross-checked against
    the host-baked construction, on the exact round count, with surplus inactive
    rounds, and across the peel-chain envelope (where `1 << k` nears int32
    overflow)."""

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
        # surplus rounds must reconstruct as the identity gather / zeros / False.
        for row_counts, nrv, niv in self.CASES:
            with self.subTest(row_counts=row_counts):
                self._check(row_counts, nrv, niv, nrv + niv + 3)

    def test_peel_chain_envelope_no_overflow(self) -> None:
        # A wide `max_rounds` (32): `bk = 1 << k` reaches 2^31 at k=31, where
        # `row_counts + (bk - 1)` would overflow int32. The saturate guard must
        # still byte-match the numpy oracle across the surplus rounds.
        for row_counts, nrv, niv in self.CASES:
            with self.subTest(row_counts=row_counts):
                self._check(row_counts, nrv, niv, 32)

    def test_under_jit_matches_eager(self) -> None:
        # The reconstruction must lower (no baked plane-width constant) and match
        # eager -- jitting it is the whole point.
        row_counts, nrv, niv = (3, 1, 5, 2), 3, 2
        plane_width = _layer_plane_width(row_counts, nrv, niv)
        max_rounds = nrv + niv + 2
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
                JaggedGkrLayerRound(layer)(carry, cheap_transcript(KB))

            return _call

        assert_single_trace(self, _jagged_round_zone, [make_call(s) for s in (7, 8, 9)])


if __name__ == "__main__":
    absltest.main()
