# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from typing import Any

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.hash.poseidon2.poseidon2 import POSEIDON2_MARKER
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.logup_gkr.prover import LogupSumcheckRound
from zorch.sumcheck import prover
from zorch.sumcheck.prover import (
    SUMCHECK_COMBINE_MARKER,
    SUMCHECK_MARKER,
    SUMCHECK_MARKER_VERSION,
    _prove_scan,
    prove,
)
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import DuplexTranscript

KB = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont
_GPU_BACKEND = jax.default_backend() == "gpu"


class ProveTest(absltest.TestCase):
    def test_rejects_zero_round_state(self) -> None:
        # A width-1 carry derives 0 rounds: the scan would yield no round polys.
        # Fail fast with a clear message instead.
        with self.assertRaisesRegex(ValueError, "at least one round"):
            prove(
                prover.SumcheckRound(1), [jnp.arange(1, dtype=KB)], cheap_transcript(KB)
            )

    def test_stacks_sumcheck_messages(self) -> None:
        f = jnp.arange(1, 17, dtype=KB)
        _, _, msgs = prove(prover.SumcheckRound(degree=1), [f], cheap_transcript(KB))
        self.assertEqual(msgs.round_poly.shape, (4, 2))  # n rounds × (degree+1)
        self.assertEqual(msgs.challenge.shape, (4,))  # one challenge per round


class ProveMarkedTest(absltest.TestCase):
    """Over a transcript whose permutation has a dedicated fusion marker (real
    poseidon2), `prove` wraps its scan in a `zorch.sumcheck` composite with
    Fiat-Shamir INSIDE. Unrecognized — every CPU path here — the marker decomposes
    to the exact same scan, so the folded state, advanced transcript, AND proof are
    byte-identical to the plain `_prove_scan`; the recognized->fused GPU path is
    exercised in zkx. The gate keeps a test `CheapPermutation` on the plain scan.

    Base field is X.1's bar — extension-field (koalabearx4) byte-identity is the
    GPU fusion's gate (X.2, zkx), where the EF sponge is the real path; the marker
    here threads operands dtype-blind, so the base-field cases prove the wrapping."""

    def _poseidon_transcript(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def _assert_marked_equals_scan(
        self,
        round: prover.SumcheckSummand,
        factors: list[Any],
        **prove_kwargs: Any,
    ) -> None:
        # Fiat-Shamir is sampled inside the marker from the threaded sponge, so the
        # marked path must match the plain scan on the folded state, the advanced
        # transcript, AND the proof — all driven from an identical fresh transcript.
        # `prove_kwargs` forward the round-shape controls (`eval_start`,
        # `challenge_dtype`, `challenge_limbs`) to both paths so the non-default
        # marker is checked against the same-shaped scan.
        got_state, got_t, got_msgs = prove(
            round, list(factors), self._poseidon_transcript(), **prove_kwargs
        )
        want_state, want_t, want_msgs = _prove_scan(
            round, list(factors), self._poseidon_transcript(), **prove_kwargs
        )
        self.assertTrue(bool(jnp.all(got_msgs.round_poly == want_msgs.round_poly)))
        self.assertTrue(bool(jnp.all(got_msgs.challenge == want_msgs.challenge)))
        # GkrLayerRound reads the folded final state for its layer openings; option
        # A keeps it (the proof-only `prove_fs` of the closed #117 dropped it).
        self.assertEqual(len(got_state), len(want_state))
        for g, w in zip(got_state, want_state):
            self.assertTrue(bool(jnp.all(g == w)))
        if not isinstance(got_t, DuplexTranscript) or not isinstance(
            want_t, DuplexTranscript
        ):
            raise AssertionError("prove must thread the DuplexTranscript back")
        gs, ws = got_t.state, want_t.state
        self.assertTrue(bool(jnp.all(gs.sponge_state == ws.sponge_state)))
        self.assertTrue(bool(jnp.all(gs.output_buffer == ws.output_buffer)))
        self.assertEqual(int(gs.in_pos), int(ws.in_pos))
        self.assertEqual(int(gs.out_pos), int(ws.out_pos))

    def test_marked_equals_scan_product_base(self) -> None:
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        self._assert_marked_equals_scan(prover.SumcheckRound(degree=2), [a, b])

    def test_marked_equals_scan_single_mle_base(self) -> None:
        f = rand_field(43, (1 << 5,), KB)
        self._assert_marked_equals_scan(prover.SumcheckRound(degree=1), [f])

    def test_marked_equals_scan_logup_combine_base(self) -> None:
        # The non-product LogUp summand `eq*(λ*(n0*d1+n1*d0)+d0*d1)` through the same
        # marker: proves the combine is generic (carried by the nested
        # `zorch.sumcheck.combine` region) and that the λ scalar threads through as a
        # marker operand byte-identically — degree 3 over 5 factors, so degree !=
        # num_factors, unlike product.
        factors = [rand_field(50 + i, (1 << 4,), KB) for i in range(5)]
        self._assert_marked_equals_scan(LogupSumcheckRound(jnp.array(7, KB)), factors)

    def test_marked_equals_scan_truncated_base(self) -> None:
        # `eval_start=1` (the SWIRL `{s(1)..}` wire form) rides the marker too: its
        # decomposition is the identical truncated scan. Base field, so it executes
        # on every backend.
        a = rand_field(44, (1 << 4,), KB)
        b = rand_field(45, (1 << 4,), KB)
        self._assert_marked_equals_scan(
            prover.SumcheckRound(degree=2), [a, b], eval_start=1
        )

    @absltest.skipIf(
        _GPU_BACKEND,
        "cuda-pjrt aborts compiling koalabearx4 EF reductions; "
        "remove when fractalyze/prime-ir#332 lands",
    )
    def test_marked_equals_scan_truncated_extension(self) -> None:
        # The full openvm SWIRL shape — `{s(1), s(2)}` round polys folded by EF
        # challenges drawn from a base sponge — through the marker, byte-identical
        # to the plain scan. The marker threads operands dtype-blind, so this is the
        # extension-field decomposition gate (CPU; the recognized GPU path is zkx).
        a = rand_field(46, (1 << 4,), KB).astype(EF)
        b = rand_field(47, (1 << 4,), KB).astype(EF)
        self._assert_marked_equals_scan(
            prover.SumcheckRound(degree=2),
            [a, b],
            eval_start=1,
            challenge_dtype=EF,
            challenge_limbs=4,
        )

    def test_cheap_transcript_stays_unmarked(self) -> None:
        # has_dedicated_fusion=False keeps the gate shut: no composite, plain scan.
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        t0 = cheap_transcript(KB)
        jaxpr = jax.make_jaxpr(
            lambda x, y: prove(prover.SumcheckRound(degree=2), [x, y], t0)
        )(a, b).jaxpr
        self.assertFalse(any(e.primitive.name == "composite" for e in jaxpr.eqns))

    def test_marker_envelope_carries_shape_attributes(self) -> None:
        # Recognition contract asserted off the jaxpr (no lowering): name is the
        # bare routing key, version gates the revision, and degree/num_vars/
        # num_factors ride in composite.attributes. Results are [2 folded factors]
        # [5 transcript leaves][round polys][challenges]; the poseidon round
        # constants auto-lift, so the marker carries more operands than the
        # [2 factors][5 leaves] explicit.
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()  # build the sponge eagerly, not under trace
        jaxpr = jax.make_jaxpr(lambda x, y: prove(rnd, [x, y], t0))(a, b).jaxpr
        eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
        self.assertEqual(eqn.params["name"], SUMCHECK_MARKER)
        self.assertEqual(eqn.params["version"], SUMCHECK_MARKER_VERSION)
        attrs = {k: leaves[0] for k, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(int(attrs["degree"]), 2)
        self.assertEqual(int(attrs["num_vars"]), 4)
        self.assertEqual(int(attrs["num_factors"]), 2)
        # `num_real` is optional: a dense (unpadded) prove emits no such attr, so
        # a recognizer that predates it sees an unchanged envelope.
        self.assertNotIn("num_real", attrs)
        self.assertEqual(len(eqn.outvars), 2 + 5 + 2)  # folded, leaves, polys, chal
        self.assertGreater(len(eqn.invars), 2 + 5)  # RC auto-lifted past explicit

    def test_marker_carries_num_real_attribute(self) -> None:
        # A jagged factor table zero-pads to the next power of two; `num_real`
        # declares the real prefix length so a vendor bounds the first round's
        # reduction to it instead of sweeping the padded tail.
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()
        jaxpr = jax.make_jaxpr(lambda x, y: prove(rnd, [x, y], t0, num_real=10))(
            a, b
        ).jaxpr
        eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
        attrs = {k: leaves[0] for k, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(int(attrs["num_real"]), 10)

    def test_truncated_round_carries_eval_start_attr(self) -> None:
        # `eval_start=1` must still ride the `zorch.sumcheck` marker (not fall to the
        # plain scan), so a vendor codegens the truncated round register-resident.
        # The domain rides as an `eval_start` attr the recognizer keys off; the
        # marker stays revision 1 (the recognizer reads the attr, not the version).
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()
        jaxpr = jax.make_jaxpr(lambda x, y: prove(rnd, [x, y], t0, eval_start=1))(
            a, b
        ).jaxpr
        eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
        self.assertEqual(eqn.params["name"], SUMCHECK_MARKER)
        self.assertEqual(eqn.params["version"], SUMCHECK_MARKER_VERSION)
        attrs = {k: leaves[0] for k, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(int(attrs["eval_start"]), 1)
        # `eval_start` is the only non-default axis carried as an attr; the marker
        # never emits a `challenge_limbs` attr (the extension fold is inferred from
        # the challenge dtype downstream).
        self.assertEqual(int(attrs["degree"]), 2)
        self.assertNotIn("challenge_limbs", attrs)

    def test_extension_challenge_round_carries_no_attr(self) -> None:
        # An EF fold challenge (eval_start=0, full domain) adds no marker attr: a
        # vendor infers the extension fold from the EF-typed challenge result. The
        # envelope is the bare degree/num_vars/num_factors with no eval_start /
        # challenge_limbs — the EF rides in the operand / result dtypes.
        a = rand_field(40, (1 << 4,), KB).astype(EF)
        b = rand_field(41, (1 << 4,), KB).astype(EF)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()
        jaxpr = jax.make_jaxpr(
            lambda x, y: prove(rnd, [x, y], t0, challenge_dtype=EF, challenge_limbs=4)
        )(a, b).jaxpr
        eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
        self.assertEqual(eqn.params["version"], SUMCHECK_MARKER_VERSION)
        attrs = {k: leaves[0] for k, leaves, _ in eqn.params["attributes"]}
        self.assertNotIn("eval_start", attrs)
        self.assertNotIn("challenge_limbs", attrs)

    def test_default_round_carries_no_eval_start(self) -> None:
        # The default round's envelope is unchanged — version 1 with no `eval_start`
        # / `challenge_limbs` attrs — so a recognizer that predates `eval_start`
        # keeps codegen'ing it exactly as before.
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()
        jaxpr = jax.make_jaxpr(lambda x, y: prove(rnd, [x, y], t0))(a, b).jaxpr
        eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
        self.assertEqual(eqn.params["version"], SUMCHECK_MARKER_VERSION)
        attrs = {k: leaves[0] for k, leaves, _ in eqn.params["attributes"]}
        self.assertNotIn("eval_start", attrs)
        self.assertNotIn("challenge_limbs", attrs)

    def test_num_real_is_metadata_only(self) -> None:
        # The attr never reaches the computation: the marked call's
        # decomposition jaxpr is identical with and without it, so the bounded
        # prove stays byte-identical by construction (bounding the reduction
        # is the vendor's move, sound because the padded tail sums to zero).
        # Structural, not executed — the dense marker's executed byte-identity
        # is test_marked_equals_scan_product_base's job, and an executed check
        # here would pay a full scan compile only to rerun the same program.
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()

        def marker_jaxpr(**kw: int) -> str:
            jaxpr = jax.make_jaxpr(lambda x, y: prove(rnd, [x, y], t0, **kw))(
                a, b
            ).jaxpr
            eqn = next(e for e in jaxpr.eqns if e.primitive.name == "composite")
            return str(eqn.params["jaxpr"])

        self.assertEqual(marker_jaxpr(num_real=10), marker_jaxpr())

    def test_num_real_out_of_range_rejected(self) -> None:
        # Mirror the zkx recognizer's bound (1 <= num_real <= table width) at
        # emission, on the marked and unmarked paths alike. `prove` raises
        # before touching the transcript, so both are safely reused across
        # the bad values.
        f = rand_field(43, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=1)
        transcripts = (cheap_transcript(KB), self._poseidon_transcript())
        for bad in (0, 17):
            for transcript in transcripts:
                with self.assertRaisesRegex(ValueError, "num_real"):
                    prove(rnd, [f], transcript, num_real=bad)

    def test_marker_and_nested_permute_survive_lowering(self) -> None:
        # The whole sumcheck lowers under the hash-agnostic zorch.sumcheck marker;
        # the FS permute survives as a nested zorch.poseidon2 marker the vendor
        # reads to run the sponge in-kernel, and the per-round combine as a nested
        # zorch.sumcheck.combine marker the vendor inlines generically.
        a = rand_field(40, (1 << 4,), KB)
        b = rand_field(41, (1 << 4,), KB)
        rnd = prover.SumcheckRound(degree=2)
        t0 = self._poseidon_transcript()
        text = jax.jit(lambda x, y: prove(rnd, [x, y], t0)).lower(a, b).as_text()
        self.assertIn(SUMCHECK_MARKER, text)
        self.assertIn(SUMCHECK_COMBINE_MARKER, text)
        self.assertIn(f'"{POSEIDON2_MARKER}"', text)


if __name__ == "__main__":
    absltest.main()
