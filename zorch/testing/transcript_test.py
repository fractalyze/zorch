# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from dataclasses import replace
from unittest import mock

import jax
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest
from jax import Array, lax, tree_util

from zorch.hash.poseidon2.testing.koalabear16 import (
    koalabear16_perm,
    koalabear16_scaled_perm,
)
from zorch.testkit.jit_cache import assert_single_trace
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import (
    DUPLEX_FS_MARKER,
    DuplexState,
    DuplexTranscript,
    GrindError,
    _absorb_permute,
    _check_witness_body,
    _observe_and_sample_body,
    _observe_body,
    _sample_body,
    observe_and_sample_marked,
    sample_challenge,
)

F = zk_dtypes.koalabear_mont  # the koalabear16 permutation's field

# The duplex sponge's `lax.scan` absorb is correct on GPU but hits a ZKX CPU
# while-emitter bug that drops the scan's array-carry update (eager runs go
# non-deterministic). Skip the duplex tests on CPU until it lands. See
# fractalyze/zkx#500.
_CPU_BACKEND = jax.default_backend() == "cpu"
_skip_on_cpu_scan_bug = absltest.skipIf(
    _CPU_BACKEND,
    "ZKX CPU scan array-carry bug (GPU-correct); remove when fractalyze/zkx#500 lands",
)


@_skip_on_cpu_scan_bug
class DuplexTranscriptTest(absltest.TestCase):
    """The real duplex-sponge transcript: a device-side JAX pytree that threads
    functionally under @jit. observe absorbs into the sponge; sample squeezes
    field elements derived from everything observed so far (Fiat-Shamir)."""

    def _new(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def test_sample_returns_n_field_elements(self) -> None:
        _, out = self._new().sample(3)
        self.assertEqual(out.shape, (3,))
        self.assertEqual(out.dtype, F)

    def test_deterministic_for_equal_observations(self) -> None:
        v = rand_field(1, (5,), F)
        _, a = self._new().observe(v).sample(2)
        _, b = self._new().observe(v).sample(2)
        self.assertTrue(bool(jnp.all(a == b)))

    def test_challenge_binds_to_observation(self) -> None:
        # Fiat-Shamir: a changed transcript must yield different challenges.
        v = rand_field(2, (5,), F)
        _, ca = self._new().observe(v).sample(2)
        _, cb = self._new().observe(v.at[2].add(jnp.array(1, F))).sample(2)
        self.assertFalse(bool(jnp.all(ca == cb)))

    def test_sample_stream_is_consistent(self) -> None:
        # sample(2) equals two sample(1)s in sequence.
        t0 = self._new().observe(rand_field(3, (4,), F))
        _, both = t0.sample(2)
        t1, x0 = t0.sample(1)
        _, x1 = t1.sample(1)
        self.assertTrue(bool(both[0] == x0[0]))
        self.assertTrue(bool(both[1] == x1[0]))

    def test_threads_under_jit(self) -> None:
        # Acceptance: state threads functionally under @jit (so the transcript
        # can later live in a lax.scan carry, issue #58).
        v = rand_field(4, (5,), F)
        got = jax.jit(lambda t, x: t.observe(x).sample(2)[1])(self._new(), v)
        _, want = self._new().observe(v).sample(2)
        self.assertTrue(bool(jnp.all(got == want)))

    def test_observe_and_sample_matches_observe_then_sample(self) -> None:
        # The fused per-round primitive is a drop-in for observe-then-sample:
        # identical challenges and identical resulting transcript state.
        v = rand_field(7, (5,), F)
        t_ref, ref = self._new().observe(v).sample(2)
        t_fused, fused = self._new().observe_and_sample(v, 2)
        self.assertTrue(bool(jnp.all(ref == fused)))
        for a, b in zip(tree_util.tree_leaves(t_ref), tree_util.tree_leaves(t_fused)):
            self.assertTrue(bool(jnp.all(a == b)))

    def test_observe_and_sample_fuses_under_one_jit(self) -> None:
        # Acceptance: absorb+squeeze are one @jit computation (fused by
        # construction), matching the eager observe-then-sample reference.
        v = rand_field(8, (5,), F)
        got = jax.jit(lambda t, x: t.observe_and_sample(x, 2)[1])(self._new(), v)
        _, want = self._new().observe(v).sample(2)
        self.assertTrue(bool(jnp.all(got == want)))

    def _assert_marked_matches_plain(self, t0: DuplexTranscript) -> None:
        # Marked hop vs plain decomposition at a fresh entry and a mid-stream
        # one (non-zero duplex positions ride as operands, so one kernel serves
        # any phase): challenge and all five state leaves byte-identical.
        v = rand_field(9, (5,), F)
        for advance in (0, 1):  # fresh, then non-zero (in_pos, out_pos)
            t = t0
            for _ in range(advance):
                t, _ = _observe_and_sample_body(t, rand_field(2, (5,), F), 3)
            t_ref, ref = _observe_and_sample_body(t, v, 4)
            t_mk, mk = observe_and_sample_marked(t, v, 4)
            self.assertTrue(bool(jnp.all(ref == mk)))
            for a, b in zip(tree_util.tree_leaves(t_ref), tree_util.tree_leaves(t_mk)):
                self.assertTrue(bool(jnp.all(a == b)))

    def test_duplex_fs_marker_byte_matches_plain(self) -> None:
        # The `zorch.duplex_fs` fusion marker is a byte-identical drop-in for the
        # plain hop (an un-emitted marker inlines to the same computation). It
        # also appears by construction in the lowered HLO for a vendor to fuse.
        self._assert_marked_matches_plain(self._new())
        hlo = (
            jax.jit(lambda t, x: observe_and_sample_marked(t, x, 4))
            .lower(self._new(), rand_field(9, (5,), F))
            .as_text()
        )
        self.assertIn(DUPLEX_FS_MARKER, hlo)

    def test_duplex_fs_marker_byte_matches_plain_scaled_j(self) -> None:
        # Same drop-in contract under a NON-identity internal_j_scale. The
        # default instance's identity scale hides a whole bug class: a vendor
        # kernel that substitutes identity for the J term's scale — e.g. by
        # re-encoding the operand's Montgomery storage as a canonical value
        # (fractalyze/xla#206, sp1-zorch#208) — is byte-invisible above but
        # diverges here on every hop.
        self._assert_marked_matches_plain(
            DuplexTranscript.new(koalabear16_scaled_perm(), rate=8)
        )

    def test_duplex_fs_marker_state_survives_squeeze_consumer(self) -> None:
        # Regression: consuming the marked hop's squeezed challenge INSIDE the same
        # jit must leave the returned transcript's squeeze residue
        # (`output_buffer`/`out_pos`) byte-identical to the plain hop. The
        # `zorch.duplex_fs` composite is multi-output; an in-graph consumer of the
        # challenge forces explicit get-tuple-elements, and a dummy fusion body whose
        # equal-shaped result leaves shared one constant once let XLA collapse
        # out_buf->in_buf and out_pos->in_pos (only the challenge itself stayed
        # exact). The eager marker check above returns the challenge directly (no
        # in-graph consumer), so it never exercised this -- the jagged prover's
        # `_fs_and_reduce` does. Only the GPU custom-fusion emitter expands the
        # composite; elsewhere it inlines to the plain hop and this trivially holds.
        v = rand_field(11, (5,), F)

        def consume(t: DuplexTranscript, x: Array) -> tuple[DuplexTranscript, Array]:
            t2, s = observe_and_sample_marked(t, x, 4)
            return t2, s * s  # give the squeeze an in-graph consumer, then discard

        t_ref, _ = _observe_and_sample_body(self._new(), v, 4)
        t_mk, _ = jax.jit(consume)(self._new(), v)
        for a, b in zip(tree_util.tree_leaves(t_ref), tree_util.tree_leaves(t_mk)):
            self.assertTrue(bool(jnp.all(a == b)))

    def test_is_pytree(self) -> None:
        # 5 state buffers are the leaves; permutation + rate are static.
        leaves, treedef = tree_util.tree_flatten(self._new())
        self.assertEqual(len(leaves), 5)
        rebuilt = tree_util.tree_unflatten(treedef, leaves)
        _, a = self._new().sample(1)
        _, b = rebuilt.sample(1)
        self.assertTrue(bool(a[0] == b[0]))


class SampleChallengeTest(absltest.TestCase):
    """`sample_challenge` over the cheap test sponge -- the squeeze rule
    itself, independent of the duplex permutation."""

    def test_single_limb_is_plain_sample(self) -> None:
        t = cheap_transcript(F)
        t_ch, ch = sample_challenge(t, F, 1)
        t_raw, raw = t.sample(1)
        self.assertTrue(bool(ch == raw[0]))
        # Both advance the sponge identically.
        _, a = t_ch.sample(1)
        _, b = t_raw.sample(1)
        self.assertTrue(bool(a[0] == b[0]))

    def test_multi_limb_reinterprets_base_samples_as_coefficients(self) -> None:
        EF = zk_dtypes.koalabearx4_mont
        t = cheap_transcript(F)
        _, ch = sample_challenge(t, EF, 4)
        _, raw = t.sample(4)
        self.assertEqual(ch.dtype, EF)
        self.assertTrue(bool(jnp.all(ch[None].view(F) == raw)))

    def test_rejects_mismatched_packing(self) -> None:
        # The squeezes are consumed before the reinterpret, so a wrong limb
        # count must fail loud rather than truncate to the first element.
        EF = zk_dtypes.koalabearx4_mont
        with self.assertRaises(ValueError):
            sample_challenge(cheap_transcript(F), EF, 8)
        with self.assertRaises(ValueError):
            sample_challenge(cheap_transcript(F), F, 2)
        with self.assertRaises(ValueError):
            sample_challenge(cheap_transcript(F), F, 0)


class TranscriptJitCacheTest(absltest.TestCase):
    """Fresh transcripts must be cache-key-equal: the permutation is pytree aux
    (meta_fields), so two `DuplexTranscript.new(...)` over independently built,
    value-equal permutations must yield IDENTICAL treedefs — otherwise every jit
    zone taking a transcript re-traces per call (issue #163: ~2 min/call on the
    jagged verify replay whose kernels run in 20 ms)."""

    def test_fresh_cheap_transcripts_share_treedef(self) -> None:
        # The cheap permutation has no jit-cache counterpart below (its scan
        # hits the zkx#500 CPU bug), so treedef equality is its only guard.
        self.assertEqual(
            tree_util.tree_structure(cheap_transcript(F)),
            tree_util.tree_structure(cheap_transcript(F)),
        )

    def test_jit_zone_does_not_retrace_per_fresh_transcript(self) -> None:
        @jax.jit
        def zone(t: DuplexTranscript) -> jnp.ndarray:
            return t.state.sponge_state

        zone(DuplexTranscript.new(koalabear16_perm(), rate=8))
        zone(DuplexTranscript.new(koalabear16_perm(), rate=8))
        # _cache_size() is a private JAX API; may change on jax upgrade.
        self.assertEqual(zone._cache_size(), 1)

    def test_sample_reuses_one_cached_zone(self) -> None:
        # Eager sample must hit the module-level zone: repeated calls and
        # fresh same-config instances add no trace (#226 — the eager Python
        # loop used to re-trace the permutation graph on every call).
        calls = [
            functools.partial(cheap_transcript(F).sample, 3),
            functools.partial(cheap_transcript(F).sample, 3),
        ]
        assert_single_trace(self, _sample_body, calls)

    @_skip_on_cpu_scan_bug
    def test_observe_family_reuses_cached_zones(self) -> None:
        # The scan-bearing ops (duplex-only: the cheap permutation's scan hits
        # zkx#500 on CPU), same contract as sample above.
        v = rand_field(11, (5,), F)
        w = jnp.zeros((), F)
        new = functools.partial(DuplexTranscript.new, koalabear16_perm(), rate=8)
        assert_single_trace(
            self, _observe_body, [functools.partial(new().observe, v) for _ in (0, 1)]
        )
        assert_single_trace(
            self,
            _observe_and_sample_body,
            [functools.partial(new().observe_and_sample, v, 2) for _ in (0, 1)],
        )
        assert_single_trace(
            self,
            _check_witness_body,
            [functools.partial(new().check_witness, 4, w) for _ in (0, 1)],
        )


@_skip_on_cpu_scan_bug
class GrindTest(absltest.TestCase):
    """Proof-of-work grind over the duplex sponge: search the full witness space
    for a witness whose squeezed challenge has `pow_bits` zero low bits, plus the
    verifier `check_witness` the prover and verifier must agree on."""

    def _seeded(self) -> DuplexTranscript:
        # A non-degenerate sponge so the first witness isn't the trivial zero
        # (an all-zero sponge maps the zero witness to a zero challenge, which
        # satisfies any pow_bits and would mask the search).
        return cheap_transcript(F).observe(rand_field(11, (5,), F))

    def test_check_witness_accepts_grind_result(self) -> None:
        for pow_bits in (0, 4, 12):
            _, witness = self._seeded().grind(pow_bits)
            _, ok = self._seeded().check_witness(pow_bits, witness)
            self.assertTrue(bool(ok), f"pow_bits={pow_bits}")

    def test_high_pow_bits_clears(self) -> None:
        # Acceptance: a pow_bits far past a single window's reach still resolves
        # to a witness the verifier accepts.
        _, witness = self._seeded().grind(22)
        _, ok = self._seeded().check_witness(22, witness)
        self.assertTrue(bool(ok))

    def test_search_spans_windows(self) -> None:
        # The fix: a single-window search covers only its first `chunk`
        # candidates, so once the satisfying witness lands past that window it
        # must keep advancing to reach it. A tiny window forces many advances to
        # the same witness a single large window finds directly -- a capped
        # search would instead miss it and return a different (invalid) witness.
        small, large = 1 << 8, 1 << 16
        _, near = self._seeded().grind(14, chunk=small)
        _, far = self._seeded().grind(14, chunk=large)
        self.assertEqual(int(near.astype(jnp.uint32)), int(far.astype(jnp.uint32)))
        self.assertGreater(int(near.astype(jnp.uint32)), small)
        _, ok = self._seeded().check_witness(14, near)
        self.assertTrue(bool(ok))

    def test_prover_and_verifier_states_agree(self) -> None:
        prover, witness = self._seeded().grind(8)
        verifier, ok = self._seeded().check_witness(8, witness)
        self.assertTrue(bool(ok))
        for a, b in zip(
            tree_util.tree_leaves(prover), tree_util.tree_leaves(verifier), strict=True
        ):
            self.assertTrue(bool(jnp.all(a == b)))

    def test_exhausted_search_raises_loudly(self) -> None:
        # Sweeping the whole field is too slow to trigger naturally, so inject an
        # exhausted search (found=False) returning a witness that fails the
        # check, and assert grind surfaces it rather than returning an unverified
        # witness.
        bad = self._a_failing_witness(8)
        with mock.patch.object(DuplexTranscript, "_grind_search", return_value=bad):
            with self.assertRaises(GrindError):
                self._seeded().grind(8)

    def test_rejects_out_of_range_pow_bits(self) -> None:
        with self.assertRaises(ValueError):
            self._seeded().grind(32)
        with self.assertRaises(ValueError):
            self._seeded().check_witness(-1, jnp.zeros((), F))

    def test_check_witness_rejects_off_domain_witness(self) -> None:
        # The witness must be a scalar base-field element -- the domain grind
        # enumerates -- so the verifier accepts exactly what the prover searched.
        seeded = self._seeded()
        with self.assertRaises(ValueError):
            seeded.check_witness(8, jnp.zeros((2,), F))  # non-scalar
        with self.assertRaises(ValueError):
            seeded.check_witness(8, jnp.zeros((), zk_dtypes.koalabearx4_mont))  # not F

    def test_grind_rejects_non_positive_chunk(self) -> None:
        with self.assertRaises(ValueError):
            self._seeded().grind(8, chunk=0)

    def test_field_wider_than_uint32_raises_loudly(self) -> None:
        # The uint32 counter/bit-check (jax x64 off) can't represent a field
        # whose order exceeds 32 bits; both entry points must say so plainly
        # rather than fail with an opaque narrowing-convert error.
        wide = zk_dtypes.goldilocks_mont
        with self.assertRaises(GrindError):
            cheap_transcript(wide).grind(8)
        with self.assertRaises(GrindError):
            cheap_transcript(wide).check_witness(8, jnp.zeros((), wide))

    def _a_failing_witness(self, pow_bits: int) -> jnp.ndarray:
        base = self._seeded()
        for cand in range(256):
            _, ok = base.check_witness(pow_bits, jnp.array(cand, F))
            if not bool(ok):
                return jnp.array(cand, F)
        raise AssertionError("expected a failing witness within range")


def _cond_sample_one(t: DuplexTranscript) -> tuple[DuplexTranscript, jnp.ndarray]:
    """The cond-based `_sample_one`: a traced-predicate `lax.cond` over
    `_duplexing`. Kept here as the byte-identity reference the production
    `select` rewrite must reproduce exactly."""
    need_perm = (t.state.in_pos > 0) | (t.state.out_pos == 0)
    t2 = lax.cond(need_perm, lambda c: c._duplexing(), lambda c: c, t)
    out_pos = t2.state.out_pos - 1
    item = t2.state.output_buffer[out_pos]
    return t2._with_state(replace(t2.state, out_pos=out_pos)), item


def _cond_observe_body(t: DuplexTranscript, values: jnp.ndarray) -> DuplexTranscript:
    """The cond-based `_observe_body` scan step: a traced-predicate `lax.cond`
    on the full-block flush. The byte-identity reference for the production
    `select` rewrite."""
    base_dtype = t.state.sponge_state.dtype
    flat = lax.bitcast_convert_type(values, base_dtype).reshape(-1)
    if flat.shape[0] == 0:
        return t
    rate = t.rate
    permutation = t.permutation

    def step(
        carry: tuple[Array, Array, Array], x: Array
    ) -> tuple[tuple[Array, Array, Array], None]:
        in_buf, in_pos, sponge = carry
        in_buf = in_buf.at[in_pos].set(x)
        new_in_pos = in_pos + 1
        full = new_in_pos == rate

        def perm(args: tuple[Array, Array]) -> tuple[Array, Array]:
            sp, ib = args
            new_sponge = _absorb_permute(permutation, sp, ib, new_in_pos, rate)
            return new_sponge, jnp.zeros_like(ib)

        sponge, in_buf = lax.cond(full, perm, lambda a: a, (sponge, in_buf))
        in_pos_out = jnp.where(full, jnp.int32(0), new_in_pos)
        return (in_buf, in_pos_out, sponge), None

    init = (t.state.input_buffer, t.state.in_pos, t.state.sponge_state)
    (in_buf, in_pos, sponge), _ = lax.scan(step, init, flat)
    last_was_perm = in_pos == 0
    out_pos = jnp.where(last_was_perm, jnp.int32(rate), jnp.int32(0))
    output_buffer = jnp.where(
        last_was_perm, sponge[:rate], jnp.zeros(rate, dtype=base_dtype)
    )
    return t._with_state(DuplexState(in_buf, output_buffer, sponge, in_pos, out_pos))


class CondToSelectByteIdentityTest(absltest.TestCase):
    """`_sample_one` replaced its traced-predicate
    `lax.cond` over `_duplexing` with a `select` of the unconditionally-permuted
    state, to drop the per-sample device->host Fiat-Shamir sync. The select must
    return the exact value the cond did -- it picks the same branch and field ops
    are exact. `sample` has no `lax.scan`, so this runs on CPU (unlike the
    observe path); `_observe_body`'s twin rewrite is pinned on GPU below."""

    def test_select_sample_matches_cond_reference(self) -> None:
        # A fresh sponge has out_pos == 0 (need_perm True -> permute); the next
        # squeeze reads the primed output buffer (need_perm False). Draining the
        # buffer over several samples cycles back to a permute, so the run covers
        # both predicate values of the out_pos == 0 disjunct.
        t_sel = cheap_transcript(F)
        t_ref = cheap_transcript(F)
        for i in range(12):
            t_sel, x_sel = t_sel._sample_one()
            t_ref, x_ref = _cond_sample_one(t_ref)
            self.assertTrue(bool(x_sel == x_ref), f"sample {i} value diverged")
            for a, b in zip(
                tree_util.tree_leaves(t_sel),
                tree_util.tree_leaves(t_ref),
                strict=True,
            ):
                self.assertTrue(bool(jnp.all(a == b)), f"sample {i} state diverged")

    @_skip_on_cpu_scan_bug
    def test_select_observe_matches_cond_reference(self) -> None:
        # 19 elements over rate 8: two full-block flushes (`full` True) plus a
        # 3-element tail (`full` False), so the run covers both branches of the
        # scan-step select. Resulting state AND a follow-on sample must match the
        # cond reference byte-for-byte.
        v = rand_field(13, (19,), F)
        new = functools.partial(DuplexTranscript.new, koalabear16_perm(), rate=8)
        t_sel = _observe_body(new(), v)
        t_ref = _cond_observe_body(new(), v)
        for a, b in zip(
            tree_util.tree_leaves(t_sel),
            tree_util.tree_leaves(t_ref),
            strict=True,
        ):
            self.assertTrue(bool(jnp.all(a == b)), "observe state diverged")
        _, x_sel = t_sel.sample(4)
        _, x_ref = t_ref.sample(4)
        self.assertTrue(bool(jnp.all(x_sel == x_ref)), "post-observe sample diverged")


def _ref_observe(t: DuplexTranscript, values: jnp.ndarray) -> DuplexTranscript:
    """Verbatim copy of the pre-rate-block `_observe_body`: a `lax.scan` that
    absorbs ONE base element per step and runs a full `_absorb_permute` on every
    element, keeping the rate-boundary one via `jnp.where`. The byte-identity
    reference the rate-block rewrite must reproduce exactly."""
    base_dtype = t.state.sponge_state.dtype
    flat = lax.bitcast_convert_type(values, base_dtype).reshape(-1)
    if flat.shape[0] == 0:
        return t
    rate = t.rate
    permutation = t.permutation

    def step(
        carry: tuple[Array, Array, Array], x: Array
    ) -> tuple[tuple[Array, Array, Array], None]:
        in_buf, in_pos, sponge = carry
        in_buf = in_buf.at[in_pos].set(x)
        new_in_pos = in_pos + 1
        full = new_in_pos == rate
        permuted_sponge = _absorb_permute(permutation, sponge, in_buf, new_in_pos, rate)
        sponge = jnp.where(full, permuted_sponge, sponge)
        in_buf = jnp.where(full, jnp.zeros_like(in_buf), in_buf)
        in_pos_out = jnp.where(full, jnp.int32(0), new_in_pos)
        return (in_buf, in_pos_out, sponge), None

    init = (t.state.input_buffer, t.state.in_pos, t.state.sponge_state)
    (in_buf, in_pos, sponge), _ = lax.scan(step, init, flat)
    last_was_perm = in_pos == 0
    out_pos = jnp.where(last_was_perm, jnp.int32(rate), jnp.int32(0))
    output_buffer = jnp.where(
        last_was_perm, sponge[:rate], jnp.zeros(rate, dtype=base_dtype)
    )
    return t._with_state(DuplexState(in_buf, output_buffer, sponge, in_pos, out_pos))


def _ref_sample_one(t: DuplexTranscript) -> tuple[DuplexTranscript, jnp.ndarray]:
    """Verbatim copy of the pre-rate-block `_sample_one`: an unconditional
    `_duplexing` (one permute) selected away when not needed."""
    need_perm = (t.state.in_pos > 0) | (t.state.out_pos == 0)
    permuted = t._duplexing()
    t = t._with_state(
        tree_util.tree_map(
            lambda p, c: jnp.where(need_perm, p, c), permuted.state, t.state
        )
    )
    out_pos = t.state.out_pos - 1
    item = t.state.output_buffer[out_pos]
    return t._with_state(replace(t.state, out_pos=out_pos)), item


def _ref_sample(t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, jnp.ndarray]:
    """Verbatim copy of the pre-rate-block `_sample_body`: `n` per-limb
    `_sample_one` calls, each running a permute unconditionally."""
    outs = []
    for _ in range(n):
        t, x = _ref_sample_one(t)
        outs.append(x.reshape(()))
    return t, jnp.stack(outs)


@_skip_on_cpu_scan_bug
class RateBlockByteIdentityTest(absltest.TestCase):
    """Rate-block batching: `_observe_body` now permutes once per
    rate-block (not once per element) and `_sample_body` once per drained
    output-block (not once per limb). Both must be byte-for-byte identical to the
    captured pre-change references -- the transcript drives the prover's
    Fiat-Shamir, so any drift breaks every proof. Pinned on GPU (the duplex
    `lax.scan` hits the zkx#500 CPU bug)."""

    def _new(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def _seeded(self, seed: int, observes: tuple, samples: tuple) -> DuplexTranscript:
        # Build a non-trivial starting state (non-zero in_pos/out_pos) by replaying
        # prior observes/samples through the PRODUCTION ops -- already proven equal
        # to the reference, so a divergence here is a fresh-state divergence.
        t = self._new()
        for i, (mlen, n) in enumerate(zip(observes, samples, strict=True)):
            if mlen:
                t = t.observe(rand_field(seed * 100 + i, (mlen,), F))
            if n:
                t, _ = t.sample(n)
        return t

    def _assert_state_eq(
        self, a: DuplexTranscript, b: DuplexTranscript, msg: str
    ) -> None:
        for x, y in zip(
            tree_util.tree_leaves(a), tree_util.tree_leaves(b), strict=True
        ):
            self.assertTrue(bool(jnp.all(x == y)), msg)

    def test_observe_matches_reference_over_many_lengths(self) -> None:
        # Rate-boundary edges explicit: 7 (under), 8 (exact one block), 9 (one
        # block + tail), 16/17 (two blocks +/- tail), plus 0/1/40.
        for mlen in (0, 1, 7, 8, 9, 16, 17, 40):
            v = rand_field(mlen + 1, (mlen,), F)
            new = _observe_body(self._new(), v)
            ref = _ref_observe(self._new(), v)
            self._assert_state_eq(new, ref, f"observe(len={mlen}) state diverged")

    def test_observe_from_nonzero_in_pos(self) -> None:
        # Start with a partial input buffer (in_pos != 0) so the combined-stream
        # gap-removal and runtime block offset are exercised, across edge lengths.
        for in_pos in (1, 3, 7):
            base_new = _observe_body(self._new(), rand_field(in_pos, (in_pos,), F))
            base_ref = _ref_observe(self._new(), rand_field(in_pos, (in_pos,), F))
            self._assert_state_eq(base_new, base_ref, f"seed in_pos={in_pos} diverged")
            self.assertEqual(int(base_new.state.in_pos), in_pos)
            for mlen in (0, 1, 7, 8, 9, 16, 17):
                v = rand_field(in_pos * 10 + mlen + 1, (mlen,), F)
                new = _observe_body(base_new, v)
                ref = _ref_observe(base_ref, v)
                self._assert_state_eq(
                    new, ref, f"observe(in_pos={in_pos}, len={mlen}) diverged"
                )

    def test_sample_matches_reference_over_many_counts(self) -> None:
        # From a fresh sponge (in_pos 0, out_pos 0 -> first sample permutes).
        for n in (1, 2, 4, 7, 8, 9, 16, 17):
            t_new, x_new = _sample_body(self._new(), n)
            t_ref, x_ref = _ref_sample(self._new(), n)
            self.assertTrue(
                bool(jnp.all(x_new == x_ref)), f"sample({n}) value diverged"
            )
            self._assert_state_eq(t_new, t_ref, f"sample({n}) state diverged")

    def test_sample_from_partial_output_buffer(self) -> None:
        # Drain some limbs first so the next sample starts mid-buffer (out_pos in
        # (0, rate)), then sample across the rate boundary.
        for drained in (1, 3, 7):
            base = self._seeded(drained, (5,), (drained,))
            for n in (1, 2, 4, 8, 9, 16):
                t_new, x_new = _sample_body(base, n)
                t_ref, x_ref = _ref_sample(base, n)
                self.assertTrue(
                    bool(jnp.all(x_new == x_ref)),
                    f"sample(drained={drained}, n={n}) value diverged",
                )
                self._assert_state_eq(
                    t_new, t_ref, f"sample(drained={drained}, n={n}) state diverged"
                )

    def test_sample_from_pending_input(self) -> None:
        # in_pos > 0 AND out_pos > 0: the first sample must flush (permute) even
        # though outputs are available -- the forced-flush first permute. This
        # state is not reachable through the public API (observe zeroes out_pos
        # when it leaves a tail), so build it directly to pin the `_duplexing`
        # flush path both implementations share.
        primed, _ = self._new().sample(2)  # out_pos in (0, rate), input buffer empty
        for in_pos in (1, 3, 7):
            pend = rand_field(in_pos, (8,), F)
            # Only [0:in_pos] is valid in overwrite mode; zero the rest.
            buf = jnp.where(jnp.arange(8) < in_pos, pend, jnp.zeros(8, F))
            state = replace(primed.state, input_buffer=buf, in_pos=jnp.int32(in_pos))
            base = primed._with_state(state)
            self.assertEqual(int(base.state.in_pos), in_pos)
            self.assertGreater(int(base.state.out_pos), 0)
            for n in (1, 2, 4, 9):
                t_new, x_new = _sample_body(base, n)
                t_ref, x_ref = _ref_sample(base, n)
                self.assertTrue(
                    bool(jnp.all(x_new == x_ref)),
                    f"pending-input sample(in_pos={in_pos}, n={n}) diverged",
                )
                self._assert_state_eq(
                    t_new, t_ref, f"pending-input sample(in_pos={in_pos}, n={n}) state"
                )

    def test_interleaved_observe_sample_sequences(self) -> None:
        # Long interleaved scripts exercise carried in_pos/out_pos across both ops.
        scripts = (
            ((1, 0), (0, 2), (7, 0), (0, 3), (8, 0), (0, 5), (17, 0), (0, 9)),
            ((9, 0), (0, 4), (3, 0), (0, 1), (16, 0), (0, 8), (0, 8), (1, 0)),
            ((40, 0), (0, 16), (0, 1), (7, 0), (0, 7), (5, 0), (0, 9)),
        )
        for si, script in enumerate(scripts):
            t_new = self._new()
            t_ref = self._new()
            for step_i, (mlen, n) in enumerate(script):
                if mlen:
                    v = rand_field(si * 1000 + step_i, (mlen,), F)
                    t_new = _observe_body(t_new, v)
                    t_ref = _ref_observe(t_ref, v)
                if n:
                    t_new, x_new = _sample_body(t_new, n)
                    t_ref, x_ref = _ref_sample(t_ref, n)
                    self.assertTrue(
                        bool(jnp.all(x_new == x_ref)),
                        f"script {si} step {step_i} value diverged",
                    )
                self._assert_state_eq(
                    t_new, t_ref, f"script {si} step {step_i} state diverged"
                )


def _pure_observe(t: DuplexTranscript, values: jnp.ndarray) -> DuplexTranscript:
    """Scan-free, CPU-safe per-element observe reference — the byte-identity spec.

    A Python loop with NO `lax.scan` and NO traced-index scatter: `in_buf[in_pos]`
    is set with a `jnp.where` select, and `_absorb_permute`'s `sponge.at[:rate]`
    write is a static-range scatter the sample path already uses CPU-safely. So
    this is correct on the ZKX CPU backend, unlike the scan-based `_ref_observe`,
    and can byte-check the production rate-block observe ON CPU."""
    base = t.state.sponge_state.dtype
    flat = lax.bitcast_convert_type(values, base).reshape(-1)
    if flat.shape[0] == 0:
        return t
    rate = t.rate
    sponge, in_buf, in_pos = (
        t.state.sponge_state,
        t.state.input_buffer,
        t.state.in_pos,
    )
    slot = jnp.arange(rate, dtype=jnp.int32)
    for i in range(int(flat.shape[0])):
        in_buf = jnp.where(slot == in_pos, flat[i], in_buf)  # set [in_pos], no scatter
        new_in_pos = in_pos + 1
        full = new_in_pos == rate
        permuted = _absorb_permute(t.permutation, sponge, in_buf, new_in_pos, rate)
        sponge = jnp.where(full, permuted, sponge)
        in_buf = jnp.where(full, jnp.zeros_like(in_buf), in_buf)
        in_pos = jnp.where(full, jnp.int32(0), new_in_pos)
    last_was_perm = in_pos == 0
    out_pos = jnp.where(last_was_perm, jnp.int32(rate), jnp.int32(0))
    output_buffer = jnp.where(last_was_perm, sponge[:rate], jnp.zeros(rate, base))
    return t._with_state(DuplexState(in_buf, output_buffer, sponge, in_pos, out_pos))


class CpuByteIdentityTest(absltest.TestCase):
    """Rate-block byte-identity on CPU (NOT `@_skip_on_cpu_scan_bug`).

    The byte-identity suite above is GPU-pinned (its `lax.scan`/`lax.cond`
    references hit the ZKX CPU scan bug, fractalyze/zkx#500), so the rate-block
    ops were never byte-checked on the CPU backend the prover runs on. These
    compare production `observe`/`sample` against scan-free, CPU-safe references
    (`_pure_observe`, the per-limb `_sample_one` loop) so a future rate-block
    LOGIC change is byte-checked on CPU as well as GPU.

    SCOPE — do NOT over-trust these. They compile `observe`/`sample` in ISOLATION,
    where the ops are correct even on the build that broke #292: zkx#500 is a
    CONTEXT-DEPENDENT CPU codegen miscompile that only fires inside the full
    prove's compilation. An isolated op — and even a transcript threaded through a
    bare `lax.scan` — compiles correctly on CPU; the divergence appears only in
    the rolled prove. So these guard rate-block LOGIC; the zkx#500 codegen
    miscompile is gated ONLY by the consumer's full CPU prove (sp1-zorch
    `logup_gkr:{prover,verifier}_test`, `shard_prover:verify_shard_test`), which
    is where it surfaced. A zorch Fiat-Shamir change must be validated there.
    """

    def _new(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8)

    def _seeded(self, seed: int, observes: tuple, samples: tuple) -> DuplexTranscript:
        # Non-trivial start state via the CPU-safe references only, so the seed
        # itself can never be the divergence under test.
        t = self._new()
        for i, (mlen, n) in enumerate(zip(observes, samples, strict=True)):
            if mlen:
                t = _pure_observe(t, rand_field(seed * 100 + i, (mlen,), F))
            for _ in range(n):
                t, _ = t._sample_one()
        return t

    def _assert_state_eq(
        self, a: DuplexTranscript, b: DuplexTranscript, msg: str
    ) -> None:
        for x, y in zip(
            tree_util.tree_leaves(a), tree_util.tree_leaves(b), strict=True
        ):
            self.assertTrue(bool(jnp.all(x == y)), msg)

    def test_observe_matches_pure_reference(self) -> None:
        for mlen in (1, 7, 8, 9, 16, 17, 37, 40):
            for seed, obs, smp in ((0, (), ()), (3, (5, 9), (2, 1))):
                t = self._seeded(seed, obs, smp)
                v = rand_field(seed * 1000 + mlen, (mlen,), F)
                self._assert_state_eq(
                    t.observe(v),
                    _pure_observe(t, v),
                    f"observe(len={mlen}, seed={seed}) diverged from per-element ref",
                )

    def test_sample_matches_per_limb_reference(self) -> None:
        for n in (1, 2, 7, 8, 9, 16, 21):  # spans the rate-8 block boundary
            for seed, obs, smp in ((0, (), ()), (5, (9,), (3,))):
                t = self._seeded(seed, obs, smp)
                got_t, got = t.sample(n)
                ref_t, outs = t, []
                for _ in range(n):
                    ref_t, x = ref_t._sample_one()
                    outs.append(x.reshape(()))
                self.assertTrue(
                    bool(jnp.all(got == jnp.stack(outs))),
                    f"sample(n={n}, seed={seed}) value diverged from per-limb ref",
                )
                self._assert_state_eq(
                    got_t, ref_t, f"sample(n={n}, seed={seed}) state diverged"
                )


if __name__ == "__main__":
    absltest.main()
