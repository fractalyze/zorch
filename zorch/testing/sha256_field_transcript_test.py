# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The field-element `Transcript` surface (`Sha256FieldTranscript`) over the
streaming SHA-256 core.

The byte transcript is the established oracle (`byte_transcript_test` pins the
framing vs `hashlib`; flock-zorch's `challenger_test` pins it to flock-core's
`FsChallenger`). This slice proves the FIELD surface reproduces the byte
transcript's slice framing exactly — the field surface is the byte surface, made
scan-threadable — and threads zorch's sumcheck round driver under `@jit`.
"""
from __future__ import annotations

from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from zorch.byte_transcript import KIND_SCALAR, OP_SQUEEZE, ByteHashTranscript
from zorch.hash.sha256 import HostSha256, Sha256
from zorch.pcs.fold import (
    _sample_distinct_positions_plain,
    sample_distinct_positions,
)
from zorch.sha256_field_transcript import (
    SAMPLE_DISTINCT_MARKER,
    SHA256_SQUEEZE_MARKER,
    Sha256FieldTranscript,
    _const_u8,
    _sha256_squeeze_zone,
    _squeeze_hop,
)


class Sha256FieldTranscriptTest(absltest.TestCase):
    def test_slice_framing_matches_byte_transcript(self) -> None:
        # observe(Array)/sample(n) use the same slice framing as the byte
        # transcript's observe_slice/sample_slice, so the squeezed challenge bytes
        # match — the field surface is the byte surface, made scan-threadable.
        vals = np.array([1, 2, 3, 4, 0xDEADBEEF, 5], dtype=np.uint32)
        vbytes = vals.astype("<u4").tobytes()

        b = ByteHashTranscript.new(b"dom", Sha256()).observe_slice(vbytes, vals.size)
        b, b_sq = b.sample_slice(3, 4)  # 3 elements * 4 bytes

        f = Sha256FieldTranscript.new(b"dom", np.uint32)
        f = f.observe(fnp.asarray(vals))
        f, f_el = f.sample(3)
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

        # A second squeeze pins the re-absorb of the first (both slice-framed).
        b, b2 = b.sample_slice(4, 4)
        f, f2 = f.sample(4)
        self.assertEqual(np.asarray(f2).astype("<u4").tobytes(), b2)

    def test_scalar_framing_matches_byte_transcript(self) -> None:
        # observe_scalar/sample_scalar use the byte transcript's scalar framing
        # (KIND_SCALAR, no length prefix), so the squeezed bytes match — and must
        # DIFFER from the slice framing of the same single element.
        v = np.uint32(0xDEADBEEF)

        b = ByteHashTranscript.new(b"dom", Sha256()).observe_scalar(v.tobytes())
        b, b_sq = b.sample_scalar(4)  # itemsize bytes

        f = Sha256FieldTranscript.new(b"dom", np.uint32)
        f, f_el = f.observe_scalar(fnp.asarray(v)).sample_scalar()
        self.assertEqual(f_el.shape, ())  # scalar squeeze is 0-D
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

        # Same element, but sampled/observed under SLICE framing — must diverge.
        g = Sha256FieldTranscript.new(b"dom", np.uint32)
        g, g_sl = g.observe(fnp.asarray(v).reshape(1)).sample(1)
        self.assertNotEqual(np.asarray(f_el).tobytes(), np.asarray(g_sl).tobytes())

    def test_vector_observe_scalar_matches_scalar_chain(self) -> None:
        # observe_scalar of an [n] array frames each element as its own scalar
        # op, byte-identical to chaining n 0-d observes — same state, so the
        # next squeeze matches.
        vals = fnp.asarray(np.array([7, 0xDEADBEEF, 0, 42], dtype=np.uint32))

        chained = Sha256FieldTranscript.new(b"dom", np.uint32)
        for v in vals:
            chained = chained.observe_scalar(v)
        chained, c_el = chained.sample_scalar()

        batched = Sha256FieldTranscript.new(b"dom", np.uint32)
        batched, b_el = batched.observe_scalar(vals).sample_scalar()
        self.assertEqual(np.asarray(c_el).tobytes(), np.asarray(b_el).tobytes())

    def test_label_and_bytes_framing_match_byte_transcript(self) -> None:
        # observe_label / observe_bytes reproduce the byte transcript's OP_LABEL /
        # OP_BYTES framing, so a challenge squeezed after them matches.
        label = b"flock-zerocheck-v0"
        root = np.arange(32, dtype=np.uint8)  # a 32-byte on-device "root"

        b = ByteHashTranscript.new(b"dom", Sha256())
        b = b.observe_label(label).observe_bytes(root.tobytes())
        b, b_sq = b.sample_slice(2, 4)

        f = Sha256FieldTranscript.new(b"dom", np.uint32)
        f = f.observe_label(label).observe_bytes(fnp.asarray(root))
        f, f_el = f.sample(2)
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

    def test_grind_check_witness_match_byte_transcript(self) -> None:
        # The device grind reproduces ByteHashTranscript's u64-nonce PoW: same
        # (lowest) nonce, and the transcripts stay in lockstep (same challenge
        # afterwards). check_witness accepts the honest witness, rejects a
        # tampered one, and advances regardless (the DuplexTranscript contract).
        root_u8 = fnp.asarray(np.frombuffer(b"root", np.uint8))
        for bits in (0, 8):
            b = ByteHashTranscript.new(b"pow", HostSha256()).observe_bytes(b"root")
            b, b_nonce = b.grind_pow(bits)
            _, b_ch = b.sample_scalar(4)

            f = Sha256FieldTranscript.new(b"pow", np.uint32).observe_bytes(root_u8)
            f, witness = f.grind(bits)
            self.assertEqual(int(witness), b_nonce)
            _, f_ch = f.sample_scalar()
            self.assertEqual(np.asarray(f_ch).astype("<u4").tobytes(), b_ch)

            # Verifier mirror accepts the honest witness and reaches the same state.
            vf = Sha256FieldTranscript.new(b"pow", np.uint32).observe_bytes(root_u8)
            vf, ok = vf.check_witness(witness, pow_bits=bits)
            self.assertTrue(bool(ok))
            _, vf_ch = vf.sample_scalar()
            self.assertEqual(np.asarray(vf_ch).astype("<u4").tobytes(), b_ch)
        bad = Sha256FieldTranscript.new(b"pow", np.uint32).observe_bytes(root_u8)
        _, bad_ok = bad.check_witness(int(witness) + 1, pow_bits=8)
        self.assertFalse(bool(bad_ok))

    def test_grind_bits_out_of_range_rejected(self) -> None:
        # Mirrors the byte transcript: > 256 (or negative) leading-zero bits on a
        # 32-byte digest is impossible and rejected up front.
        t = Sha256FieldTranscript.new(b"pow", np.uint32)
        for bits in (-1, 257):
            with self.assertRaises(ValueError):
                t.grind(bits)
            with self.assertRaises(ValueError):
                t.check_witness(0, pow_bits=bits)

    def test_ghash_dtype_matches_byte_transcript_via_uint32_lanes(self) -> None:
        # ghash <-> bytes routes through uint32 lanes to stay
        # correct on the CPU PJRT backend. Observe a ghash element and sample
        # ghash challenges; the wire bytes match the byte transcript over the same
        # 16-byte serialization, and the samples come back as device ghash.
        import zk_dtypes  # noqa: F401  (registers fnp.binary_field_ghash)

        gh = fnp.binary_field_ghash
        lanes = np.array([1, 2, 3, 0xDEADBEEF], dtype=np.uint32)  # one 16-byte elem
        v_host = lanes.view(np.dtype(gh))  # shape (1,), known LE bytes
        vbytes = lanes.tobytes()  # 16 LE bytes

        b = ByteHashTranscript.new(b"gh", Sha256()).observe_slice(vbytes, 1)
        b, b_sq = b.sample_slice(2, 16)  # two ghash-width challenges

        f = Sha256FieldTranscript.new(b"gh", gh)
        f, f_el = f.observe(fnp.asarray(v_host)).sample(2)
        self.assertEqual(np.asarray(f_el).dtype, np.dtype(gh))  # device ghash
        self.assertEqual(f_el.shape, (2,))
        self.assertEqual(np.asarray(f_el).tobytes(), b_sq)

    def test_threads_under_jit(self) -> None:
        vals = np.arange(6, dtype=np.uint32)

        def run(x: fnp.ndarray) -> fnp.ndarray:
            f = Sha256FieldTranscript.new(b"dom", np.uint32)
            _, r = f.observe_and_sample(x, 1)
            return r

        eager = np.asarray(run(fnp.asarray(vals)))
        jitted = np.asarray(frx.jit(run)(fnp.asarray(vals)))
        self.assertEqual(eager.tobytes(), jitted.tobytes())

    def test_ghash_threads_under_jit(self) -> None:
        # The 16-byte-element serde must be byte-identical under `@jit` — the
        # bitcast-chain simplification path has regressed before.
        # Eager is the pinned reference
        # (test_ghash_dtype_matches_byte_transcript_via_uint32_lanes).
        import zk_dtypes  # noqa: F401  (registers fnp.binary_field_ghash)

        gh = fnp.binary_field_ghash
        v = np.array([1, 2, 3, 0xDEADBEEF], dtype=np.uint32).view(np.dtype(gh))

        def run(x: fnp.ndarray) -> fnp.ndarray:
            f = Sha256FieldTranscript.new(b"gh", gh)
            f = f.observe_scalar(x[0]).observe(x)
            f, one = f.sample_scalar()
            _, vec = f.sample(2)
            return fnp.concatenate([one.reshape(1), vec])

        eager = np.asarray(run(fnp.asarray(v)))
        jitted = np.asarray(frx.jit(run)(fnp.asarray(v)))
        self.assertEqual(eager.tobytes(), jitted.tobytes())

    def test_threads_through_sumcheck_prove(self) -> None:
        # The acceptance-critical path: the transcript threads the sumcheck round
        # driver (fold_rounds over StandardRound) under jit, no host callback. A
        # uint32 ring stands in for a scalar challenge field (flock's F128 rides
        # the GHASH dtype seam); the point here is the transcript
        # threading, not the sumcheck math.
        from zorch.challenge import ChallengePolicy
        from zorch.prove import fold_rounds
        from zorch.sumcheck.prover import (
            ProductSummand,
            StandardRound,
            initial_claim,
        )

        a = fnp.arange(8, dtype=fnp.uint32) + 1
        b = fnp.arange(8, dtype=fnp.uint32) + 2
        rnd = StandardRound(
            ProductSummand(degree=2), challenges=ChallengePolicy(fnp.uint32)
        )
        tr = Sha256FieldTranscript.new(b"sc", np.uint32)

        def run(x: fnp.ndarray, y: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
            # 3 rounds folds the 2^3 stacked factors down to width 1.
            stacked = fnp.stack([x, y])
            carry, _, msgs = fold_rounds(
                rnd, initial_claim(stacked, fnp.sum(x * y), 3), tr, 3
            )
            return carry.state[:, 0], fnp.stack(msgs)

        eager = frx.tree_util.tree_map(np.asarray, run(a, b))
        jitted = frx.tree_util.tree_map(np.asarray, frx.jit(run)(a, b))
        self.assertEqual(eager[0].tobytes(), jitted[0].tobytes())
        self.assertEqual(eager[1].tobytes(), jitted[1].tobytes())


class Sha256SqueezeMarkerTest(absltest.TestCase):
    """The `zorch.sha256_squeeze` hop marker is a byte-identical drop-in for the
    plain hop, at every stream position."""

    def _at_pending_len(self, nbytes: int) -> Sha256FieldTranscript:
        """A transcript whose `pending_len` is `nbytes % 64`, reached by absorbing
        that many opaque bytes."""
        t = Sha256FieldTranscript.new(b"dom", np.uint32)
        return t.observe_bytes(fnp.zeros(nbytes, fnp.uint8))

    def test_marked_hop_matches_plain_at_every_stream_position(self) -> None:
        # `pending_len` decides both data-dependent branches the hop speculates
        # on — whether an absorb fills the 64-byte block, and whether finalize
        # needs a second padding block — and it rides as a runtime operand, so
        # ONE compiled hop must serve every residue. Sweep all 64.
        for pad in range(64):
            t = self._at_pending_len(pad)
            for nbytes in (4, 16, 32, 40):  # 1-block and 2-block squeezes
                framing = _const_u8(bytes([OP_SQUEEZE, KIND_SCALAR]))
                ref_state, ref = _squeeze_hop(t.state, framing, nbytes)
                mk_state, mk = _sha256_squeeze_zone(t.state, framing, nbytes)
                self.assertEqual(
                    np.asarray(ref).tobytes(),
                    np.asarray(mk).tobytes(),
                    f"squeezed bytes differ at pending_len={pad % 64}, {nbytes=}",
                )
                for name in ("h", "pending", "pending_len", "total_len"):
                    self.assertEqual(
                        np.asarray(getattr(ref_state, name)).tobytes(),
                        np.asarray(getattr(mk_state, name)).tobytes(),
                        f"state.{name} differs at pending_len={pad % 64}, {nbytes=}",
                    )

    def test_marker_appears_in_lowered_hlo(self) -> None:
        # Present by construction for a vendor to fuse — on both squeeze framings.
        t = Sha256FieldTranscript.new(b"dom", np.uint32)
        for fn in (lambda x: x.sample_scalar(), lambda x: x.sample(3)):
            hlo = frx.jit(fn).lower(t).as_text()
            self.assertIn(SHA256_SQUEEZE_MARKER, hlo)

    def test_marked_squeeze_survives_in_graph_consumer(self) -> None:
        # The composite is multi-output; consuming the challenge INSIDE the same
        # jit forces explicit get-tuple-elements, which is where an emitter can
        # collapse equal-shaped state leaves onto one another. Pin that the
        # returned state still matches the eager hop leaf for leaf. (Mirrors the
        # duplex marker's regression for the same failure mode.)
        t = Sha256FieldTranscript.new(b"dom", np.uint32)

        def consume(
            x: Sha256FieldTranscript,
        ) -> tuple[Sha256FieldTranscript, fnp.ndarray]:
            t2, s = x.sample(3)
            return t2, s * s  # in-graph consumer, then discarded

        ref, _ = t.sample(3)
        mk, _ = frx.jit(consume)(t)
        for a, b in zip(
            frx.tree_util.tree_leaves(ref), frx.tree_util.tree_leaves(mk), strict=True
        ):
            self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())


class SampleDistinctMarkerTest(absltest.TestCase):
    """The `zorch.sample_distinct` draw marker is a byte-identical drop-in for
    the plain rejection-sampling body, on both framings and any limb width."""

    # (block_len, count) — the last pair is deliberately not a power of two, so
    # the reduction's modulus is exercised rather than a mask.
    SHAPES = ((256, 53), (1024, 71), (100, 40))

    def _reference(
        self,
        t: Sha256FieldTranscript,
        block_len: int,
        count: int,
        limb_bytes: int,
        scalar: bool,
    ) -> tuple[Sha256FieldTranscript, list[int]]:
        """The plain loop, written straight off the transcript's own draw
        surface — no marked hop anywhere on this path."""
        out: list[int] = []
        while len(out) < count:
            t, g = t.sample_scalar() if scalar else t.sample(1)
            raw = np.asarray(g).tobytes()[:limb_bytes]
            pos = int.from_bytes(raw, "little") % block_len
            if pos not in out:
                out.append(pos)
        return t, sorted(out)

    def _assert_matches_reference(
        self, dtype: Any, limb_bytes: int, scalar: bool
    ) -> None:
        for block_len, count in self.SHAPES:
            t = Sha256FieldTranscript.new(b"dom", dtype)
            ref_t, ref = self._reference(t, block_len, count, limb_bytes, scalar)
            draw = t.sample_distinct_scalar if scalar else t.sample_distinct
            mk_t, mk = draw(block_len, count, limb_bytes=limb_bytes)

            self.assertEqual(
                np.asarray(mk).tolist(), ref, f"positions differ at {block_len=}"
            )
            # The state matters as much as the positions: the next phase
            # continues this transcript, so a hop that stopped one draw early
            # would still return the right answer here and diverge later.
            for a, b in zip(
                frx.tree_util.tree_leaves(ref_t),
                frx.tree_util.tree_leaves(mk_t),
                strict=True,
            ):
                self.assertEqual(
                    np.asarray(a).tobytes(),
                    np.asarray(b).tobytes(),
                    f"transcript state differs at {block_len=}",
                )

    def test_slice_framing_matches_the_plain_loop(self) -> None:
        self._assert_matches_reference(np.uint32, limb_bytes=4, scalar=False)

    def test_scalar_framing_matches_the_plain_loop(self) -> None:
        self._assert_matches_reference(np.uint32, limb_bytes=4, scalar=True)

    def test_narrow_limb_matches_the_plain_loop(self) -> None:
        # A limb narrower than the element is the general shape the consumer
        # case has (flock reduces the low uint64 of a 16-byte element); the
        # reduction assembles the bytes little-endian rather than bitcasting, so
        # any width exercises the same code. The 8-byte width itself needs an
        # 8-byte challenge dtype — x64, which this file does not enable — and is
        # covered end to end by the consumer's byte gates.
        self._assert_matches_reference(np.uint32, limb_bytes=2, scalar=True)

    def test_matches_the_unmarked_pcs_fold_loop(self) -> None:
        # The strongest byte-identity claim available: equality with `pcs/fold`'s
        # own unmarked loop, which is slice-framed and reduces the low uint32.
        # Against the PLAIN loop specifically — `sample_distinct_positions` now
        # routes this transcript to the marked draw, so comparing against the
        # public entry would compare the marked path to itself.
        for block_len, count in self.SHAPES:
            t = Sha256FieldTranscript.new(b"dom", np.uint32)
            ref_t, ref = _sample_distinct_positions_plain(t, block_len, count)
            mk_t, mk = t.sample_distinct(block_len, count)
            self.assertEqual(np.asarray(mk).tolist(), np.asarray(ref).tolist())
            for a, b in zip(
                frx.tree_util.tree_leaves(ref_t),
                frx.tree_util.tree_leaves(mk_t),
                strict=True,
            ):
                self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())

    def test_pcs_fold_routes_this_transcript_to_the_marked_draw(self) -> None:
        # The wiring itself: a consumer calling the generic entry gets the
        # marker, so it is not a surface only a future consumer would reach.
        t = Sha256FieldTranscript.new(b"dom", np.uint32)
        hlo = frx.jit(lambda x: sample_distinct_positions(x, 256, 8)).lower(t).as_text()
        self.assertIn(SAMPLE_DISTINCT_MARKER, hlo)

    def test_marker_appears_in_lowered_hlo(self) -> None:
        # Present by construction for a vendor to fuse — on both framings.
        t = Sha256FieldTranscript.new(b"dom", np.uint32)
        for fn in (
            lambda x: x.sample_distinct(256, 8),
            lambda x: x.sample_distinct_scalar(256, 8),
        ):
            hlo = frx.jit(fn).lower(t).as_text()
            self.assertIn(SAMPLE_DISTINCT_MARKER, hlo)

    def test_rejects_a_block_too_small_and_an_unrepresentable_limb(self) -> None:
        t = Sha256FieldTranscript.new(b"dom", np.uint32)
        with self.assertRaises(ValueError):
            t.sample_distinct(8, 9)  # more positions than the block holds
        with self.assertRaises(ValueError):
            # Wider than the 4-byte element: the draw would index past the
            # squeezed bytes, and a traced gather clamps instead of raising.
            t.sample_distinct(256, 4, limb_bytes=8)


if __name__ == "__main__":
    absltest.main()
