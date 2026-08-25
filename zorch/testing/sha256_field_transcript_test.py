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

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import HostSha256, Sha256

from zorch.byte_transcript import KIND_SCALAR, OP_SQUEEZE, ByteHashTranscript
from zorch.grind import GRIND_WINDOW, MIN_GRIND_WINDOW, grind_window_for
from zorch.sha256_field_transcript import (
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

    def test_zero_width_slice_still_absorbs_framing(self) -> None:
        b = ByteHashTranscript.new(b"dom", Sha256())
        b, b_empty = b.sample_slice(0, 4)
        b, b_next = b.sample_scalar(4)

        f = Sha256FieldTranscript.new(b"dom", np.uint32)
        f, f_empty = f.sample(0)
        f, f_next = f.sample_scalar()

        self.assertEqual(b_empty, b"")
        self.assertEqual(f_empty.shape, (0,))
        self.assertEqual(np.asarray(f_next).astype("<u4").tobytes(), b_next)

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

    def test_window_width_does_not_change_the_witness(self) -> None:
        # The window is a work/launch trade, never a protocol parameter: the
        # search scans windows in increasing order and takes the lowest hit
        # inside one, so every width returns the same counter and leaves the
        # transcript in the same state. This is what lets `grind_window_for`
        # size the window to the difficulty without touching the proof — widths
        # deliberately span both sides of the expected hit (2^bits) so a
        # multi-window search is covered, not just a one-shot one.
        root_u8 = fnp.asarray(np.frombuffer(b"root", np.uint8))
        bits = 10
        ref_witness = ref_ch = None
        for chunk in (1, 7, 64, 1 << 10, 1 << 16, None):
            t = Sha256FieldTranscript.new(b"pow", np.uint32).observe_bytes(root_u8)
            t, witness = t.grind(bits) if chunk is None else t.grind(bits, chunk=chunk)
            _, ch = t.sample_scalar()
            ch_bytes = np.asarray(ch).astype("<u4").tobytes()
            if ref_witness is None:
                ref_witness, ref_ch = int(witness), ch_bytes
                self.assertGreater(ref_witness, 0)  # a real search, not witness 0
                continue
            self.assertEqual(int(witness), ref_witness, f"chunk={chunk}")
            self.assertEqual(ch_bytes, ref_ch, f"chunk={chunk}")

    def test_grind_window_for_is_bounded_and_difficulty_sized(self) -> None:
        # Sized to the difficulty between a floor that keeps an easy search one
        # wide batch and the GRIND_WINDOW ceiling the large grinds already used.
        self.assertEqual(grind_window_for(0), MIN_GRIND_WINDOW)
        self.assertEqual(grind_window_for(40), GRIND_WINDOW)
        for bits in range(0, 40):
            w = grind_window_for(bits)
            self.assertGreaterEqual(w, MIN_GRIND_WINDOW)
            self.assertLessEqual(w, GRIND_WINDOW)
            self.assertEqual(w & (w - 1), 0, f"bits={bits} window not a power of two")
        # Monotone in the difficulty, so a harder grind never gets a narrower batch.
        widths = [grind_window_for(b) for b in range(0, 40)]
        self.assertEqual(widths, sorted(widths))
        with self.assertRaises(ValueError):
            grind_window_for(-1)

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


if __name__ == "__main__":
    absltest.main()
