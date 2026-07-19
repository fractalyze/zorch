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

from zorch.byte_transcript import ByteHashTranscript
from zorch.hash.sha256 import HostSha256, Sha256
from zorch.sha256_field_transcript import Sha256FieldTranscript


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
            vf, ok = vf.check_witness(witness, bits)
            self.assertTrue(bool(ok))
            _, vf_ch = vf.sample_scalar()
            self.assertEqual(np.asarray(vf_ch).astype("<u4").tobytes(), b_ch)
        bad = Sha256FieldTranscript.new(b"pow", np.uint32).observe_bytes(root_u8)
        _, bad_ok = bad.check_witness(int(witness) + 1, 8)
        self.assertFalse(bool(bad_ok))

    def test_grind_bits_out_of_range_rejected(self) -> None:
        # Mirrors the byte transcript: > 256 (or negative) leading-zero bits on a
        # 32-byte digest is impossible and rejected up front.
        t = Sha256FieldTranscript.new(b"pow", np.uint32)
        for bits in (-1, 257):
            with self.assertRaises(ValueError):
                t.grind(bits)
            with self.assertRaises(ValueError):
                t.check_witness(0, bits)

    def test_ghash_dtype_matches_byte_transcript_via_uint32_lanes(self) -> None:
        # flock-zorch#75: ghash <-> bytes routes through uint32 lanes to stay
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
        # bitcast-chain simplification path has regressed before (xla#259).
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
        # the GHASH dtype seam, flock-zorch#9); the point here is the transcript
        # threading, not the sumcheck math.
        from zorch.prove import fold_rounds
        from zorch.sumcheck.prover import ProductSummand, StandardRound

        a = fnp.arange(8, dtype=fnp.uint32) + 1
        b = fnp.arange(8, dtype=fnp.uint32) + 2
        rnd = StandardRound(ProductSummand(degree=2))
        tr = Sha256FieldTranscript.new(b"sc", np.uint32)

        def run(x: fnp.ndarray, y: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
            # 3 rounds folds the 2^3 stacked factors down to width 1.
            state, _, msgs = fold_rounds(rnd, fnp.stack([x, y]), tr, 3)
            return state[:, 0], fnp.stack(msgs)

        eager = frx.tree_util.tree_map(np.asarray, run(a, b))
        jitted = frx.tree_util.tree_map(np.asarray, frx.jit(run)(a, b))
        self.assertEqual(eager[0].tobytes(), jitted[0].tobytes())
        self.assertEqual(eager[1].tobytes(), jitted[1].tobytes())


if __name__ == "__main__":
    absltest.main()
